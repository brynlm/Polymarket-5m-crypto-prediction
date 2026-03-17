import asyncio
import json
import os
import time
import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
import websockets
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    asyncio.create_task(poll_btc_price())
    yield


app = FastAPI(title="Polymarket Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POLYMARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_API = "https://gamma-api.polymarket.com"
BINANCE_TICKER_URL  = "https://api.binance.com/api/v3/ticker/price"
BINANCE_KLINES_URL  = "https://api.binance.com/api/v3/klines"
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

MODEL_NAME = 'xgb_qreg_10s'

_xgb_models: Optional[dict] = None   # {0.1: Pipeline, 0.5: Pipeline, 0.9: Pipeline}
_xgb_feat_cols:   list[str]   = []
_xgb_quantiles:   list[float] = []

# Feature-engineering config — populated from metadata at load time
_n_levels:      int        = 5
_max_min_cols:  list[str]  = []
_ave_cols:      list[str]  = []
_lagged_cols:   list[str]  = []
_lags:          list[int]  = []
_roll_ave_cols: list[str]  = []
_roll_windows:  list[int]  = []
_pred_window:   int        = 5

# Derived: min rows needed in the buffer before we can predict.
# ofi = bid_vol.diff(1) → row 0 is NaN; max_lag further shifts require max_lag more rows.
# Add 1 for safety → total = max_lag + 2.
_XGB_MIN_BUFFER: int = 7   # updated dynamically in _load_model()


def _load_model() -> None:
    global _xgb_models, _xgb_feat_cols, _xgb_quantiles
    global _n_levels, _max_min_cols, _ave_cols, _lagged_cols, _lags
    global _roll_ave_cols, _roll_windows, _pred_window, _XGB_MIN_BUFFER

    model_path = os.path.join(_PROJ_ROOT, f'{MODEL_NAME}.joblib')
    meta_path  = os.path.join(_PROJ_ROOT, f'{MODEL_NAME}_meta.json')
    if not os.path.exists(model_path):
        logger.warning("xgb_quantile_reg.joblib not found — predictions disabled")
        return
    if not os.path.exists(meta_path):
        logger.warning("xgb_quantile_reg_meta.json not found — predictions disabled")
        return

    _xgb_models = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    _xgb_feat_cols  = meta['feat_cols']
    _xgb_quantiles  = meta['quantiles']
    _n_levels       = meta.get('n_levels',      _n_levels)
    _max_min_cols   = meta.get('max_min_cols',   _max_min_cols)
    _ave_cols       = meta.get('ave_cols',       _ave_cols)
    _lagged_cols    = meta.get('lagged_cols',    _lagged_cols)
    _lags           = meta.get('lags',           _lags)
    _roll_ave_cols  = meta.get('roll_ave_cols',  _roll_ave_cols)
    _roll_windows   = meta.get('roll_windows',   _roll_windows)
    _pred_window    = meta.get('pred_window',    _pred_window)
    min_for_lags  = (max(_lags) + 2) if _lags else 7
    min_for_rolls = max(_roll_windows) if _roll_windows else 1
    _XGB_MIN_BUFFER = max(min_for_lags, min_for_rolls)

    logger.info(
        f"XGB model loaded: {len(_xgb_feat_cols)} features, quantiles={_xgb_quantiles}, "
        f"lags={_lags}, roll_windows={_roll_windows}, buffer={_XGB_MIN_BUFFER}"
    )


# ──────────────────────────────────────────────────────────────
# Live state
# ──────────────────────────────────────────────────────────────

# {asset_id: {'bids': [{'price': float, 'size': float}, ...], 'asks': [...]}}
_curr_books: dict[str, dict] = {}
_feature_buffer: list[dict]  = []
_tick_buffer:    list[dict]  = []   # sub-second raw snapshots, aggregated each second
_current_second: int         = 0    # last second boundary that was aggregated

_btc_price:    float = float('nan')
_btc_open:     float = float('nan')
_interval_end: int   = 0          # Unix-second timestamp of current interval end


def _get_interval_end() -> int:
    return (int(time.time()) // 300) * 300 + 300


def _compute_base_features() -> dict | None:
    """Derive per-snapshot feature row for the XGB model from the current live order book."""
    book = next(iter(_curr_books.values()), None)
    if not book:
        return None

    bids = sorted(book['bids'], key=lambda x: -x['price'])
    asks = sorted(book['asks'], key=lambda x:  x['price'])
    if not bids or not asks:
        return None

    top_bids = bids[:_n_levels]
    top_asks = asks[:_n_levels]

    best_bid    = bids[0]['price']
    best_ask    = asks[0]['price']
    bid_vol     = sum(b['size'] for b in top_bids)
    ask_vol     = sum(a['size'] for a in top_asks)
    bid_vol_all = sum(b['size'] for b in bids)
    ask_vol_all = sum(a['size'] for a in asks)

    mid_raw  = (best_bid + best_ask) / 2
    spread   = best_ask - best_bid
    denom_t  = bid_vol + ask_vol + 1e-9
    denom_a  = bid_vol_all + ask_vol_all + 1e-9

    bid_pxs  = sum(b['price'] * b['size'] for b in bids)
    ask_pxs  = sum(a['price'] * a['size'] for a in asks)
    micro    = (best_ask * bid_vol + best_bid * ask_vol) / denom_t

    # Logit transform mid: log(1 / (1 - x)), matching the notebook's formula
    mid_clipped = float(np.clip(mid_raw, 1e-6, 1.0 - 1e-6))
    mid_logit   = float(np.log(1.0 / (1.0 - mid_clipped)))

    ts_s = int(time.time())

    row: dict = {
        'best_bid':             best_bid,
        'best_ask':             best_ask,
        'bid_vol':              bid_vol,
        'ask_vol':              ask_vol,
        'bid_vol_all':          bid_vol_all,
        'ask_vol_all':          ask_vol_all,
        'bid_n_levels':         float(len(bids)),
        'ask_n_levels':         float(len(asks)),
        'btc_price':            _btc_price,
        'btc_price_from_open':  _btc_price - _btc_open,
        'vwap':                 (bid_pxs + ask_pxs) / denom_a,
        'mid':                  mid_logit,          # logit-transformed
        'spread':               spread,
        'rel_spread':           spread / (mid_raw + 1e-9),
        'imbalance':            (bid_vol - ask_vol) / denom_t,
        'imbalance_all':        (bid_vol_all - ask_vol_all) / denom_a,
        'microprice':           micro,
        'micro_minus_mid':      micro - mid_raw,    # price-space difference
        # Temporal (raw seconds, matching notebook: index % 300 / % 86400)
        'time_in_interval':     float(ts_s % 300),
        'time_since_midnight':  float(ts_s % 86400),
    }

    for i in range(_n_levels):
        row[f'bid_size_L{i+1}'] = top_bids[i]['size'] if i < len(top_bids) else 0.0
        row[f'ask_size_L{i+1}'] = top_asks[i]['size'] if i < len(top_asks) else 0.0

    return row


def _aggregate_1s_row(ticks: list[dict]) -> dict:
    """Aggregate sub-second tick dicts into a 1-second bar, matching downsample_features_1s:
    - base columns: last tick value (.last())
    - max_min_cols: per-second max → {col}_max, min → {col}_min
    - ave_cols:     per-second mean → {col}_ave
    """
    row = dict(ticks[-1])  # last tick as base (matches groupby.last())
    for col in _max_min_cols:
        vals = [t[col] for t in ticks if col in t]
        if vals:
            row[f'{col}_max'] = max(vals)
            row[f'{col}_min'] = min(vals)
    for col in _ave_cols:
        vals = [t[col] for t in ticks if col in t]
        if vals:
            row[f'{col}_ave'] = sum(vals) / len(vals)
    return row


def _build_xgb_features(buf_df: pd.DataFrame) -> pd.DataFrame:
    """Compute OFI, lagged, and rolling-average features from buffer DataFrame."""
    df = buf_df.copy()

    df['ofi'] = df['bid_vol'].diff(1) - df['ask_vol'].diff(1)

    for lag in _lags:
        for c in _lagged_cols:
            df[f'{c}_lag{lag}'] = df[c].shift(lag)

    for w in _roll_windows:
        cols = [c for c in _roll_ave_cols if c in df.columns]
        for c in cols:
            df[f'{c}_ave{w}'] = df[c].rolling(w).mean()

    return df


def _try_predict() -> dict | None:
    """Build XGB feature vector and return quantile predictions, or None."""
    if _xgb_models is None or not _xgb_feat_cols:
        return None
    if len(_feature_buffer) < _XGB_MIN_BUFFER:
        return None

    buf_df   = pd.DataFrame(_feature_buffer[-_XGB_MIN_BUFFER:]).reset_index(drop=True)
    feat_df  = _build_xgb_features(buf_df)
    last_row = feat_df.iloc[[-1]].copy()

    for c in _xgb_feat_cols:
        if c not in last_row.columns:
            last_row[c] = float('nan')

    X = last_row[_xgb_feat_cols].values.astype(float)
    if np.any(np.isnan(X)):
        nan_cols = [_xgb_feat_cols[i] for i in range(len(_xgb_feat_cols)) if np.isnan(X[0, i])]
        logger.debug(f"Skipping prediction — NaN in: {nan_cols[:5]}")
        return None

    # Current mid in price space (inverse of notebook logit: x = 1 - exp(-logit))
    mid_logit = float(last_row['mid'].values[0])
    mid_raw   = 1.0 - float(np.exp(-mid_logit))

    result: dict = {'mid': mid_raw}
    for q, pipe in _xgb_models.items():
        return_pred = float(pipe.predict(X)[0])
        pred_logit  = mid_logit + return_pred
        pred_price  = float(np.clip(1.0 - np.exp(-pred_logit), 0.0, 1.0))
        result[f'q{int(q * 100)}'] = pred_price  # e.g. 'q10', 'q50', 'q90'

    return result


# ──────────────────────────────────────────────────────────────
# Background tasks
# ──────────────────────────────────────────────────────────────

def _fetch_btc_interval_open() -> float:
    """Return the open price of the current 5-minute BTC/USDT candle from Binance klines."""
    try:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": "BTCUSDT", "interval": "5m", "limit": 1},
            timeout=3,
        )
        return float(resp.json()[0][1])   # [0] = latest candle, [1] = open price
    except Exception as e:
        logger.warning(f"BTC interval open fetch failed: {e}")
        return float('nan')


async def poll_btc_price() -> None:
    global _btc_price, _btc_open
    while True:
        try:
            resp = requests.get(BINANCE_TICKER_URL, params={"symbol": "BTCUSDT"}, timeout=3)
            price = float(resp.json()['price'])
            _btc_price = price
            if np.isnan(_btc_open):
                # Fetch the actual candle open so btc_price_from_open matches training
                _btc_open = _fetch_btc_interval_open()
        except Exception as e:
            logger.warning(f"BTC price poll failed: {e}")
        await asyncio.sleep(2)


# ──────────────────────────────────────────────────────────────
# WebSocket helpers
# ──────────────────────────────────────────────────────────────

active_clients: set[WebSocket] = set()
stream_task: Optional[asyncio.Task] = None
current_token_ids: list[str] = []


async def broadcast(message: str) -> None:
    dead = set()
    for client in active_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead.add(client)
    active_clients.difference_update(dead)


def _apply_book_update(asset_id: str, msg: dict) -> None:
    """Update _curr_books from a Polymarket book or price_change message."""
    global _curr_books
    event_type = msg.get('event_type')

    if event_type == 'book':
        _curr_books[asset_id] = {
            'bids': [{'price': float(e['price']), 'size': float(e['size'])} for e in msg.get('bids', [])],
            'asks': [{'price': float(e['price']), 'size': float(e['size'])} for e in msg.get('asks', [])],
        }

    elif event_type == 'price_change' and asset_id in _curr_books:
        book = _curr_books[asset_id]
        changes = msg.get('changes') or msg.get('price_changes') or []
        for ch in changes:
            price = float(ch['price'])
            size  = float(ch['size'])
            side  = 'bids' if ch['side'] == 'BUY' else 'asks'
            arr   = book[side]
            idx   = next((i for i, e in enumerate(arr) if e['price'] == price), -1)
            if size == 0:
                if idx >= 0:
                    arr.pop(idx)
            elif idx >= 0:
                arr[idx]['size'] = size
            else:
                arr.append({'price': price, 'size': size})


async def run_polymarket_stream(token_ids: list[str]) -> None:
    global _curr_books, _feature_buffer, _tick_buffer, _current_second, _btc_open, _interval_end
    _curr_books.clear()
    _feature_buffer.clear()
    _tick_buffer.clear()

    logger.info(f"Starting Polymarket stream for {len(token_ids)} tokens")
    try:
        async with websockets.connect(POLYMARKET_WS, ping_interval=5) as ws:
            await ws.send(json.dumps({
                "action": "subscribe",
                "type": "market",
                "assets_ids": token_ids,
                "custom_feature_enabled": True,
            }))

            while True:
                raw = await ws.recv()
                await broadcast(raw)

                # Track book state
                messages = json.loads(raw)
                if not isinstance(messages, list):
                    messages = [messages]

                for m in messages:
                    asset_id = m.get('asset_id')
                    if asset_id:
                        _apply_book_update(asset_id, m)

                now = time.time()
                if not _curr_books:
                    continue

                # Accumulate every sub-second tick
                tick = _compute_base_features()
                if tick is None:
                    continue
                _tick_buffer.append(tick)

                # At each new second boundary: aggregate ticks → 1s row
                this_second = int(now)
                if this_second <= _current_second:
                    continue
                _current_second = this_second

                # Reset buffer on interval rollover
                interval_end = _get_interval_end()
                if interval_end != _interval_end:
                    _interval_end = interval_end
                    _btc_open = _fetch_btc_interval_open()
                    _feature_buffer.clear()

                row_1s = _aggregate_1s_row(_tick_buffer)
                _tick_buffer.clear()

                _feature_buffer.append(row_1s)
                if len(_feature_buffer) > _XGB_MIN_BUFFER * 4:
                    _feature_buffer = _feature_buffer[-_XGB_MIN_BUFFER * 4:]

                preds = _try_predict()
                if preds:
                    await broadcast(json.dumps({
                        'event_type':  'prediction',
                        'timestamp':   int(now * 1000),
                        'predictions': preds,
                    }))

    except asyncio.CancelledError:
        logger.info("Stream cancelled")
    except Exception as e:
        logger.error(f"Stream error: {e}")
        await broadcast(json.dumps({"event_type": "error", "message": str(e)}))


# ──────────────────────────────────────────────────────────────
# App lifecycle
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# REST endpoints
# ──────────────────────────────────────────────────────────────

def get_token_ids(slug: str) -> tuple[list[str], dict]:
    try:
        resp = requests.get(f"{GAMMA_API}/markets", params={"slug": slug}, timeout=5).json()
        if not resp:
            return [], {}
        market = resp[0]
        return json.loads(market["clobTokenIds"]), market
    except Exception as e:
        logger.error(f"Error fetching market {slug}: {e}")
        return [], {}


@app.get("/api/market/{slug}")
def get_market(slug: str):
    token_ids, market = get_token_ids(slug)
    if not token_ids:
        return {"error": "Market not found"}
    return {
        "slug":      slug,
        "token_ids": token_ids,
        "question":  market.get("question"),
        "end_date":  market.get("endDate"),
    }


@app.get("/api/markets/active")
def get_active_markets():
    current_time = int(time.time())
    markets = []
    for interval, label in [(300, "5m"), (900, "15m")]:
        slug_time = (current_time // interval) * interval
        slug = f"btc-updown-{label}-{slug_time}"
        token_ids, market = get_token_ids(slug)
        if token_ids:
            markets.append({
                "slug":      slug,
                "label":     f"BTC {label}",
                "token_ids": token_ids,
                "question":  market.get("question"),
            })
    return markets


# ──────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global stream_task, current_token_ids
    await websocket.accept()
    active_clients.add(websocket)
    logger.info(f"Client connected. Total: {len(active_clients)}")

    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)

            if msg.get("action") == "subscribe":
                slug = msg.get("slug", "").strip()
                if not slug:
                    continue

                token_ids, _ = get_token_ids(slug)
                if not token_ids:
                    await websocket.send_text(json.dumps({
                        "event_type": "error",
                        "message":    f"Market not found: {slug}",
                    }))
                    continue

                if token_ids != current_token_ids:
                    if stream_task and not stream_task.done():
                        stream_task.cancel()
                        await asyncio.sleep(0.1)
                    current_token_ids = token_ids
                    stream_task = asyncio.create_task(run_polymarket_stream([token_ids[0]]))
                elif not stream_task or stream_task.done():
                    stream_task = asyncio.create_task(run_polymarket_stream([token_ids[0]]))

                await websocket.send_text(json.dumps({
                    "event_type": "subscribed",
                    "slug":       slug,
                    "token_ids":  token_ids,
                }))

    except WebSocketDisconnect:
        active_clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(active_clients)}")
        if not active_clients and stream_task and not stream_task.done():
            stream_task.cancel()
            current_token_ids = []
