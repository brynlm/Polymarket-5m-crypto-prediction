from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import asyncio
import asyncpg
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timezone
from xgboost import XGBRegressor
from dotenv import load_dotenv

load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL")

# ── Feature config (must match main.py / ml_training_pipeline.py) ─────────────
N_LEVELS      = 5
MAX_MIN_COLS  = ['best_bid', 'best_ask', 'mid', 'ask_vol_all', 'bid_vol_all']
AVE_COLS      = ['spread', 'rel_spread']
LAGGED_COLS   = ['best_bid', 'best_ask', 'mid',
                 'best_bid_max', 'best_ask_max', 'mid_max',
                 'best_bid_min', 'best_ask_min', 'mid_min', 'ofi']
LAGS          = [1, 2, 3, 4, 5]
ROLL_AVE_COLS = ['mid', 'spread', 'vwap', 'best_bid', 'best_ask', 'rel_spread']
ROLL_WINDOWS  = [3, 4, 5, 10]
PRED_WINDOW   = 5
QUANTILES     = [0.1, 0.5, 0.9]
MODEL_NAME    = 'xgb_qreg_5s'
MARKETS       = ['UP', 'DOWN']


async def load_from_db() -> dict[str, pd.DataFrame]:
    """Load BTC market data from market_features, return one DataFrame per market keyed by market label."""
    pool = await asyncpg.create_pool(DATABASE_URL, ssl='require')
    async with pool.acquire() as conn:
        records = await conn.fetch(
            """
            SELECT timestamp, market,
                   best_bid, best_ask,
                   bid_size_l1, bid_size_l2, bid_size_l3, bid_size_l4, bid_size_l5,
                   ask_size_l1, ask_size_l2, ask_size_l3, ask_size_l4, ask_size_l5,
                   bid_vol_all, ask_vol_all, bid_n_levels, ask_n_levels, vwap
            FROM market_features
            WHERE instrument = 'btc'
            ORDER BY timestamp
            """
        )
    await pool.close()

    df = pd.DataFrame([dict(r) for r in records])
    df['ts_s'] = df['timestamp'].astype(int)
    df = df.set_index('ts_s').drop(columns=['timestamp']).sort_index()

    return {mkt: df[df['market'] == mkt].drop(columns=['market']) for mkt in MARKETS}


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features derivable from stored DB primitives."""
    feat = df.copy()

    feat['mid']        = (feat['best_bid'] + feat['best_ask']) / 2
    feat['spread']     = feat['best_ask'] - feat['best_bid']
    feat['rel_spread'] = feat['spread'] / (feat['mid'] + 1e-9)

    # Top-5 volumes (NULL level = book had fewer than 5 levels, treat as 0)
    bid_l = [f'bid_size_l{i+1}' for i in range(N_LEVELS)]
    ask_l = [f'ask_size_l{i+1}' for i in range(N_LEVELS)]
    feat['bid_vol'] = feat[bid_l].fillna(0).sum(axis=1)
    feat['ask_vol'] = feat[ask_l].fillna(0).sum(axis=1)

    denom_top = feat['bid_vol']     + feat['ask_vol']     + 1e-9
    denom_all = feat['bid_vol_all'] + feat['ask_vol_all'] + 1e-9
    feat['imbalance']       = (feat['bid_vol']     - feat['ask_vol'])     / denom_top
    feat['imbalance_all']   = (feat['bid_vol_all'] - feat['ask_vol_all']) / denom_all
    feat['microprice']      = (feat['best_ask'] * feat['bid_vol'] + feat['best_bid'] * feat['ask_vol']) / denom_top
    feat['micro_minus_mid'] = feat['microprice'] - feat['mid']

    # Postgres folds unquoted identifiers to lowercase, so market_features
    # returns bid_size_l1../ask_size_l1.. — rename to uppercase L to match
    # backend/main.py's live feature-row convention (row[f'bid_size_L{i+1}']).
    feat = feat.rename(columns={
        **{f'bid_size_l{i+1}': f'bid_size_L{i+1}' for i in range(N_LEVELS)},
        **{f'ask_size_l{i+1}': f'ask_size_L{i+1}' for i in range(N_LEVELS)},
    })
    return feat


def add_1s_aliases(feat: pd.DataFrame) -> pd.DataFrame:
    """DB data is one snapshot per second: max/min/ave over the interval equal the
    value itself. market_features does have intra_max_bid/intra_min_ask (real
    intra-second extremes from a separate best_bid_ask push feed), but those are
    deliberately not used here — that feed isn't practical to reproduce reliably
    at serving time (backend/main.py only tracks book/price_change events), so
    self-aliasing on both sides keeps train/serve consistent even though it
    means these columns carry no real intra-second signal. Must match
    backend/main.py's _aggregate_1s_row exactly.
    """
    for col in MAX_MIN_COLS:
        feat[f'{col}_max'] = feat[col]
        feat[f'{col}_min'] = feat[col]
    for col in AVE_COLS:
        feat[f'{col}_ave'] = feat[col]
    return feat


def _logit(x):
    return np.log(1 / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def transform_features(feat: pd.DataFrame) -> pd.DataFrame:
    feat['ofi']    = feat['bid_vol'].diff(1) - feat['ask_vol'].diff(1)
    feat['return'] = feat['mid'].shift(-PRED_WINDOW) - feat['mid']

    for lag in LAGS:
        feat = pd.concat(
            [feat, feat[LAGGED_COLS].shift(lag).rename(columns=lambda c: f'{c}_lag{lag}')],
            axis=1
        )
    for w in ROLL_WINDOWS:
        feat = pd.concat(
            [feat, feat[ROLL_AVE_COLS].rolling(w).mean().rename(columns=lambda c: f'{c}_ave{w}')],
            axis=1
        )
    feat['time_in_interval']    = feat.index % 300
    feat['time_since_midnight'] = feat.index % 86400
    return feat


def prepare_market(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline for a single market's raw DB rows."""
    feat = derive_features(df)
    feat['mid'] = _logit(feat['mid'])
    feat = add_1s_aliases(feat)
    feat = transform_features(feat)
    return feat


def train_and_save() -> dict:
    """Full pipeline: load, engineer features, train, save, return diagnostics.

    Shared entry point for both manual runs (python3 retrain.py) and the
    scheduled cron job (scheduled_retrain.py), so there's exactly one code
    path to keep correct.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data from DB...")
    raw_by_market = asyncio.run(load_from_db())
    for mkt, df in raw_by_market.items():
        print(f"  {mkt}: {len(df):,} rows")

    # ── Feature engineering per market ────────────────────────────────────────
    feat_by_market = {mkt: prepare_market(df) for mkt, df in raw_by_market.items()}

    # ── Join both markets into a single wide row per second ───────────────────
    # Features from each market are suffixed _up / _down.
    # The return target for each market is kept as return_up / return_down.
    dfs = {}
    for mkt in MARKETS:
        suffix = f'_{mkt.lower()}'
        dfs[mkt] = feat_by_market[mkt].add_suffix(suffix)

    combined = dfs[MARKETS[0]].join(dfs[MARKETS[1]], how='inner')
    combined = combined.dropna()
    print(f"\n{len(combined):,} rows in combined (inner join, after dropna)")

    # Shared feature matrix (all columns except the two return targets)
    target_cols = [f'return_{mkt.lower()}' for mkt in MARKETS]
    feat_cols   = [c for c in combined.columns if c not in target_cols]
    X = combined[feat_cols].values

    # Regression guard: column *names* must stay identical to the previously
    # committed contract even though this pipeline sources different data —
    # catches e.g. forgetting to drop intra_max_bid/intra_min_ask before they
    # leak into feat_cols as columns backend/main.py can never supply.
    _prev_meta_path = f'{MODEL_NAME}_meta.json'
    if os.path.exists(_prev_meta_path):
        with open(_prev_meta_path) as f:
            _prev_feat_cols = set(json.load(f)['feat_cols'])
        if _prev_feat_cols != set(feat_cols):
            missing = _prev_feat_cols - set(feat_cols)
            extra   = set(feat_cols) - _prev_feat_cols
            raise ValueError(
                f"feat_cols mismatch vs previous {_prev_meta_path}: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    # ── Train 2 × 3 quantile models (one set per market) ──────────────────────
    # All models share the same combined feature set X.
    models = {
        mkt: {
            q: Pipeline([
                ('scaler', MinMaxScaler()),
                ('model',  XGBRegressor(objective='reg:quantileerror', quantile_alpha=q,
                                        n_estimators=200, random_state=42)),
            ])
            for q in QUANTILES
        }
        for mkt in MARKETS
    }

    tscv = TimeSeriesSplit(n_splits=5)

    for mkt in MARKETS:
        y = combined[f'return_{mkt.lower()}'].values
        print(f'\n--- {mkt}: TimeSeriesSplit CV (5 folds) ---')
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            pb_losses = {}
            for q in QUANTILES:
                models[mkt][q].fit(X[tr_idx], y[tr_idx])
                y_pred = models[mkt][q].predict(X[val_idx])
                err = y[val_idx] - y_pred
                pb_losses[q] = float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))
            print(f'  Fold {fold+1}: PBL_10={pb_losses[0.1]:.6f}  PBL_50={pb_losses[0.5]:.6f}  '
                  f'PBL_90={pb_losses[0.9]:.6f}  (n_train={len(tr_idx):,}  n_val={len(val_idx):,})')

    # ── Final fit + diagnostics per market ────────────────────────────────────
    diagnostics: dict = {'n_combined_rows': len(combined), 'markets': {}}
    for mkt in MARKETS:
        y = combined[f'return_{mkt.lower()}'].values
        mid_col = f'mid_{mkt.lower()}'

        pred_returns   = {}
        pred_quantiles = {}
        for q in QUANTILES:
            models[mkt][q].fit(X, y)
            pred_returns[q]   = models[mkt][q].predict(X)
            pred_quantiles[q] = (combined[mid_col] + pred_returns[q]).shift(PRED_WINDOW)
            pred_quantiles[q] = pred_quantiles[q].apply(lambda x: 1 / (1 + np.exp(-x)))

        actual = combined[mid_col].apply(lambda x: 1 / (1 + np.exp(-x)))
        s = PRED_WINDOW
        within_80 = (
            (actual.values[s:] >= pred_quantiles[0.1].values[s:]) &
            (actual.values[s:] <= pred_quantiles[0.9].values[s:])
        )
        coverage_80      = float(np.mean(within_80))
        coverage_90      = float(np.mean(actual.values[s:] <= pred_quantiles[0.9].values[s:]))
        coverage_10      = float(np.mean(actual.values[s:] <= pred_quantiles[0.1].values[s:]))
        directional_acc  = float(np.mean(np.sign(y) == np.sign(pred_returns[0.5])))

        print(f"\n{mkt} diagnostics:")
        print(f"  80% interval coverage:    {coverage_80:.4f}")
        print(f"  90th quantile coverage:   {coverage_90:.4f}")
        print(f"  10th quantile coverage:   {coverage_10:.4f}")
        print(f"  Directional acc (median): {directional_acc:.4f}")

        diagnostics['markets'][mkt] = {
            'n_rows':          int(len(raw_by_market[mkt])),
            'coverage_80':     coverage_80,
            'coverage_90':     coverage_90,
            'coverage_10':     coverage_10,
            'directional_acc': directional_acc,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    trained_at = datetime.now(timezone.utc).isoformat()
    joblib.dump(models, MODEL_NAME + '.joblib')
    meta = {
        'feat_cols':     feat_cols,
        'markets':       MARKETS,
        'quantiles':     QUANTILES,
        'pred_window':   PRED_WINDOW,
        'n_features':    len(feat_cols),
        'n_levels':      N_LEVELS,
        'max_min_cols':  MAX_MIN_COLS,
        'ave_cols':      AVE_COLS,
        'lagged_cols':   LAGGED_COLS,
        'lags':          LAGS,
        'roll_ave_cols': ROLL_AVE_COLS,
        'roll_windows':  ROLL_WINDOWS,
        'trained_at':    trained_at,
    }
    with open(f'{MODEL_NAME}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved → {MODEL_NAME}.joblib + {MODEL_NAME}_meta.json  ({len(feat_cols)} features)")

    diagnostics['trained_at'] = trained_at
    return diagnostics


if __name__ == "__main__":
    train_and_save()
