"""
feature_pipeline.py

End-to-end pipeline:
  1. Load raw orderbook snapshots from raw_book_data/
  2. Extract per-timestamp features (best prices, depth, imbalance, microprice)
  3. Build targets on raw [0,1] tick data (before downsampling and logit transform):
       target_logit_change : logit(mid_{t+10s}) − logit(mid_t)
       target_mid_anchor   : raw [0,1] mid at t (for plot reconstruction only)
  4. Apply logit transform to price features (mid, best_bid, best_ask, microprice, vwap)
  5. Downsample: feature columns via mean (or last), target columns always via last
  6. Add time-lagged / rolling features (now in logit space for price cols)
  7. Train a model with TimeSeriesSplit CV
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────
# 1. Load raw book data
# ──────────────────────────────────────────────────────────────

def load_raw_books(data_dir='raw_book_data') -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, 'book-*.pkl')))
    if not paths:
        raise FileNotFoundError(f'No book-*.pkl files found in {data_dir}')
    print(f'Loading {len(paths)} file(s)...')
    return pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)


def _file_ts_from_path(path: str) -> int:
    """Extract the Unix-second timestamp from a book-*.pkl filename."""
    return int(os.path.basename(path).rsplit('-', 1)[-1].replace('.pkl', ''))


# ──────────────────────────────────────────────────────────────
# 2. Feature extraction
# ──────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """
    Convert long-format book data into a wide per-timestamp feature matrix.
    Features per snapshot:
      - best_bid, best_ask, mid, spread
      - bid/ask volume for top N levels and all levels
      - orderbook imbalance (top N and all levels)
      - microprice (volume-weighted mid)
      - per-level sizes for top N bid/ask levels
    """
    bids = df[df['order_type'] == 'bid']
    asks = df[df['order_type'] == 'ask']

    # Best prices per timestamp
    best_bid = bids.groupby('timestamp')['price'].max().rename('best_bid')
    best_ask = asks.groupby('timestamp')['price'].min().rename('best_ask')

    # Rank levels within each snapshot (0 = best)
    bids_s = bids.sort_values(['timestamp', 'price'], ascending=[True, False]).copy()
    bids_s['_lvl'] = bids_s.groupby('timestamp').cumcount()

    asks_s = asks.sort_values(['timestamp', 'price'], ascending=[True, True]).copy()
    asks_s['_lvl'] = asks_s.groupby('timestamp').cumcount()

    top_bids = bids_s[bids_s['_lvl'] < n_levels]
    top_asks = asks_s[asks_s['_lvl'] < n_levels]

    # Aggregate volumes
    bid_vol     = top_bids.groupby('timestamp')['size'].sum().rename('bid_vol')
    ask_vol     = top_asks.groupby('timestamp')['size'].sum().rename('ask_vol')
    bid_vol_all = bids.groupby('timestamp')['size'].sum().rename('bid_vol_all')
    ask_vol_all = asks.groupby('timestamp')['size'].sum().rename('ask_vol_all')
    bid_n_lvl   = bids.groupby('timestamp').size().rename('bid_n_levels')
    ask_n_lvl   = asks.groupby('timestamp').size().rename('ask_n_levels')

    # Per-level size columns  (unstack is faster than pivot_table for this shape)
    bid_piv = (
        top_bids.set_index(['timestamp', '_lvl'])['size']
        .unstack('_lvl')
        .reindex(columns=range(n_levels), fill_value=0.0)
    )
    bid_piv.columns = [f'bid_size_L{i+1}' for i in range(n_levels)]

    ask_piv = (
        top_asks.set_index(['timestamp', '_lvl'])['size']
        .unstack('_lvl')
        .reindex(columns=range(n_levels), fill_value=0.0)
    )
    ask_piv.columns = [f'ask_size_L{i+1}' for i in range(n_levels)]

    # BTC spot price per timestamp (consistent within a snapshot, take last)
    btc_price = df.groupby('timestamp')['btc_price'].last().rename('btc_price') \
        if 'btc_price' in df.columns else pd.Series(dtype='float64', name='btc_price')

    # Volume-weighted average price (VWAP) across all visible bid + ask levels
    bid_pxs = (bids['price'] * bids['size']).groupby(bids['timestamp']).sum()
    ask_pxs = (asks['price'] * asks['size']).groupby(asks['timestamp']).sum()
    vwap = ((bid_pxs + ask_pxs) / (bid_vol_all + ask_vol_all + 1e-9)).rename('vwap')

    feat = pd.concat(
        [best_bid, best_ask, bid_vol, ask_vol, bid_vol_all, ask_vol_all,
         bid_n_lvl, ask_n_lvl, bid_piv, ask_piv, btc_price, vwap],
        axis=1
    ).sort_index()

    # Derived features
    feat['mid']            = (feat['best_bid'] + feat['best_ask']) / 2
    feat['spread']         = feat['best_ask'] - feat['best_bid']
    feat['rel_spread']     = feat['spread'] / (feat['mid'] + 1e-9)

    denom_top = feat['bid_vol'] + feat['ask_vol'] + 1e-9
    denom_all = feat['bid_vol_all'] + feat['ask_vol_all'] + 1e-9

    feat['imbalance']      = (feat['bid_vol'] - feat['ask_vol']) / denom_top
    feat['imbalance_all']  = (feat['bid_vol_all'] - feat['ask_vol_all']) / denom_all

    # Microprice: volume-weighted mid skewed toward the larger side
    feat['microprice']     = (
        feat['best_ask'] * feat['bid_vol'] + feat['best_bid'] * feat['ask_vol']
    ) / denom_top
    feat['micro_minus_mid'] = feat['microprice'] - feat['mid']

    return feat


# ──────────────────────────────────────────────────────────────
# 3. Downsampling + interval/temporal features
# ──────────────────────────────────────────────────────────────

def filter_to_last_per_second(feat: pd.DataFrame, interval_ms: int = 1000, mean=False) -> pd.DataFrame:
    """Keep only the last snapshot within each time interval.
    Index remains in milliseconds, floored to the interval boundary.

    Parameters
    ----------
    interval_ms : interval width in milliseconds (default 1000 = 1 second)
    mean : If True, the mean is used to downsample points in the same interval instead of the last element.
    The default behaviour is False.
    """
    bins = feat.index // interval_ms
    result = feat.groupby(bins).last() if not mean else feat.groupby(bins).mean()
    result.index = result.index * interval_ms  # restore ms scale
    return result


def downsample_features(feat: pd.DataFrame, interval_ms: int = 1000, mean: bool = False) -> pd.DataFrame:
    """
    Downsample a feature DataFrame to one row per interval with richer aggregation
    for price columns.

    Price columns (_LOGIT_PRICE_COLS present in feat):
      - Point value : last tick in the interval (always, regardless of `mean`)
      - Extra cols  : <col>_imax, <col>_imin, <col>_istd — interval max / min / std

    All other columns:
      - last tick when mean=False, interval mean when mean=True

    Target columns should NOT be passed here; downsample them separately with
    filter_to_last_per_second(..., mean=False).
    """
    bins = feat.index // interval_ms

    price_cols = [c for c in _LOGIT_PRICE_COLS if c in feat.columns]
    other_cols = [c for c in feat.columns if c not in price_cols]

    parts = []
    if price_cols:
        grp = feat[price_cols].groupby(bins)
        parts += [
            grp.last(),
            grp.max().add_suffix('_imax'),
            grp.min().add_suffix('_imin'),
            grp.std().add_suffix('_istd'),
        ]
    if other_cols:
        grp = feat[other_cols].groupby(bins)
        parts.append(grp.mean() if mean else grp.last())

    result = pd.concat(parts, axis=1)
    result.index = result.index * interval_ms
    return result


def add_interval_features(feat: pd.DataFrame, file_ts_s: int) -> pd.DataFrame:
    """
    Add temporal and market-context features for a single 5-min file.

    Parameters
    ----------
    feat        : feature DataFrame with ms timestamp index
    file_ts_s   : Unix-second timestamp extracted from the filename
                  (represents the end of the 5-min interval)
    """
    interval_start_ms = (file_ts_s - 300) * 1000  # 5 min = 300 s

    # Seconds elapsed since the start of this 5-min interval  [0, 300)
    feat['time_in_interval_s'] = (feat.index - interval_start_ms) / 1000.0

    # Time of day in seconds since midnight  [0, 86400)
    feat['time_of_day_s'] = (feat.index // 1000) % 86400

    # BTC price change from the opening (earliest) BTC price in this interval
    if 'btc_price' in feat.columns:
        feat['btc_price_from_open'] = feat['btc_price'] - feat['btc_price'].iloc[0]

    return feat


# ──────────────────────────────────────────────────────────────
# 4. Logit transform for price features
# ──────────────────────────────────────────────────────────────

# Columns representing [0,1] prediction-market prices to be logit-transformed.
# btc_price is excluded — it is a USD spot price, not a probability.
_LOGIT_PRICE_COLS = ['mid', 'best_bid', 'best_ask', 'microprice', 'vwap']


def apply_logit_to_prices(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Transform [0,1] price columns to logit (log-odds) space in-place, then
    recompute the derived features that reference absolute price levels so they
    remain consistent.

    IMPORTANT: call this AFTER add_target (which needs raw [0,1] prices) and
               BEFORE downsampling / add_time_features.
    """
    for col in _LOGIT_PRICE_COLS:
        if col in feat.columns:
            feat[col] = _logit(feat[col].values)

    # Recompute derived features now that base prices are in logit space
    if 'best_bid' in feat.columns and 'best_ask' in feat.columns:
        feat['spread']     = feat['best_ask'] - feat['best_bid']
        feat['rel_spread'] = feat['spread'] / (feat['mid'].abs() + _LOGIT_EPS)
    if 'microprice' in feat.columns and 'mid' in feat.columns:
        feat['micro_minus_mid'] = feat['microprice'] - feat['mid']

    return feat


# ──────────────────────────────────────────────────────────────
# 5. Time-lagged & rolling features
# ──────────────────────────────────────────────────────────────

# Columns to lag
_LAG_COLS    = ['mid', 'spread', 'rel_spread', 'imbalance', 'imbalance_all',
                'microprice', 'micro_minus_mid', 'bid_vol', 'ask_vol',
                'btc_price', 'btc_price_from_open', 'vwap', 'ofi']
_LAGS        = list(range(5)) #[1, 2, 5, 10, 20]
_DIFF_COLS   = ['mid', 'spread', 'imbalance']
_ROLL_COLS   = ['mid', 'imbalance', 'spread']
_ROLL_WINS   = [5, 10]


def add_time_features(feat: pd.DataFrame) -> pd.DataFrame:
    new_cols: dict[str, pd.Series] = {}

    # Order flow imbalance: signed change in net top-N bid pressure per interval
    new_cols['ofi'] = feat['bid_vol'].diff(1) - feat['ask_vol'].diff(1)

    for col in _LAG_COLS:
        # Source may be an already-existing column or one just computed above
        src = feat[col] if col in feat.columns else new_cols.get(col)
        if src is None:
            continue
        for lag in _LAGS:
            new_cols[f'{col}_lag{lag}'] = src.shift(lag)

    for col in _DIFF_COLS:
        new_cols[f'{col}_diff1'] = feat[col].diff(1)
        new_cols[f'{col}_diff5'] = feat[col].diff(5)

    for col in _ROLL_COLS:
        for w in _ROLL_WINS:
            new_cols[f'{col}_rmean{w}'] = feat[col].rolling(w).mean()
            new_cols[f'{col}_rstd{w}']  = feat[col].rolling(w).std()

    return pd.concat([feat, pd.DataFrame(new_cols, index=feat.index)], axis=1)


# ──────────────────────────────────────────────────────────────
# 5. Build targets on raw [0,1] tick data (BEFORE logit transform + downsampling)
# ──────────────────────────────────────────────────────────────

_LOGIT_EPS = 1e-6  # clip prices away from 0/1 before logit


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(x / (1 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def add_target(feat: pd.DataFrame, horizon_ms: int = 10000) -> pd.DataFrame:
    """
    Compute targets on raw [0,1] tick data, before any logit transform or
    downsampling, so neither corrupts the labels.

    IMPORTANT: call this before apply_logit_to_prices and
               filter_to_last_per_second.

    Columns added:
      target_logit_change : logit(mid_{t+horizon}) − logit(mid_t)
      target_mid_anchor   : raw [0,1] mid at t — not a model target, used only
                            for price-band reconstruction in plots
    """
    ts_arr  = feat.index.values
    mid_arr = feat['mid'].values
    logit_mid = _logit(mid_arr)

    logit_change = np.full(len(ts_arr), np.nan)
    volatility   = np.full(len(ts_arr), np.nan)

    for i, t in enumerate(ts_arr):
        j = np.searchsorted(ts_arr, t + horizon_ms)
        if j < len(ts_arr):
            logit_change[i] = logit_mid[j] - logit_mid[i]

        # Realized vol: std of logit changes over [t, t+horizon]
        mask = (ts_arr >= t) & (ts_arr <= t + horizon_ms)
        window_logits = logit_mid[mask]
        if len(window_logits) > 1:
            volatility[i] = float(np.std(np.diff(window_logits)))

    feat['target_logit_change'] = logit_change
    # Anchor: raw [0,1] mid — downsampled with 'last' alongside targets so it
    # stays consistent with the logit-change base for reconstruction.
    feat['target_mid_anchor']   = mid_arr
    return feat


# ──────────────────────────────────────────────────────────────
# 6. Training & evaluation
# ──────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss for a single quantile alpha ∈ (0, 1)."""
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))



class QuantileGBR:
    """
    Trains one GradientBoostingRegressor(loss='quantile', alpha=q) per quantile.
    Accepts a 1-D target and produces (n_samples, n_quantiles) predictions.

    Parameters
    ----------
    quantiles     : tuple of quantile levels in (0, 1), e.g. (0.1, 0.25, 0.5, 0.75, 0.9)
    n_estimators  : trees per quantile model
    max_depth     : max tree depth
    learning_rate : GBR learning rate
    """
    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ):
        self.quantiles     = quantiles
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileGBR':
        y = np.asarray(y).ravel()
        self.estimators_ = [
            GradientBoostingRegressor(
                loss='quantile', alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            ).fit(X, y)
            for q in self.quantiles
        ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (n_samples, n_quantiles)."""
        return np.column_stack([e.predict(X) for e in self.estimators_])


def baseline_metrics(y_true: np.ndarray, target_name: str = '') -> None:
    """Naive zero baseline (predict no logit change)."""
    y_zero = np.zeros_like(y_true)
    mae  = mean_absolute_error(y_true, y_zero)
    rmse = _rmse(y_true, y_zero)
    r2   = r2_score(y_true, y_zero)
    label = f' [{target_name}]' if target_name else ''
    print(f'  Baseline (zero){label}: MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}')


def train_and_evaluate(
    feat: pd.DataFrame,
    target_cols: list[str] | None = None,
    model: QuantileGBR | None = None,
    n_splits: int = 5,
) -> tuple[Pipeline, list[str], list[str]]:
    """
    Train a QuantileGBR using TimeSeriesSplit CV.

    The model fits on a single 1-D target (target_logit_change) and predicts
    one value per quantile level.  CV reports pinball loss per quantile and MAE
    for the median quantile.

    Parameters
    ----------
    feat        : feature DataFrame (output of build_pipeline)
    target_cols : single-element list, defaults to ['target_logit_change']
    model       : QuantileGBR instance; constructed with defaults if None
    n_splits    : number of TimeSeriesSplit folds

    Returns
    -------
    pipeline    : fitted Pipeline(MinMaxScaler → QuantileGBR)
    feat_cols   : feature column names used
    target_cols : target column names (always ['target_logit_change'])
    """
    all_target_cols = [c for c in feat.columns if c.startswith('target_')]
    if target_cols is None:
        target_cols = [c for c in all_target_cols
                       if c not in ('target_mid_anchor',)]

    feat_cols = [c for c in feat.columns if c not in set(all_target_cols)]

    clean = feat[feat_cols + target_cols].dropna()
    X = clean[feat_cols].values
    y = clean[target_cols].values.ravel()   # 1-D

    print(f'\nDataset: {len(clean):,} rows × {len(feat_cols)} features')
    print(f'Target : {target_cols}')
    print(f'\n--- Baseline (zero logit-change prediction) ---')
    baseline_metrics(y, target_cols[0])

    if model is None:
        model = QuantileGBR()

    pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
    tscv     = TimeSeriesSplit(n_splits=n_splits)
    quantiles = np.asarray(model.quantiles)
    median_idx = int(np.argmin(np.abs(quantiles - 0.5)))

    print(f'\n--- TimeSeriesSplit CV ({n_splits} folds) ---')
    print(f'Quantiles: {list(quantiles)}')

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        pipeline.fit(X[tr_idx], y[tr_idx])
        preds = pipeline.predict(X[val_idx])          # (n_val, n_quantiles)
        y_val = y[val_idx]

        pinball = [_pinball_loss(y_val, preds[:, i], q)
                   for i, q in enumerate(quantiles)]
        mae_med = mean_absolute_error(y_val, preds[:, median_idx])

        pb_str = '  '.join(f'q{q:.2f}={pb:.5f}' for q, pb in zip(quantiles, pinball))
        print(f'  Fold {fold+1}: MedianMAE={mae_med:.6f}  Pinball [{pb_str}]  '
              f'(n_train={len(tr_idx):,}  n_val={len(val_idx):,})')

    # Refit on full dataset
    pipeline.fit(X, y)
    return pipeline, feat_cols, target_cols


# ──────────────────────────────────────────────────────────────
# 7. Prediction plot for a single file
# ──────────────────────────────────────────────────────────────

def plot_predictions(
    pkl_path: str,
    pipeline: Pipeline,
    feat_cols: list[str],
    target_cols: list[str],
    n_levels: int = 5,
    horizon_ms: int = 10000,
    downsample_ms: int = 1000,
    mean: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Run the feature pipeline on a single .pkl file and produce a three-panel figure:

      Top    : Price band plot — current mid, actual future mid, predicted median,
               and shaded bands for each symmetric quantile pair (e.g. 10%–90%, 25%–75%).
      Bottom : logit_change actual vs predicted median with quantile shading (left)
               and residuals (right).

    Price band is derived as (all in [0,1] via sigmoid):
      predicted_future_mid = sigmoid(logit(anchor) + pred_q50)
      band_upper / lower   = sigmoid(logit(anchor) + pred_q_hi / pred_q_lo)

    Parameters
    ----------
    pkl_path    : path to a raw book-*.pkl file
    pipeline    : fitted Pipeline (output of train_and_evaluate)
    feat_cols   : feature column list (output of train_and_evaluate)
    target_cols : target column list (output of train_and_evaluate)
    """
    file_ts_s = _file_ts_from_path(pkl_path)
    df = pd.read_pickle(pkl_path)

    feat = extract_features(df, n_levels=n_levels)

    # Build targets on raw [0,1] tick data BEFORE logit transform or downsampling
    feat = add_target(feat, horizon_ms=horizon_ms)

    # Transform price features to logit space
    feat = apply_logit_to_prices(feat)

    # Downsample: price features → last + interval stats; other features → mean/last;
    # targets → always last (never averaged).
    target_cols_raw = [c for c in feat.columns if c.startswith('target_')]
    feat_cols_raw   = [c for c in feat.columns if not c.startswith('target_')]
    feat_ds    = downsample_features(feat[feat_cols_raw], interval_ms=downsample_ms, mean=mean)
    targets_ds = filter_to_last_per_second(feat[target_cols_raw], interval_ms=downsample_ms, mean=False)
    feat = feat_ds.join(targets_ds)

    feat = add_interval_features(feat, file_ts_s)
    feat = add_time_features(feat)

    # target_mid_anchor is not in target_cols (excluded from model outputs) but
    # we need it for price-band reconstruction.
    needed = feat_cols + target_cols + ['target_mid_anchor']
    missing = [c for c in needed if c not in feat.columns]
    if missing:
        raise ValueError(f'Columns missing from file features: {missing}')

    clean = feat[needed].dropna()
    if clean.empty:
        raise ValueError('No complete rows after dropping NaNs.')

    X      = clean[feat_cols].values
    Y_true = clean[target_cols].values   # (n_rows, n_targets)
    Y_pred = pipeline.predict(X)         # (n_rows, n_targets)

    timestamps = pd.to_datetime(clean.index, unit='ms', utc=True)
    mid_anchor = clean['target_mid_anchor'].values
    logit_anchor = _logit(mid_anchor)
    true_lc = Y_true.ravel()                     # actual logit changes, 1-D

    # Y_pred is (n_rows, n_quantiles) from QuantileGBR
    inner_model = pipeline.named_steps['model']
    quantiles   = np.asarray(inner_model.quantiles)
    median_idx  = int(np.argmin(np.abs(quantiles - 0.5)))
    pred_median = Y_pred[:, median_idx]

    # Build symmetric quantile pairs for bands, widest first
    pairs: list[tuple[int, int, str]] = []
    for i, q in enumerate(quantiles):
        if q >= 0.5:
            continue
        mirror = 1.0 - q
        j_arr = np.where(np.isclose(quantiles, mirror))[0]
        if len(j_arr):
            pairs.append((i, int(j_arr[0]), f'{q:.0%}–{mirror:.0%}'))
    pairs.sort(key=lambda t: quantiles[t[1]] - quantiles[t[0]], reverse=True)

    # ── Layout: 2 rows × 2 cols; top row spans both columns ──────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)
    ax_band = fig.add_subplot(gs[0, :])
    ax_lc   = fig.add_subplot(gs[1, 0])
    ax_res  = fig.add_subplot(gs[1, 1], sharex=ax_lc)

    fig.suptitle(f'Quantile Predictions & Price Bands — {os.path.basename(pkl_path)}',
                 fontsize=13)
    bar_w = pd.Timedelta(milliseconds=downsample_ms * 0.8)

    # ── Top panel: price bands in [0,1] ──────────────────────────────────────
    actual_future_mid = _sigmoid(logit_anchor + true_lc)
    pred_future_mid   = _sigmoid(logit_anchor + pred_median)

    ax_band.plot(timestamps, mid_anchor,        color='grey',      lw=1.0, label='Mid (anchor)',        zorder=3)
    ax_band.plot(timestamps, actual_future_mid, color='steelblue', lw=1.2, label='Actual future mid',   zorder=4)
    ax_band.plot(timestamps, pred_future_mid,   color='tomato',    lw=1.2, linestyle='--',
                 label=f'Predicted median (q{quantiles[median_idx]:.0%})', zorder=5)

    band_alphas = np.linspace(0.30, 0.10, max(len(pairs), 1))
    for (lo_idx, hi_idx, label), alpha in zip(pairs, band_alphas):
        upper = _sigmoid(logit_anchor + Y_pred[:, hi_idx])
        lower = _sigmoid(logit_anchor + Y_pred[:, lo_idx])
        ax_band.fill_between(timestamps, lower, upper,
                             color='tomato', alpha=alpha, label=f'Band {label}', zorder=2)

    ax_band.set_ylabel('Price [0, 1]')
    ax_band.set_title(f'Price bands at t+{horizon_ms // 1000}s  '
                      f'— sigmoid(logit(anchor) + quantile_prediction)', fontsize=9)
    ax_band.legend(loc='upper right', fontsize=8, ncol=3)
    ax_band.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # ── Bottom-left: logit_change actual vs median + quantile shading ─────────
    mae = mean_absolute_error(true_lc, pred_median)
    r2  = r2_score(true_lc, pred_median)
    ax_lc.plot(timestamps, true_lc,    color='steelblue', lw=1.1, label='Actual',         zorder=4)
    ax_lc.plot(timestamps, pred_median, color='tomato',   lw=1.1, linestyle='--',
               label='Predicted median', zorder=5)

    for (lo_idx, hi_idx, label), alpha in zip(pairs, band_alphas):
        ax_lc.fill_between(timestamps, Y_pred[:, lo_idx], Y_pred[:, hi_idx],
                           color='tomato', alpha=alpha, label=f'Band {label}', zorder=2)

    ax_lc.axhline(0, color='black', lw=0.7, linestyle=':')
    ax_lc.set_ylabel('logit change')
    ax_lc.set_title(f'Logit change  MAE={mae:.5f}  R²={r2:.4f}', fontsize=9)
    ax_lc.legend(loc='upper right', fontsize=8)
    ax_lc.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # ── Bottom-right: residuals (actual − median) ─────────────────────────────
    residuals = pred_median - true_lc
    ax_res.bar(timestamps, residuals, width=bar_w, color='steelblue', alpha=0.6)
    ax_res.axhline(0, color='black', lw=0.8)
    ax_res.set_ylabel('Residual (pred − actual)')
    ax_res.set_title('Logit-change residuals (median)', fontsize=9)
    ax_res.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    fig.autofmt_xdate()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    plt.show()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def build_pipeline(
    data_dir: str = 'raw_book_data',
    n_levels: int = 5,
    horizon_ms: int = 5000,
    downsample_ms: int = 1000,
    mean=False
) -> pd.DataFrame:
    """
    Run the full feature extraction pipeline.

    Per-file steps:
      1. Load raw book data
      2. Extract per-timestamp features
      3. Downsample to last snapshot per interval (default 1 s)
      4. Add interval / temporal / market-context features

    Cross-file steps:
      5. Concatenate all files, sort by time
      6. Add time-lagged & rolling features
      7. Build prediction target

    Parameters
    ----------
    downsample_ms : downsample interval in milliseconds (default 1000 = 1 s)
    """
    paths = sorted(glob.glob(os.path.join(data_dir, 'book-*.pkl')))
    if not paths:
        raise FileNotFoundError(f'No book-*.pkl files found in {data_dir}')
    print(f'Processing {len(paths)} file(s)...')

    per_file_feats = []
    for p in paths:
        df = pd.read_pickle(p)
        file_ts_s = _file_ts_from_path(p)

        feat = extract_features(df, n_levels=n_levels)

        # Build targets on raw [0,1] tick data BEFORE logit transform or downsampling.
        print(f'  {os.path.basename(p)}: building targets on {len(feat)} raw ticks...')
        feat = add_target(feat, horizon_ms=horizon_ms)

        # Transform price features to logit space AFTER targets are computed.
        feat = apply_logit_to_prices(feat)

        # Downsample: price features → last + interval stats; other features → mean/last;
        # targets → always last (never averaged).
        target_cols_raw = [c for c in feat.columns if c.startswith('target_')]
        feat_cols_raw   = [c for c in feat.columns if not c.startswith('target_')]
        feat_ds    = downsample_features(feat[feat_cols_raw], interval_ms=downsample_ms, mean=mean)
        targets_ds = filter_to_last_per_second(feat[target_cols_raw], interval_ms=downsample_ms, mean=False)
        feat = feat_ds.join(targets_ds)

        feat = add_interval_features(feat, file_ts_s)
        per_file_feats.append(feat)
        print(f'  {os.path.basename(p)}: {len(feat)} rows after {downsample_ms}ms downsample')

    feat = pd.concat(per_file_feats).sort_index()
    print(f'Total: {len(feat):,} rows across {len(paths)} file(s)')

    print('Adding time-lagged features...')
    feat = add_time_features(feat)

    return feat


if __name__ == '__main__':
    MODEL_NAME = 'model_log_ret'
    DATA_DIR    = 'raw_book_data'
    N_LEVELS    = 5
    HORIZON_MS  = 10000   # predict log return and volatility at t+10s
    DOWNSAMPLE  = 1000

    model = QuantileGBR(quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
    feat = build_pipeline(data_dir=DATA_DIR, n_levels=N_LEVELS,
                          horizon_ms=HORIZON_MS, downsample_ms=DOWNSAMPLE, 
                          mean=True)
    pipeline, feat_cols, target_cols = train_and_evaluate(feat, n_splits=5, model=model)
    print(feat_cols)

    joblib.dump(pipeline, f'{MODEL_NAME}.joblib')
    print(f'Model saved to {MODEL_NAME}.joblib')

    import json as _json
    with open(f'{MODEL_NAME}_meta.json', 'w') as _f:
        _json.dump({'feat_cols': feat_cols, 'target_cols': target_cols}, _f)
    print(f'Metadata saved to {MODEL_NAME}_meta.json')

    # Plot predictions on sample file
    sample_pkl = sorted(glob.glob(os.path.join(DATA_DIR, 'book-*.pkl')))[0]
    plot_predictions(
        sample_pkl, pipeline, feat_cols, target_cols,
        n_levels=N_LEVELS, horizon_ms=HORIZON_MS, downsample_ms=DOWNSAMPLE,
        mean=False,
        save_path='predictions.png',
    )
