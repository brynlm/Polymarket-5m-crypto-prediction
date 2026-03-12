"""
feature_pipeline.py

End-to-end pipeline:
  1. Load raw orderbook snapshots from raw_book_data/
  2. Extract per-timestamp features (best prices, depth, imbalance, microprice)
  3. Add time-lagged / rolling features
  4. Build target: mid price 5 seconds ahead
  5. Train a GradientBoostingRegressor with TimeSeriesSplit CV
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
# 4. Time-lagged & rolling features
# ──────────────────────────────────────────────────────────────

# Columns to lag
_LAG_COLS    = ['mid', 'spread', 'rel_spread', 'imbalance', 'imbalance_all',
                'microprice', 'micro_minus_mid', 'bid_vol', 'ask_vol',
                'btc_price', 'btc_price_from_open', 'vwap', 'ofi']
_LAGS        = list(range(10)) #[1, 2, 5, 10, 20]
_DIFF_COLS   = ['mid', 'spread', 'imbalance']
_ROLL_COLS   = ['mid', 'imbalance', 'spread']
_ROLL_WINS   = [5, 10, 20]


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
# 5. Build target: mid price N ms ahead
# ──────────────────────────────────────────────────────────────

def add_target(feat: pd.DataFrame, horizon_ms: int = 5000, step_ms: int = 1000) -> pd.DataFrame:
    """
    For each row at timestamp t, find the mid price at each step within the
    horizon window and record them as separate target columns.

    With the defaults this produces targets at t+1s, t+2s, t+3s, t+4s, t+5s.

    Columns added:
      target_<k>s      : mid price at t + k*step_ms  (k = 1 … n_steps)
      target_diff_<k>s : mid price change from current mid
    """
    ts_arr  = feat.index.values
    mid_arr = feat['mid'].values
    n_steps = horizon_ms // step_ms

    for k in range(1, n_steps + 1):
        offset = k * step_ms
        future_mid = np.full(len(ts_arr), np.nan)
        for i, t in enumerate(ts_arr):
            j = np.searchsorted(ts_arr, t + offset)
            if j < len(ts_arr):
                future_mid[i] = mid_arr[j]
        feat[f'target_{k}s']      = future_mid
        feat[f'target_diff_{k}s'] = future_mid - mid_arr

    return feat


# ──────────────────────────────────────────────────────────────
# 6. Training & evaluation
# ──────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _supports_multioutput(estimator) -> bool:
    """Return True if estimator natively handles multi-output targets."""
    if isinstance(estimator, MultiOutputRegressor):
        return True
    # sklearn >= 1.6 uses __sklearn_tags__; older versions use _get_tags()
    try:
        return estimator.__sklearn_tags__().target_tags.multi_output
    except AttributeError:
        pass
    try:
        return estimator._get_tags().get('multioutput', False)
    except AttributeError:
        return False


def _get_feature_importances(model, feat_cols: list[str]) -> pd.Series | None:
    """
    Extract feature importances from a fitted model regardless of whether it
    is a MultiOutputRegressor wrapper or a native multi-output estimator.
    Returns None if the model type doesn't expose importances.
    """
    # Unwrapped MultiOutputRegressor: average across per-output estimators
    if isinstance(model, MultiOutputRegressor) and hasattr(model, 'estimators_'):
        arrays = [e.feature_importances_ for e in model.estimators_
                  if hasattr(e, 'feature_importances_')]
        if arrays:
            return pd.Series(np.mean(arrays, axis=0), index=feat_cols)
        return None
    # Native multi-output model (e.g. RandomForest)
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=feat_cols)
    return None


def baseline_metrics(y_true, y_current_mid):
    """Naive baseline: predict current mid = future mid (no-change)."""
    mae  = mean_absolute_error(y_true, y_current_mid)
    rmse = _rmse(y_true, y_current_mid)
    r2   = r2_score(y_true, y_current_mid)
    print(f'  Baseline (no-change): MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}')
    return mae, rmse, r2


def train_and_evaluate(
    feat: pd.DataFrame,
    target_cols: list[str] | None = None,
    model=None,
    n_splits: int = 5,
) -> tuple[Pipeline, list[str], list[str]]:
    """
    Train a MultiOutputRegressor(GradientBoostingRegressor) using TimeSeriesSplit CV.

    Parameters
    ----------
    feat        : feature DataFrame (output of full pipeline)
    target_cols : list of target column names to predict; defaults to all
                  target_<k>s columns found in feat
    model       : sklearn estimator for a single output; defaults to
                  GradientBoostingRegressor wrapped in MultiOutputRegressor
    n_splits    : number of TimeSeriesSplit folds

    Returns
    -------
    pipeline    : fitted Pipeline(StandardScaler + MultiOutputRegressor) on all data
    feat_cols   : list of feature column names used
    target_cols : list of target column names predicted
    """
    all_target_cols = [c for c in feat.columns if c.startswith('target_')]
    if target_cols is None:
        target_cols = sorted(
            [c for c in all_target_cols if not c.startswith('target_diff_')],
            key=lambda c: int(c.split('_')[1].rstrip('s'))
        )

    feat_cols = [c for c in feat.columns if c not in set(all_target_cols)]

    clean = feat[feat_cols + target_cols].dropna()
    X = clean[feat_cols].values
    Y = clean[target_cols].values  # shape (n_samples, n_steps)

    mid_idx = feat_cols.index('mid')
    y_mid   = X[:, mid_idx]

    print(f'\nDataset: {len(clean):,} rows × {len(feat_cols)} features → {len(target_cols)} targets')
    print(f'Targets: {target_cols}')
    print(f'\n--- Baseline (per target) ---')
    for k in range(len(target_cols)):
        baseline_metrics(Y[:, k], y_mid)

    if model is None:
        model = GradientBoostingRegressor(
            loss='absolute_error',
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=1,
            min_samples_leaf=1,
            random_state=42,
        )

    if not _supports_multioutput(model):
        model = MultiOutputRegressor(model, n_jobs=-1)
    # TODO: Add Column Transformer to exclude time_of_day from midnight feature from MinMaxScaling
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
    tscv     = TimeSeriesSplit(n_splits=n_splits)

    print(f'\n--- TimeSeriesSplit CV ({n_splits} folds) ---')
    fold_maes, fold_rmses, fold_r2s = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        pipeline.fit(X[tr_idx], Y[tr_idx])
        preds = pipeline.predict(X[val_idx])  # (n_val, n_steps)

        mae  = mean_absolute_error(Y[val_idx], preds)
        rmse = np.sqrt(mean_squared_error(Y[val_idx], preds))
        r2   = r2_score(Y[val_idx], preds)

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        print(f'  Fold {fold+1}: MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}  '
              f'(n_train={len(tr_idx):,}  n_val={len(val_idx):,})')

    print(f'\n  Mean   MAE={np.mean(fold_maes):.6f}  '
          f'RMSE={np.mean(fold_rmses):.6f}  R²={np.mean(fold_r2s):.4f}')

    # Refit on full dataset
    pipeline.fit(X, Y)

    # # Feature importances (if the model exposes them)
    # imp = _get_feature_importances(pipeline.named_steps['model'], feat_cols)
    # if imp is not None:
    #     print(f'\n--- Top 20 feature importances (mean across outputs) ---')
    #     print(imp.sort_values(ascending=False).head(20).to_string())

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
    horizon_ms: int = 5000,
    downsample_ms: int = 1000,
    save_path: str | None = None,
) -> None:
    """
    Run the feature pipeline on a single .pkl file, generate multi-output
    predictions, and plot actual vs predicted for each horizon step.

    One row of subplots per target (actual + predicted on top, residuals below).

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
    feat = filter_to_last_per_second(feat, interval_ms=downsample_ms)
    feat = add_interval_features(feat, file_ts_s)
    feat = add_time_features(feat)
    feat = add_target(feat, horizon_ms=horizon_ms)

    needed = feat_cols + target_cols
    missing = [c for c in needed if c not in feat.columns]
    if missing:
        raise ValueError(f'Columns missing from file features: {missing}')

    clean = feat[needed].dropna()
    if clean.empty:
        raise ValueError('No complete rows after dropping NaNs.')

    X      = clean[feat_cols].values
    Y_true = clean[target_cols].values          # (n_rows, n_steps)
    Y_pred = pipeline.predict(X)                # (n_rows, n_steps)

    timestamps = pd.to_datetime(clean.index, unit='ms', utc=True)
    n_steps    = len(target_cols)

    fig, axes = plt.subplots(n_steps, 2, figsize=(14, 3.5 * n_steps), sharex=True)
    if n_steps == 1:
        axes = axes[np.newaxis, :]  # keep 2-D indexing for single target

    fig.suptitle(f'Actual vs Predicted — {os.path.basename(pkl_path)}', fontsize=13)

    for i, tc in enumerate(target_cols):
        y_true = Y_true[:, i]
        y_pred = Y_pred[:, i]
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        # Left: actual vs predicted
        ax = axes[i, 0]
        ax.plot(timestamps, y_true, label='Actual',    linewidth=1.1)
        ax.plot(timestamps, y_pred, label='Predicted', linewidth=1.1, linestyle='--', alpha=0.85)
        ax.set_ylabel(tc)
        ax.set_title(f'{tc}  MAE={mae:.5f}  RMSE={rmse:.5f}  R²={r2:.4f}', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)

        # Right: residuals
        ax2 = axes[i, 1]
        residuals = y_pred - y_true
        ax2.bar(timestamps, residuals,
                width=pd.Timedelta(milliseconds=downsample_ms * 0.8),
                color='steelblue', alpha=0.6)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel('Residual (pred − actual)')
        ax2.set_title(f'{tc} residuals', fontsize=9)

    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    fig.autofmt_xdate()
    plt.tight_layout()
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
        feat = filter_to_last_per_second(feat, interval_ms=downsample_ms, mean=mean)
        feat = add_interval_features(feat, file_ts_s)
        per_file_feats.append(feat)
        print(f'  {os.path.basename(p)}: {len(feat)} rows after {downsample_ms}ms filter')

    feat = pd.concat(per_file_feats).sort_index()
    print(f'Total: {len(feat):,} rows across {len(paths)} file(s)')

    print('Adding time-lagged features...')
    feat = add_time_features(feat)

    print(f'Adding target (horizon={horizon_ms}ms)...')
    feat = add_target(feat, horizon_ms=horizon_ms)

    return feat


if __name__ == '__main__':
    MODEL_NAME = 'model_mean_downsample'
    DATA_DIR    = 'raw_book_data'
    N_LEVELS    = 5
    HORIZON_MS  = 5000
    DOWNSAMPLE  = 1000

    # model = sklearn.svm.SVR()
    model = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=42)
    # model = None
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

    sample_pkl = sorted(glob.glob(os.path.join(DATA_DIR, 'book-*.pkl')))[0]
    plot_predictions(
        sample_pkl, pipeline, feat_cols, target_cols,
        n_levels=N_LEVELS, horizon_ms=HORIZON_MS, downsample_ms=DOWNSAMPLE,
        save_path='predictions.png',
    )
