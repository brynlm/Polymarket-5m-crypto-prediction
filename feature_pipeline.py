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
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
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

    feat = pd.concat(
        [best_bid, best_ask, bid_vol, ask_vol, bid_vol_all, ask_vol_all,
         bid_n_lvl, ask_n_lvl, bid_piv, ask_piv, btc_price],
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
# 3. Per-second downsampling + interval/temporal features
# ──────────────────────────────────────────────────────────────

def filter_to_last_per_second(feat: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last snapshot within each 1-second interval.
    Index remains in milliseconds (start-of-second, i.e. floor to 1000 ms)."""
    second_bins = feat.index // 1000
    result = feat.groupby(second_bins).last()
    result.index = result.index * 1000  # restore ms scale
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

    # # Prediction-market target: earliest midprice in this file
    # target_price = feat['mid'].iloc[0]
    # feat['dist_from_target'] = feat['mid'] - target_price
    # feat['above_target']     = (feat['mid'] > target_price).astype(float)

    return feat


# ──────────────────────────────────────────────────────────────
# 4. Time-lagged & rolling features
# ──────────────────────────────────────────────────────────────

# Columns to lag
_LAG_COLS    = ['mid', 'spread', 'rel_spread', 'imbalance', 'imbalance_all',
                'microprice', 'micro_minus_mid', 'bid_vol', 'ask_vol',
                'btc_price', 'btc_price_from_open']
_LAGS        = [1, 2, 5, 10, 20]
_DIFF_COLS   = ['mid', 'spread', 'imbalance']
_ROLL_COLS   = ['mid', 'imbalance', 'spread']
_ROLL_WINS   = [5, 10, 20]


def add_time_features(feat: pd.DataFrame) -> pd.DataFrame:
    for col in _LAG_COLS:
        if col not in feat.columns:
            continue
        for lag in _LAGS:
            feat[f'{col}_lag{lag}'] = feat[col].shift(lag)

    for col in _DIFF_COLS:
        feat[f'{col}_diff1'] = feat[col].diff(1)
        feat[f'{col}_diff5'] = feat[col].diff(5)

    for col in _ROLL_COLS:
        for w in _ROLL_WINS:
            feat[f'{col}_rmean{w}'] = feat[col].rolling(w).mean()
            feat[f'{col}_rstd{w}']  = feat[col].rolling(w).std()

    return feat


# ──────────────────────────────────────────────────────────────
# 5. Build target: mid price N ms ahead
# ──────────────────────────────────────────────────────────────

def add_target(feat: pd.DataFrame, horizon_ms: int = 5000) -> pd.DataFrame:
    """
    For each row at timestamp t, find the snapshot at the earliest timestamp
    >= t + horizon_ms and record its mid price as the target.
    Also records the price change (future_mid - current_mid) for reference.
    """
    ts_arr  = feat.index.values
    mid_arr = feat['mid'].values
    future_mid = np.full(len(ts_arr), np.nan)

    for i, t in enumerate(ts_arr):
        j = np.searchsorted(ts_arr, t + horizon_ms)
        if j < len(ts_arr):
            future_mid[i] = mid_arr[j]

    feat['target']      = future_mid
    feat['target_diff'] = future_mid - feat['mid'].values  # mid price change
    return feat


# ──────────────────────────────────────────────────────────────
# 6. Training & evaluation
# ──────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def baseline_metrics(y_true, y_current_mid):
    """Naive baseline: predict current mid = future mid (no-change)."""
    mae  = mean_absolute_error(y_true, y_current_mid)
    rmse = _rmse(y_true, y_current_mid)
    r2   = r2_score(y_true, y_current_mid)
    print(f'  Baseline (no-change): MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}')
    return mae, rmse, r2


def train_and_evaluate(
    feat: pd.DataFrame,
    target_col: str = 'target',
    model=None,
    n_splits: int = 5,
) -> tuple[Pipeline, list[str]]:
    """
    Train a GradientBoostingRegressor using TimeSeriesSplit CV.

    Parameters
    ----------
    feat       : feature DataFrame (output of full pipeline)
    target_col : 'target' (raw future mid) or 'target_diff' (price change)
    model      : sklearn estimator; defaults to GradientBoostingRegressor
    n_splits   : number of TimeSeriesSplit folds

    Returns
    -------
    pipeline   : fitted Pipeline(StandardScaler + model) on all data
    feat_cols  : list of feature column names used
    """
    drop_cols = {'target', 'target_diff'}
    feat_cols = [c for c in feat.columns if c not in drop_cols]

    clean = feat[feat_cols + [target_col]].dropna()
    X = clean[feat_cols].values
    y = clean[target_col].values

    # Current mid for baseline comparison
    mid_idx  = feat_cols.index('mid')
    y_mid    = X[:, mid_idx]

    print(f'\nDataset: {len(clean):,} rows × {len(feat_cols)} features')
    print(f'Target: {target_col}  |  mean={y.mean():.5f}  std={y.std():.5f}')
    print(f'\n--- Baseline ---')
    baseline_metrics(y, y_mid)

    if model is None:
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )

    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    tscv     = TimeSeriesSplit(n_splits=n_splits)

    print(f'\n--- TimeSeriesSplit CV ({n_splits} folds) ---')
    fold_maes, fold_rmses, fold_r2s = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        pipeline.fit(X[tr_idx], y[tr_idx])
        preds = pipeline.predict(X[val_idx])

        mae  = mean_absolute_error(y[val_idx], preds)
        rmse = _rmse(y[val_idx], preds)
        r2   = r2_score(y[val_idx], preds)

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        print(f'  Fold {fold+1}: MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}  '
              f'(n_train={len(tr_idx):,}  n_val={len(val_idx):,})')

    print(f'\n  Mean   MAE={np.mean(fold_maes):.6f}  '
          f'RMSE={np.mean(fold_rmses):.6f}  R²={np.mean(fold_r2s):.4f}')

    # Refit on full dataset
    pipeline.fit(X, y)

    # Feature importances
    mdl = pipeline.named_steps['model']
    if hasattr(mdl, 'feature_importances_'):
        imp = pd.Series(mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print(f'\n--- Top 20 feature importances ---')
        print(imp.head(20).to_string())

    return pipeline, feat_cols


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def build_pipeline(
    data_dir: str = 'raw_book_data',
    n_levels: int = 5,
    horizon_ms: int = 5000,
) -> pd.DataFrame:
    """
    Run the full feature extraction pipeline.

    Per-file steps:
      1. Load raw book data
      2. Extract per-timestamp features
      3. Downsample to last snapshot per 1-second interval
      4. Add interval / temporal / market-context features

    Cross-file steps:
      5. Concatenate all files, sort by time
      6. Add time-lagged & rolling features
      7. Build prediction target
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
        feat = filter_to_last_per_second(feat)
        feat = add_interval_features(feat, file_ts_s)
        per_file_feats.append(feat)
        print(f'  {os.path.basename(p)}: {len(feat)} rows after 1-s filter')

    feat = pd.concat(per_file_feats).sort_index()
    print(f'Total: {len(feat):,} rows across {len(paths)} file(s)')

    print('Adding time-lagged features...')
    feat = add_time_features(feat)

    print(f'Adding target (horizon={horizon_ms}ms)...')
    feat = add_target(feat, horizon_ms=horizon_ms)

    return feat


if __name__ == '__main__':
    feat = build_pipeline(data_dir='raw_book_data', n_levels=5, horizon_ms=5000)
    pipeline, feat_cols = train_and_evaluate(feat, target_col='target', n_splits=5)
