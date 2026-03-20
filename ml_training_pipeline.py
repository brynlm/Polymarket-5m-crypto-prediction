from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
import json
import glob
import os
from xgboost import XGBRegressor

# Downsample columns to 1s
MAX_MIN_COLS = ['best_bid', 'best_ask', 'mid', 'ask_vol_all', 'bid_vol_all']
AVE_COLS = ['spread', 'rel_spread']

# Order book depth
N_LEVELS = 5

# Add time-lagged + rolling average features and diff features
LAGGED_COLS = ['best_bid', 'best_ask', 'mid',
               'best_bid_max', 'best_ask_max', 'mid_max',
               'best_bid_min', 'best_ask_min', 'mid_min', 'ofi']
LAGS = [1,2,3,4,5]
ROLL_AVE_COLS = ['mid', 'spread', 'vwap', 'best_bid', 'best_ask',
                'rel_spread', 'btc_price', 'btc_price_from_open']
ROLL_WINDOWS = [3,4,5,10]

# Prediction window
PRED_WINDOW = 5


def load_raw_books(data_dir='raw_book_data') -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, 'book-*.pkl')))
    if not paths:
        raise FileNotFoundError(f'No book-*.pkl files found in {data_dir}')
    print(f'Loading {len(paths)} file(s)...')
    data = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)
    return data

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

def _logit(x):
    x_clipped = np.clip(x, 1e-6, 1-1e-6)
    logits = np.log(1 / (1-x_clipped))
    return logits

def downsample_features_1s(feat, max_min_cols, ave_cols):
    # Downsample 1s intervals
    filter_1s = feat.index // 1000
    downsamp_feat = feat.groupby(filter_1s).last()  # Get last point in each interval for (all features)
    downsamp_maxmin = feat.groupby(filter_1s)[max_min_cols].agg(['max', 'min']) # Get max and min in each interval
    downsamp_maxmin.columns = ['_'.join(col) for col in downsamp_maxmin.columns.values]
    downsamp_ave = (feat.groupby(filter_1s)[ave_cols] # Get average over each interval
                    .mean()
                    .rename(columns=lambda x: x+'_ave'))
    downsamp_feat = pd.concat([downsamp_feat, downsamp_maxmin, downsamp_ave], axis=1)
    return downsamp_feat


def transform_features(downsamp_feat):
    # Add BTC Price From Open features:
    downsamp_feat['btc_price_from_open'] = downsamp_feat['btc_price'].groupby(
        (downsamp_feat.index // 300) * 300, group_keys=False).apply(
            lambda x: x - x.iloc[0], include_groups=False)

    # Add Order Flow Imbalance feature:
    downsamp_feat['ofi'] = downsamp_feat['bid_vol'].diff(1) - downsamp_feat['ask_vol'].diff(1)

    # Add target columns (return)
    downsamp_feat['return'] = downsamp_feat['mid'].shift(-PRED_WINDOW) - downsamp_feat['mid']

    # Add lagged features
    for lag in LAGS:
        lagged_feats = downsamp_feat[LAGGED_COLS].shift(lag).rename(columns=lambda x: x+f'_lag{lag}')
        downsamp_feat = pd.concat([downsamp_feat, lagged_feats], axis=1)

    # Add rolling ave features
    for w in ROLL_WINDOWS:
        ave_feats = downsamp_feat[ROLL_AVE_COLS].rolling(w).mean().rename(columns=lambda x: x+f'_ave{w}')
        downsamp_feat = pd.concat([downsamp_feat, ave_feats], axis=1)

    # Add temporal features
    downsamp_feat['time_in_interval'] = downsamp_feat.index % 300
    downsamp_feat['time_since_midnight'] = downsamp_feat.index % 86400

    return downsamp_feat


if __name__ == "__main__":
    MODEL_NAME = 'xgb_qreg_5s'
    TARGET_COLS = ['return']
    QUANTILES = [0.1, 0.5, 0.9]
    LOAD_RAW_DATA = True

    models = {q: Pipeline([('scaler', MinMaxScaler()), 
                           ('model', XGBRegressor(
                               objective='reg:quantileerror', 
                               quantile_alpha=q, 
                               n_estimators=200, 
                               random_state=42))]) 
                for q in QUANTILES}
    
    feat = pd.read_pickle('raw_extracted_features.pkl') # Load data
    if LOAD_RAW_DATA:
        raw_data = load_raw_books() # Load raw data (if any)
        raw_feat = extract_features(raw_data) # Get orderbook features
        feat = pd.concat([feat, raw_feat]) # Combine with old data
        feat.to_pickle('raw_extracted_features.pkl') # Save to pkl for later

    feat['mid'] = _logit(feat['mid']) # Transform mid price to logit space
    downsamp_feat = downsample_features_1s(feat, MAX_MIN_COLS, AVE_COLS) # Downsample to 1s intervals
    downsamp_feat = transform_features(downsamp_feat) # Apply feature transformations

    clean = downsamp_feat.dropna()
    X = clean.drop(columns=TARGET_COLS).values
    y = clean[TARGET_COLS].values
    tscv = TimeSeriesSplit(n_splits=5)
    print(f'\n--- TimeSeriesSplit CV (5 folds) ---')
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        pb_losses = {}
        for q in QUANTILES:
            models[q].fit(X[tr_idx], y[tr_idx])
            y_pred = models[q].predict(X[val_idx])
            y_val = y[val_idx]

            err = y_val - y_pred
            loss = float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))
            pb_losses[q] = loss
        print(f'  Fold {fold+1}: PBL_10={pb_losses[0.1]:.6f}  PBL_50={pb_losses[0.5]:.6f}  PBL_90={pb_losses[0.9]:.6f}'
            f'(n_train={len(tr_idx):,}  n_val={len(val_idx):,})')
        
    # Re-fit on whole dataset
    pred_returns = {}
    pred_quantiles = {}
    for q in QUANTILES:
        models[q].fit(X, y)
        pred_returns[q] = models[q].predict(X)
        pred_quantiles[q] = (clean['mid'] + pred_returns[q]).shift(PRED_WINDOW)
        # Transform back from logits to price-space:
        pred_quantiles[q] = pred_quantiles[q].apply(lambda x: 1 / (1 + np.exp(-x)))

    # Computing other diagnostics
    actual = clean['mid'].apply(lambda x: 1 / (1 + np.exp(-x)))
    within_80_interval = (actual.values[PRED_WINDOW:] >= pred_quantiles[0.1].values[PRED_WINDOW:]) & \
        ((actual.values[PRED_WINDOW:] <= pred_quantiles[0.9].values[PRED_WINDOW:]))
    coverage = np.mean(within_80_interval)
    coverage_90 = np.mean(actual.values[PRED_WINDOW:] <= pred_quantiles[0.9].values[PRED_WINDOW:])
    coverage_10 = np.mean(actual.values[PRED_WINDOW:] <= pred_quantiles[0.1].values[PRED_WINDOW:])
    directional_acc = np.mean(np.sign(clean['return']).values == np.sign(pred_returns[0.5]))
    print(f"80% interval coverage: {coverage}")
    print(f"90% quantile coverage: {coverage_90}")
    print(f"10% quantile coverage: {coverage_10}")
    print(f"Directional accuracy of median: {directional_acc}")
    
    # Save model:
    joblib.dump(models, MODEL_NAME + '.joblib')

    # Save metadata (feature config + ordered column list for inference)
    feat_cols = clean.drop(columns=TARGET_COLS).columns.tolist()
    meta = {
        'feat_cols':     feat_cols,
        'quantiles':     QUANTILES,
        'target':        TARGET_COLS[0],
        'n_features':    len(feat_cols),
        'pred_window':   PRED_WINDOW,
        'n_levels':      N_LEVELS,
        'max_min_cols':  MAX_MIN_COLS,
        'ave_cols':      AVE_COLS,
        'lagged_cols':   LAGGED_COLS,
        'lags':          LAGS,
        'roll_ave_cols': ROLL_AVE_COLS,
        'roll_windows':  ROLL_WINDOWS,
    }
    with open(f'{MODEL_NAME}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model ({len(feat_cols)} features) → {MODEL_NAME}.joblib + {MODEL_NAME}_meta.json")