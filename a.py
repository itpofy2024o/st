import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
from pathlib import Path
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.display.max_rows = None
pd.set_option('future.no_silent_downcasting', True)

# ------------------ DATA LOADING + CLEANING ------------------
ndf = pd.read_csv("../Downloads/st_exportd.csv").iloc[:,1:]
for i in range(len(ndf)):
    ndf.iloc[i,-3] = ndf.iloc[i,-3].split("GMT")[0].strip()
ndf['timestamp'] = pd.to_datetime(ndf['timestamp'])

train_df = ndf.iloc[:121455, :]
actual_x = ndf.iloc[121455:, :]

assets_to_train = [i for i in actual_x["asset"].unique().tolist() if i in train_df["asset"].unique().tolist()]
c = [[len(train_df[train_df["asset"]==i]), i] for i in assets_to_train]
c = sorted(c, key=lambda x: x[0], reverse=True)
print(c)

# ------------------ OUTPUT DIRECTORIES ------------------
OUT_DIR = Path("./eval_outputs")
MODELS_DIR = OUT_DIR / "asset_models"
PRED_CSV = OUT_DIR / "predictions_nndf_with_metrics.csv"
PER_ASSET_CSV = OUT_DIR / "per_asset_metrics.csv"
PER_ASSET_GAP_CSV = OUT_DIR / "per_asset_gap_metrics.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 200

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ GAP HANDLING ------------------
def parse_gap_to_minutes(g):
    if pd.isna(g):
        return np.nan
    s = str(g).strip()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    try:
        return int(s)
    except:
        return np.nan

GAP_TOTAL_MIN = {
    '1m': 300, '3m': 300, '5m': 300, '15m': 300, '30m': 300, '1h': 300,
    '2h': 7200, '4h': 7200, '6h': 7200, '8h': 7200, '12h': 7200
}

# ------------------ PREP DATA ------------------
df_raw = train_df.copy()
nndf = actual_x.copy()
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
nndf['timestamp'] = pd.to_datetime(nndf['timestamp'])

df_raw['gap_min'] = df_raw['gap'].apply(parse_gap_to_minutes)
nndf['gap_min'] = nndf['gap'].apply(parse_gap_to_minutes)

df_sorted = df_raw.sort_values(['asset','gap','timestamp']).copy()
df_sorted['true_next_vwap'] = df_sorted.groupby(['asset','gap'])['vwap'].shift(-1)
train_rows = df_sorted.dropna(subset=['true_next_vwap']).copy()
print("Train rows with valid true_next_vwap:", len(train_rows))

meta_cols = {'asset','timestamp','gap','gaplimit','gap_min','true_next_vwap'}
numeric_cols = train_rows.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric_cols if c not in meta_cols]
if 'gap_min' not in features:
    features.append('gap_min')
if 'gaplimit' not in features and 'gaplimit' in df_sorted.columns:
    features.append('gaplimit')
features = list(dict.fromkeys(features))
print("Using features:", features)

ensure_dirs()
assets = sorted(list(set(nndf['asset'].unique()) & set(df_raw['asset'].unique())))
print("Assets to train/evaluate:", assets)

# ------------------ TWO-MODEL-PER-ASSET TRAINING ------------------
asset_models = {}
train_stats = []

for asset in assets:
    rows = train_rows[train_rows['asset'] == asset]
    if len(rows) < 2:
        print(f"Skipping {asset} (insufficient training rows: {len(rows)})")
        continue

    # Split into low-gap (<=1h) and high-gap (>1h) buckets
    low_rows = rows[rows['gap_min'] <= 60]
    high_rows = rows[rows['gap_min'] > 60]

    low_model = None
    high_model = None

    if len(low_rows) >= 2:
        X_low = low_rows[features].fillna(0).values
        y_low = low_rows['true_next_vwap'].values
        X_tr, X_val, y_tr, y_val = train_test_split(X_low, y_low, test_size=0.15, random_state=RANDOM_STATE)
        low_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        low_model.fit(X_tr, y_tr)
    if len(high_rows) >= 2:
        X_high = high_rows[features].fillna(0).values
        y_high = high_rows['true_next_vwap'].values
        X_tr, X_val, y_tr, y_val = train_test_split(X_high, y_high, test_size=0.15, random_state=RANDOM_STATE)
        high_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        high_model.fit(X_tr, y_tr)

    asset_models[asset] = {'low_obj': low_model, 'high_obj': high_model, 'features': features}

# ------------------ PREDICTION ------------------
pred_records = []
gaps_universe = sorted(df_raw['gap'].dropna().unique().tolist(), key=str)

for asset, minfo in asset_models.items():
    adf = df_raw[df_raw['asset'] == asset]
    low_model = minfo['low_obj']
    high_model = minfo['high_obj']
    feats = minfo['features']

    for gap in gaps_universe:
        sub = adf[adf['gap'] == gap]
        if sub.empty:
            continue
        last = sub.sort_values('timestamp').iloc[-1]
        X_row = last[feats].fillna(0).values.reshape(1,-1)
        if last['gap_min'] <= 60:
            model = low_model
        else:
            model = high_model
        if model is None:
            continue
        pred = float(model.predict(X_row)[0])
        pred_records.append({
            'asset': asset,
            'gap': gap,
            'gap_min': parse_gap_to_minutes(gap),
            'gaplimit': last.get('gaplimit', np.nan),
            'last_df_timestamp': last['timestamp'],
            'last_df_vwap': float(last['vwap']),
            'pred_next_vwap': pred
        })

pred_df = pd.DataFrame(pred_records)
print("Predictions produced (asset x gap):", len(pred_df))

# ------------------ EVALUATION ------------------
eval_rows = []

for _, prow in pred_df.iterrows():
    asset = prow['asset']
    gap = prow['gap']
    last_ts = prow['last_df_timestamp']
    gap_min = prow['gap_min']
    gaplimit = prow['gaplimit']
    pred_val = prow['pred_next_vwap']
    last_vwap = prow['last_df_vwap']

    cand = nndf[(nndf['asset'] == asset) &
                (nndf['gap'] == gap) &
                (nndf['timestamp'] > last_ts)]
    total_min = GAP_TOTAL_MIN[gap]
    window_end = last_ts + timedelta(minutes=total_min)
    cand_sel = cand[cand['timestamp'] <= window_end]

    if cand_sel.empty:
        eval_rows.append({
            'asset': asset,
            'gap': gap,
            'count_truths': 0,
            'pred_next_vwap': pred_val,
            'rmse': np.nan,
            'mae': np.nan,
            'direction_acc': np.nan
        })
        continue

    truths = cand_sel['vwap'].values.astype(float)
    errors = truths - pred_val
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))

    pred_dir = np.sign(pred_val - last_vwap)
    true_dirs = np.sign(truths - last_vwap)
    dir_acc = np.mean(true_dirs == pred_dir)

    eval_rows.append({
        'asset': asset,
        'gap': gap,
        'count_truths': len(truths),
        'pred_next_vwap': pred_val,
        'rmse': float(rmse),
        'mae': float(mae),
        'direction_acc': float(dir_acc)
    })

eval_df = pd.DataFrame(eval_rows)

# ------------------- AGGREGATE PER-ASSET -------------------
agg_records = []
for asset, g in eval_df.groupby('asset'):
    valid = g.dropna(subset=['rmse', 'mae', 'direction_acc'])
    if valid.empty:
        agg_records.append({
            'asset': asset,
            'RMSE': np.nan,
            'MAE': np.nan,
            'Direction_Acc': np.nan,
            'Count': int(g['count_truths'].sum())
        })
        continue
    counts = valid['count_truths'].values
    rmse_w = np.average(valid['rmse'], weights=counts)
    mae_w = np.average(valid['mae'], weights=counts)
    diracc_w = np.average(valid['direction_acc'], weights=counts)
    agg_records.append({
        'asset': asset,
        'RMSE': float(rmse_w),
        'MAE': float(mae_w),
        'Direction_Acc': float(diracc_w),
        'Count': int(g['count_truths'].sum())
    })

per_asset_metrics_df = pd.DataFrame(agg_records).sort_values('RMSE', ascending=False)

# ------------------- SAVE OUTPUTS -------------------
ensure_dirs()
pred_full = pred_df.merge(eval_df, on=['asset','gap'], how='left')
pred_full.to_csv(PRED_CSV, index=False)
per_asset_metrics_df.to_csv(PER_ASSET_CSV, index=False)
eval_df.to_csv(PER_ASSET_GAP_CSV, index=False)
f=['BTC','ETH','SOL','ZEC','BNB', 'XRP']
# ------------------- FINAL REPORT -------------------
print("Saved outputs to:", OUT_DIR)
print("Models created (count assets with >=1 bucket trained):", 
      sum(1 for v in asset_models.values() if (v['low_obj'] is not None) or (v['high_obj'] is not None)))
print("\nTop assets by RMSE (per_asset_metrics):")
# print(per_asset_metrics_df.dropna(subset=['RMSE', 'MAE', 'Direction_Acc']))
print(per_asset_metrics_df[
    # per_asset_metrics_df['asset'].isin(f)
    # & 
    per_asset_metrics_df[['RMSE', 'MAE', 'Direction_Acc']].notna().all(axis=1)
].reset_index(drop=True))
print("\nPer-(asset,gap) eval (top RMSE rows):")
#print(eval_df.dropna(subset=['rmse', 'mae', 'direction_acc'].sort_values('rmse', ascending=False)))
print(eval_df[eval_df['asset'].isin(f)& eval_df[['rmse', 'mae', 'direction_acc']].notna().all(axis=1)
].reset_index(drop=True))
print("Pipeline finished.")

