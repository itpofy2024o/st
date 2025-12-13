import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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
    # Safely handle potential non-string values before split
    val = ndf.iloc[i,-3]
    if isinstance(val, str):
        ndf.iloc[i,-3] = val.split("GMT")[0].strip()
    else:
        ndf.iloc[i,-3] = np.nan
ndf['timestamp'] = pd.to_datetime(ndf['timestamp'], errors='coerce')

# Global sorting here is safest for the initial split
ndf = ndf.sort_values('timestamp').reset_index(drop=True) 

# Note: Using your original data split points
train_df = ndf.iloc[:122684, :].copy()
actual_x = ndf.iloc[122684:, :].copy()

# --- FIXED: Use simple filtering based on asset names only ---
assets_in_train = set(train_df["asset"].unique().tolist())
assets_in_actual_x = set(actual_x["asset"].unique().tolist())
assets_to_train = sorted(list(assets_in_train.intersection(assets_in_actual_x)))

# Recalculate 'c' correctly for the printout
c = [[len(train_df[train_df["asset"]==i]), i] for i in assets_to_train]
c = sorted(c, key=lambda x: x[0], reverse=True)
print("Assets by data count (Filtered for Eval):", c)

# ------------------ OUTPUT DIRECTORIES ------------------
OUT_DIR = Path("./eval_outputs_gbm_fixed")
MODELS_DIR = OUT_DIR / "asset_models"
PRED_CSV = OUT_DIR / "predictions_gbm_with_metrics.csv"
PER_ASSET_CSV = OUT_DIR / "per_asset_metrics_gbm_with_mape.csv" # Renamed output file
PER_ASSET_GAP_CSV = OUT_DIR / "per_asset_gap_metrics_gbm.csv"

# Hyperparameters for Gradient Boosting
RANDOM_STATE = 42
N_ESTIMATORS = 200
LEARNING_RATE = 0.1
MAX_DEPTH = 5

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
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
    '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720
}

# ------------------ PREP DATA ------------------
df_raw = train_df.copy() # CRITICAL FIX: Define df_raw for training data
nndf = actual_x.copy()

df_raw['gap_min'] = df_raw['gap'].apply(parse_gap_to_minutes)
nndf['gap_min'] = nndf['gap'].apply(parse_gap_to_minutes)

# Sort by asset and GAP for correct shifting
df_sorted = df_raw.sort_values(['asset','gap','timestamp']).copy()
df_sorted['next_vwap'] = df_sorted.groupby(['asset','gap'])['vwap'].shift(-1)

# --- FIXED: Predict Returns (%) to enable extrapolation ---
epsilon = 1e-6 # For safe division
df_sorted['target_return'] = (df_sorted['next_vwap'] - df_sorted['vwap']) / (df_sorted['vwap'] + epsilon)

train_rows = df_sorted.dropna(subset=['target_return']).copy()
print("Train rows with valid target_return:", len(train_rows))

meta_cols = {'asset','timestamp','gap','gaplimit','gap_min','true_next_vwap','next_vwap','target_return'}
numeric_cols = train_rows.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric_cols if c not in meta_cols]
if 'gap_min' not in features:
    features.append('gap_min')
if 'gaplimit' not in features and 'gaplimit' in df_sorted.columns:
    features.append('gaplimit')
features = list(dict.fromkeys(features))
print("Using features:", features)

ensure_dirs()
assets = assets_to_train # Use the previously calculated list
print("Assets to train/evaluate:", assets)

# ------------------ TWO-MODEL-PER-ASSET TRAINING ------------------
asset_models = {}

for asset in assets:
    rows = train_rows[train_rows['asset'] == asset]
    if len(rows) < 10:
        print(f"Skipping {asset} (insufficient training rows: {len(rows)})")
        continue

    # Split into low-gap (<=1h) and high-gap (>1h) buckets
    low_rows = rows[rows['gap_min'] <= 60]
    high_rows = rows[rows['gap_min'] > 60]

    low_model = None
    high_model = None
    
    # --- Data Handling: Forward Fill then Zero Fill for safety ---
    X_low_full = low_rows[features].ffill().fillna(0).values
    y_low_full = low_rows['target_return'].values
    X_high_full = high_rows[features].ffill().fillna(0).values
    y_high_full = high_rows['target_return'].values

    if len(X_low_full) >= 10:
        # --- FIXED: shuffle=False for time series validation ---
        X_tr, X_val, y_tr, y_val = train_test_split(X_low_full, y_low_full, test_size=0.15, random_state=RANDOM_STATE, shuffle=False)
        
        low_model = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS, 
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE
        )
        low_model.fit(X_tr, y_tr)
        
    if len(X_high_full) >= 10:
        # --- FIXED: shuffle=False for time series validation ---
        X_tr, X_val, y_tr, y_val = train_test_split(X_high_full, y_high_full, test_size=0.15, random_state=RANDOM_STATE, shuffle=False)
        
        high_model = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS, 
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE
        )
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
        
        # --- Data Handling: Forward Fill then Zero Fill for safety ---
        X_row = last[feats].to_frame().T.ffill().fillna(0).values.reshape(1,-1)
        
        gap_min = parse_gap_to_minutes(gap)
        
        if gap_min is None: continue

        if gap_min <= 60:
            model = low_model
        else:
            model = high_model
            
        if model is None:
            continue
            
        # Predict RETURN, then convert to PRICE
        pred_return = float(model.predict(X_row)[0])
        last_vwap = float(last['vwap'])
        
        # Final price calculation using predicted return
        pred_price = last_vwap * (1 + pred_return)
        
        pred_records.append({
            'asset': asset,
            'gap': gap,
            'gap_min': gap_min,
            'gaplimit': last.get('gaplimit', np.nan),
            'last_df_timestamp': last['timestamp'],
            'last_df_vwap': last_vwap,
            'pred_next_vwap': pred_price # Final price prediction
        })

pred_df = pd.DataFrame(pred_records)
print("Predictions produced (asset x gap):", len(pred_df))

# ------------------ EVALUATION (FIXED SINGLE-POINT + MAPE) ------------------
eval_rows = []

for _, prow in pred_df.iterrows():
    asset = prow['asset']
    gap = prow['gap']
    last_ts = prow['last_df_timestamp']
    gap_min = prow['gap_min']
    pred_val = prow['pred_next_vwap']
    last_vwap = prow['last_df_vwap']
    
    # Define the exact timestamp we are aiming for
    target_ts = last_ts + timedelta(minutes=int(gap_min))
    
    # Find the actual row in nndf (actual_x) that has the same gap and is closest to the target_ts
    cand = nndf[(nndf['asset'] == asset) & (nndf['gap'] == gap) & (nndf['timestamp'] >= last_ts)].copy()
    
    count_truths = 0
    truth_vwap = np.nan
    rmse = np.nan
    mae = np.nan
    dir_acc = np.nan
    absolute_pct_error = np.nan # New metric

    if not cand.empty:
        # Calculate time difference and find the closest row (closest in time to the target)
        cand['time_diff'] = (cand['timestamp'] - target_ts).abs()
        closest_row = cand.nsmallest(1, 'time_diff').iloc[0]

        # Use the actual VWAP of the closest data point as the "Truth"
        truth_vwap = float(closest_row['vwap'])
        count_truths = 1 

        # Metrics for the single point (Truth vs Pred)
        error = truth_vwap - pred_val
        rmse = np.abs(error)
        mae = np.abs(error)
        
        # --- NEW: Mean Absolute Percentage Error (MAPE) Calculation ---
        # MAPE is the average of the Absolute Percentage Error for each point
        if truth_vwap != 0:
            absolute_pct_error = np.abs(error / truth_vwap) * 100
        else:
            absolute_pct_error = np.nan
        # -------------------------------------------------------------
        
        pred_dir = np.sign(pred_val - last_vwap)
        true_dir = np.sign(truth_vwap - last_vwap)
        dir_acc = 1.0 if pred_dir == true_dir and pred_dir != 0 else 0.0 # Directional accuracy

    eval_rows.append({
        'asset': asset,
        'gap': gap,
        'count_truths': count_truths, # 1 if truth found, 0 otherwise
        'pred_next_vwap': pred_val,
        'true_next_vwap': truth_vwap, # Store truth for verification
        'rmse': float(rmse),
        'mae': float(mae),
        'direction_acc': float(dir_acc),
        'absolute_pct_error': float(absolute_pct_error)
    })

eval_df = pd.DataFrame(eval_rows)

# ------------------- AGGREGATE PER-ASSET -------------------
agg_records = []
for asset, g in eval_df.groupby('asset'):
    valid = g.dropna(subset=['rmse', 'mae', 'direction_acc', 'absolute_pct_error'])
    
    if valid.empty:
        agg_records.append({
            'asset': asset,
            'RMSE': np.nan,
            'MAE': np.nan,
            'MAPE': np.nan, # Added MAPE
            'Direction_Acc': np.nan,
            'Count': int(g['count_truths'].sum())
        })
        continue
    
    # Calculate simple means of the single-point errors/accuracy
    rmse_avg = valid['rmse'].mean()
    mae_avg = valid['mae'].mean()
    mape_avg = valid['absolute_pct_error'].mean() # New aggregation
    diracc_avg = valid['direction_acc'].mean()
    
    agg_records.append({
        'asset': asset,
        'RMSE': float(rmse_avg),
        'MAE': float(mae_avg),
        'MAPE': float(mape_avg), # New metric column
        'Direction_Acc': float(diracc_avg),
        'Count': len(valid)
    })

# --- FIXED: Sort by MAPE instead of RMSE ---
per_asset_metrics_df = pd.DataFrame(agg_records).sort_values('MAPE', ascending=True)

# ------------------- SAVE OUTPUTS -------------------
ensure_dirs()
pred_full = pred_df.merge(eval_df, on=['asset','gap'], how='left')
pred_full.to_csv(PRED_CSV, index=False)
per_asset_metrics_df.to_csv(PER_ASSET_CSV, index=False)
eval_df.to_csv(PER_ASSET_GAP_CSV, index=False)
f=['BTC','ETH','SOL','ZEC','BNB', 'XRP']
# ------------------- FINAL REPORT -------------------
print("\nSaved outputs to:", OUT_DIR)
print("Models created (count assets with >=1 bucket trained):", 
      sum(1 for v in asset_models.values() if (v['low_obj'] is not None) or (v['high_obj'] is not None)))
print("\nTop assets by MAPE (Mean Absolute Percentage Error):")
print("----------------------------------------------------------------------------------------------------")
print("This table is now sorted by MAPE (lowest is best). It provides a fair comparison across all asset prices.")
print("----------------------------------------------------------------------------------------------------")

final_metrics = per_asset_metrics_df[
    per_asset_metrics_df[['RMSE', 'MAE', 'Direction_Acc']].notna().all(axis=1)
].reset_index(drop=True)

# Display only the relevant columns for the final report to emphasize MAPE
print(final_metrics[['asset', 'MAPE', 'RMSE', 'Direction_Acc', 'Count']])

print("\nPer-(asset,gap) eval (top rows for key assets - sorted by Absolute RMSE):")
per_gap_metrics = eval_df[
    eval_df['asset'].isin(f) & 
    eval_df[['rmse', 'mae', 'direction_acc']].notna().all(axis=1)
].reset_index(drop=True).sort_values('rmse', ascending=False)

# Display the same top-rows as before for comparison, but include the new percentage error
print(per_gap_metrics[['asset', 'gap', 'true_next_vwap', 'pred_next_vwap', 'rmse', 'absolute_pct_error', 'direction_acc']])

print("\nPipeline finished.")
