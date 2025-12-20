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
 
 
 
===================================
 
 
 
Assets by data count (Filtered for Eval): [[11928, 'BTC'], [11902, 'ETH'], [11333, 'SOL'], [11325, 'ZEC'], [10970, 'XRP'], [10804, 'BNB'], [5468, 'SUI'], [3745, 'ICP'], [3359, 'DASH'], [3257, 'GIGGLE'], [3236, 'LINK'], [3147, 'LTC'], [2950, 'ZEN'], [2502, 'UNI'], [2491, 'NEAR'], [2450, 'TAO'], [2199, 'FIL'], [1985, 'TRUMP'], [1919, 'AVAX'], [1354, 'PAXG'], [1346, 'BCH'], [859, 'AAVE'], [710, 'AR'], [635, 'DOT'], [532, 'ETC'], [412, 'CAKE'], [410, 'RENDER'], [400, 'ORDI'], [397, 'WBTC'], [379, 'INJ'], [364, 'APT'], [341, 'TON'], [323, 'PENDLE'], [281, 'ATOM'], [257, 'TRB'], [236, 'NMR'], [219, 'ZRO'], [210, 'WBETH'], [100, 'OG'], [97, 'COMP']]
Train rows with valid target_return: 122081
Using features: ['klineacc', 'spread', 'spreadper', 'x', 'vwap', 'deviation', 'ratio', 'term', 'sigma', 'e', 'h', 'gap_min', 'gaplimit']
Assets to train/evaluate: ['AAVE', 'APT', 'AR', 'ATOM', 'AVAX', 'BCH', 'BNB', 'BTC', 'CAKE', 'COMP', 'DASH', 'DOT', 'ETC', 'ETH', 'FIL', 'GIGGLE', 'ICP', 'INJ', 'LINK', 'LTC', 'NEAR', 'NMR', 'OG', 'ORDI', 'PAXG', 'PENDLE', 'RENDER', 'SOL', 'SUI', 'TAO', 'TON', 'TRB', 'TRUMP', 'UNI', 'WBETH', 'WBTC', 'XRP', 'ZEC', 'ZEN', 'ZRO']
Predictions produced (asset x gap): 390

Saved outputs to: eval_outputs_gbm_fixed
Models created (count assets with >=1 bucket trained): 40

Top assets by MAPE (Mean Absolute Percentage Error):
----------------------------------------------------------------------------------------------------
This table is now sorted by MAPE (lowest is best). It provides a fair comparison across all asset prices.
----------------------------------------------------------------------------------------------------
     asset       MAPE        RMSE  Direction_Acc  Count
0      ETC   0.204895    0.027595       1.000000      5
1      SOL   0.242311    0.325600       0.545455     11
2    TRUMP   0.341071    0.019736       0.600000      5
3      BNB   0.358703    3.177175       0.727273     11
4     ATOM   0.388280    0.008697       0.800000      5
5      BTC   0.434661  396.555748       0.636364     11
6     DASH   0.470654    0.228843       0.200000      5
7   RENDER   0.488239    0.007983       0.800000      5
8      ETH   0.525004   16.845574       0.454545     11
9      XRP   0.546715    0.011168       0.818182     11
10     DOT   0.635293    0.013635       1.000000      5
11     ICP   0.672354    0.023370       1.000000      5
12    WBTC   0.873563  797.074426       0.000000      5
13  GIGGLE   0.917473    0.779207       1.000000      5
14    AAVE   1.070353    2.103273       0.000000      5
15     ZEC   1.089065    4.572191       0.909091     11
16     INJ   1.156535    0.064866       1.000000      5
17    CAKE   1.163052    0.026695       1.000000      5
18   WBETH   1.171106   41.141159       0.200000      5
19    PAXG   1.179060   50.254158       0.454545     11
20     APT   1.294015    0.022965       1.000000      5
21     ZRO   1.332610    0.019362       0.200000      5
22     BCH   1.393809    8.068601       0.600000     10
23     SUI   1.471509    0.023605       0.454545     11
24  PENDLE   1.665614    0.039134       1.000000      5
25     ZEN   1.730300    0.163932       0.400000      5
26     NMR   1.755630    0.197387       0.200000      5
27     TON   1.789476    0.029269       0.200000      5
28      AR   2.153580    0.081624       0.800000      5
29    LINK   2.730736    0.378421       0.818182     11
30    ORDI   3.489515    0.145451       0.600000      5
31     LTC   3.964906    3.123024       0.500000      8
32     TRB   4.074901    0.841566       0.400000      5
33     UNI   4.436057    0.228788       1.000000     11
34    AVAX   4.783864    0.630152       0.909091     11
35     FIL   7.441416    0.108113       0.200000      5
36    NEAR  10.534803    0.163035       0.727273     11
37    COMP  12.935108    3.879848       1.000000      5
38     TAO  14.852928   35.505492       0.454545     11
39      OG  25.210703    3.142101       0.000000      4

Per-(asset,gap) eval (top rows for key assets - sorted by Absolute RMSE):
   asset  gap  true_next_vwap  pred_next_vwap         rmse  absolute_pct_error  direction_acc
11   BTC  12h    91577.476562    90425.214499  1152.262064            1.258237            0.0
21   BTC   8h    91531.898438    90496.104571  1035.793867            1.131621            0.0
20   BTC   6h    91582.500000    90716.703948   865.796052            0.945373            0.0
12   BTC  15m    89906.781250    89607.981682   298.799568            0.332344            1.0
13   BTC   1h    89793.203125    90077.929251   284.726126            0.317091            1.0
18   BTC   4h    91401.218750    91122.941684   278.277066            0.304457            0.0
15   BTC   2h    91344.687500    91091.110802   253.576698            0.277604            1.0
16   BTC  30m    89860.250000    89716.420990   143.829010            0.160059            1.0
22   ETH  12h     3223.503662     3185.554939    37.948723            1.177251            0.0
32   ETH   8h     3222.227295     3186.020227    36.207068            1.123666            0.0
19   BTC   5m    90022.867188    90050.956440    28.089252            0.031202            1.0
23   ETH  15m     3180.890137     3156.019968    24.870169            0.781862            1.0
27   ETH  30m     3178.109131     3155.874473    22.234658            0.699619            1.0
31   ETH   6h     3229.648926     3208.561299    21.087627            0.652939            0.0
29   ETH   4h     3215.084473     3197.551446    17.533027            0.545336            0.0
55   ZEC  12h      423.566895      407.955137    15.611758            3.685783            1.0
26   ETH   2h     3214.173096     3199.432428    14.740668            0.458615            0.0
17   BTC   3m    90006.898438    89993.706536    13.191901            0.014657            1.0
10   BNB   8h      890.379456      880.259100    10.120356            1.136634            1.0
9    BNB   6h      893.146057      885.003451     8.142606            0.911677            1.0
65   ZEC   8h      418.778656      410.756462     8.022194            1.915617            1.0
64   ZEC   6h      412.633240      404.783021     7.850219            1.902469            1.0
14   BTC   1m    90029.171875    90036.943503     7.771628            0.008632            1.0
62   ZEC   4h      407.126740      400.028444     7.098296            1.743510            1.0
0    BNB  12h      888.337646      881.265920     7.071727            0.796063            1.0
60   ZEC  30m      446.536255      441.898210     4.638044            1.038671            0.0
59   ZEC   2h      401.364136      397.053223     4.310913            1.074065            1.0
24   ETH   1h     3174.722168     3170.496628     4.225540            0.133100            1.0
25   ETH   1m     3185.333740     3182.067628     3.266112            0.102536            1.0
5    BNB  30m      865.521240      868.626873     3.105633            0.358817            0.0
28   ETH   3m     3183.606201     3186.511345     2.905143            0.091253            0.0
7    BNB   4h      892.203247      890.225473     1.977774            0.221673            1.0
1    BNB  15m      865.434753      863.677678     1.757076            0.203028            1.0
33   SOL  12h      135.421677      133.987858     1.433818            1.058780            0.0
56   ZEC  15m      446.255219      445.030116     1.225102            0.274530            1.0
2    BNB   1h      865.259216      864.247811     1.011406            0.116890            1.0
43   SOL   8h      135.533096      134.667096     0.866000            0.638959            0.0
61   ZEC   3m      445.446564      444.636753     0.809810            0.181797            1.0
4    BNB   2h      893.214905      893.950144     0.735239            0.082314            1.0
57   ZEC   1h      445.902802      445.270024     0.632778            0.141909            1.0
8    BNB   5m      865.739624      866.312724     0.573100            0.066198            0.0
34   SOL  15m      131.162674      130.806689     0.355985            0.271407            0.0
3    BNB   1m      865.794189      866.140113     0.345924            0.039954            0.0
30   ETH   5m     3184.851318     3185.133895     0.282576            0.008873            1.0
35   SOL   1h      131.216949      131.477424     0.260475            0.198507            1.0
42   SOL   6h      136.048813      135.875914     0.172899            0.127086            0.0
40   SOL   4h      135.568497      135.413425     0.155072            0.114387            1.0
41   SOL   5m      131.194092      131.341967     0.147875            0.112715            1.0
38   SOL  30m      131.321625      131.202706     0.118919            0.090555            1.0
6    BNB   3m      865.638306      865.746389     0.108084            0.012486            1.0
58   ZEC   1m      444.587402      444.517598     0.069805            0.015701            1.0
39   SOL   3m      131.215485      131.254955     0.039470            0.030081            0.0
44   XRP  12h        2.054189        2.017194     0.036995            1.800968            1.0
37   SOL   2h      135.693329      135.723123     0.029794            0.021957            1.0
54   XRP   8h        2.057851        2.031018     0.026832            1.303906            1.0
63   ZEC   5m      444.291656      444.316837     0.025180            0.005667            1.0
53   XRP   6h        2.064876        2.049694     0.015182            0.735242            1.0
46   XRP   1h        1.994150        2.008145     0.013994            0.701778            0.0
51   XRP   4h        2.063388        2.051785     0.011604            0.562353            1.0
49   XRP  30m        1.995838        1.987183     0.008655            0.433661            1.0
45   XRP  15m        1.997515        1.992280     0.005236            0.262115            1.0
48   XRP   2h        2.067385        2.065157     0.002228            0.107767            1.0
36   SOL   1m      131.201828      131.200531     0.001297            0.000989            1.0
52   XRP   5m        2.001107        2.002305     0.001198            0.059881            0.0
47   XRP   1m        2.001120        2.000588     0.000532            0.026570            1.0
50   XRP   3m        1.999967        1.999574     0.000392            0.019622            1.0

Pipeline finished.
