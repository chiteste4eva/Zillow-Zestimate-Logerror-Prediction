"""Train + evaluate models for Zillow logerror prediction.

Reads data/merged.csv (written by eda.py), builds features via features.py,
trains on an 80/20 split (seed 42):
  a. median baseline
  b. LightGBM (early stopping on validation MAE)
  c. XGBoost (early stopping on validation MAE)
Evaluates MAE / RMSE / R^2 on validation. Writes outputs/metrics_comparison.csv,
feature-importance plot and predicted-vs-actual scatter for the best model.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import build_features

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(SEED)

print("Loading data/merged.csv and building features ...")
df = pd.read_csv(os.path.join(DATA, "merged.csv"))
X, y, cat_cols = build_features(df)
print(f"  feature matrix: {X.shape}, categoricals: {len(cat_cols)}")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=SEED)
print(f"  train {X_train.shape}, valid {X_valid.shape}")

def scores(name, pred):
    mae = mean_absolute_error(y_valid, pred)
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    r2 = r2_score(y_valid, pred)
    print(f"  {name:15s} MAE={mae:.5f} RMSE={rmse:.5f} R2={r2:.5f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}

results = []

# ---------------- a. median baseline ----------------
med = float(np.median(y_train))
print(f"\nBaseline: predict train median {med:.5f}")
results.append(scores("median_baseline",
                       np.full_like(y_valid, med, dtype=float)))

cat = [c for c in cat_cols if c in X.columns]

# ---------------- b. LightGBM ----------------
import lightgbm as lgb
print("\nTraining LightGBM ...")
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train,
                        categorical_feature=cat)
lgb_params = dict(objective="regression", metric="mae", seed=SEED,
                  num_leaves=64, learning_rate=0.03,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  min_data_in_leaf=100, verbose=-1)
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=3000,
                      valid_sets=[lgb_valid],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
pred_lgb = lgb_model.predict(X_valid)
results.append(scores("lightgbm", pred_lgb))

# ---------------- c. XGBoost ----------------
import xgboost as xgb
print("\nTraining XGBoost ...")
def to_codes(frame):
    f = frame.copy()
    for c in f.columns:
        if isinstance(f[c].dtype, pd.CategoricalDtype):
            f[c] = f[c].cat.codes.replace(-1, np.nan)
    return f
xgb_train = xgb.DMatrix(to_codes(X_train), label=y_train)
xgb_valid = xgb.DMatrix(to_codes(X_valid), label=y_valid)
xgb_params = dict(objective="reg:squarederror", eval_metric="mae",
                  seed=SEED, max_depth=6, eta=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=10, tree_method="hist")
xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=2000,
                      evals=[(xgb_valid, "valid")],
                      early_stopping_rounds=100, verbose_eval=False)
pred_xgb = xgb_model.predict(xgb_valid)
results.append(scores("xgboost", pred_xgb))

# ---------------- save metrics ----------------
metrics = pd.DataFrame(results)
metrics.to_csv(os.path.join(OUT, "metrics_comparison.csv"), index=False)
print("\nsaved outputs/metrics_comparison.csv")
print(metrics.to_string(index=False))

best = metrics.loc[metrics.mae.idxmin()]
print(f"\nBest model by validation MAE: {best['model']} "
      f"(MAE={best['mae']:.5f})")

# ---------------- feature importance (best model) ----------------
if best["model"] == "lightgbm":
    imp = pd.Series(lgb_model.feature_importance(importance_type="gain"),
                    index=X.columns).sort_values(ascending=False)
    best_pred = pred_lgb
elif best["model"] == "xgboost":
    raw = xgb_model.get_score(importance_type="gain")
    imp = pd.Series({k: raw.get(k, 0.0) for k in X.columns}
                    ).sort_values(ascending=False)
    best_pred = pred_xgb
else:
    imp = pd.Series(dtype=float)
    best_pred = np.full_like(y_valid, med, dtype=float)

imp.head(20).to_csv(os.path.join(OUT, "feature_importance.csv"),
                     header=["gain"])
top = imp.head(15)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(np.arange(len(top)), top.values, color="#c4703f")
ax.set_yticks(np.arange(len(top)))
ax.set_yticklabels(top.index)
ax.invert_yaxis()
ax.set_xlabel("gain importance")
ax.set_title(f"Top features — {best['model']}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "feature_importance.png"), dpi=120)
plt.close(fig)
print("saved outputs/feature_importance.png")

# ---------------- predicted vs actual ----------------
fig, ax = plt.subplots(figsize=(6, 6))
idx = rng.choice(len(y_valid), size=min(8000, len(y_valid)), replace=False)
ax.scatter(y_valid[idx], best_pred[idx], s=4, alpha=0.25, color="#262522")
lim = [min(y_valid.min(), best_pred.min()), max(y_valid.max(), best_pred.min())]
ax.plot(lim, lim, color="#c4703f", lw=1.5, ls="--")
ax.set_xlabel("actual logerror")
ax.set_ylabel("predicted logerror")
ax.set_title(f"Predicted vs actual — {best['model']} (validation)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "predicted_vs_actual.png"), dpi=120)
plt.close(fig)
print("saved outputs/predicted_vs_actual.png")

print("\nTop 5 features:")
print(imp.head(5).to_string())
print("\nDone.")
