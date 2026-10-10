"""Quick LightGBM config comparison on the validation split (seed 42).

Not part of the final pipeline — used once to pick sensible hyperparameters.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from features import build_features

SEED = 42
df = pd.read_csv("data/merged.csv")
X, y, cat_cols = build_features(df)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=SEED)
cat = [c for c in cat_cols if c in X.columns]

configs = {
    "A_orig": dict(num_leaves=96, learning_rate=0.05, min_data_in_leaf=60,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1),
    "B_small_lr": dict(num_leaves=64, learning_rate=0.03, min_data_in_leaf=100,
                       feature_fraction=0.9, bagging_fraction=0.8,
                       bagging_freq=1),
    "C_classic": dict(num_leaves=31, learning_rate=0.05, min_data_in_leaf=20,
                      feature_fraction=0.9, bagging_fraction=0.8,
                      bagging_freq=5),
}
for name, extra in configs.items():
    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain,
                         categorical_feature=cat)
    params = dict(objective="regression", metric="mae", seed=SEED,
                  verbose=-1, **extra)
    model = lgb.train(params, dtrain, num_boost_round=3000,
                      valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    print(f"{name:10s} MAE={mae:.5f}  best_iter={model.best_iteration}")
