"""EDA for Zillow Zestimate logerror project.

Reads data/, merges train_2016_v2 with only the needed parcelids from
properties_2016 (chunked, to stay fast on the 648MB file), writes the merged
frame to data/merged.csv for reuse, and saves EDA plots to outputs/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

print("Loading train_2016_v2.csv ...")
train = pd.read_csv(os.path.join(DATA, "train_2016_v2.csv"),
                    parse_dates=["transactiondate"])
print(f"  train shape: {train.shape}")
print(f"  logerror: mean={train.logerror.mean():.4f} std={train.logerror.std():.4f} "
      f"min={train.logerror.min():.4f} max={train.logerror.max():.4f}")
print(f"  unique parcelids: {train.parcelid.nunique()}")

needed = set(train.parcelid.unique())
print(f"Reading properties_2016.csv in chunks, keeping {len(needed)} parcelids ...")
chunks = []
for chunk in pd.read_csv(os.path.join(DATA, "properties_2016.csv"),
                         chunksize=250_000, low_memory=False):
    hit = chunk[chunk.parcelid.isin(needed)]
    if len(hit):
        chunks.append(hit)
props = pd.concat(chunks, ignore_index=True)
print(f"  matched properties rows: {len(props)} / {len(train)} train rows")

merged = train.merge(props, on="parcelid", how="left")
print(f"  merged shape: {merged.shape}")
merged.to_csv(os.path.join(DATA, "merged.csv"), index=False)
print("  saved data/merged.csv")

# ---- missing values ----
miss = merged.isna().mean().sort_values(ascending=False)
print("\nTop 15 columns by missing fraction:")
print(miss.head(15).to_string())
miss.to_csv(os.path.join(OUT, "missing_fractions.csv"), header=["missing_frac"])

# ---- logerror distribution ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.hist(merged.logerror, bins=120, color="#c4703f", edgecolor="none")
ax.set_title("logerror distribution (train_2016_v2)")
ax.set_xlabel("logerror = log(Zestimate) - log(SalePrice)")
ax.set_ylabel("count")
ax = axes[1]
ax.boxplot(merged.logerror, vert=True)
ax.set_title("logerror boxplot")
ax.set_ylabel("logerror")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "logerror_distribution.png"), dpi=120)
plt.close(fig)
print("saved outputs/logerror_distribution.png")

# ---- correlations of numeric features with logerror ----
num = merged.select_dtypes(include=[np.number]).drop(columns=["parcelid"])
corr = num.corr()["logerror"].drop("logerror").sort_values(key=abs, ascending=False)
print("\nTop 15 features by |correlation| with logerror:")
print(corr.head(15).to_string())
corr.to_csv(os.path.join(OUT, "correlations.csv"), header=["corr_with_logerror"])

top = corr.head(15)
fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(top))
ax.barh(y, top.values, color="#c4703f")
ax.set_yticks(y)
ax.set_yticklabels(top.index)
ax.invert_yaxis()
ax.set_xlabel("correlation with logerror")
ax.set_title("Top features by correlation with logerror")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "top_correlations.png"), dpi=120)
plt.close(fig)
print("saved outputs/top_correlations.png")

# ---- transaction date spread ----
print("\ntransactiondate range:",
      merged.transactiondate.min(), "->", merged.transactiondate.max())
print("\nEDA done.")
