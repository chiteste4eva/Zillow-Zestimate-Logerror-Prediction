"""Feature engineering for the Zillow logerror project.

build_features(df) -> (X, y, cat_cols)
  df: merged train+properties frame with parcelid, logerror, transactiondate.
"""
import numpy as np
import pandas as pd


def build_features(df):
    d = df.copy()
    d["transactiondate"] = pd.to_datetime(d["transactiondate"])

    # transaction timing
    d["trans_month"] = d.transactiondate.dt.month
    d["trans_quarter"] = d.transactiondate.dt.quarter

    # property age
    d["property_age"] = 2016 - d["yearbuilt"]

    # size / value ratios
    d["sqft_per_bedroom"] = d["finishedsquarefeet12"] / (d["bedroomcnt"] + 1)
    d["sqft_per_bathroom"] = d["finishedsquarefeet12"] / (d["bathroomcnt"] + 1)
    d["tax_per_sqft"] = d["taxamount"] / (d["finishedsquarefeet12"] + 1)
    d["lot_to_living_ratio"] = d["lotsizesquarefeet"] / (d["finishedsquarefeet12"] + 1)
    d["total_rooms"] = d["bedroomcnt"].fillna(0) + d["bathroomcnt"].fillna(0)

    # log transforms of right-skewed money/size columns
    for c in ["finishedsquarefeet12", "lotsizesquarefeet", "taxamount",
              "taxvaluedollarcnt", "landtaxvaluedollarcnt",
              "structuretaxvaluedollarcnt", "calculatedfinishedsquarefeet"]:
        if c in d.columns:
            d[f"log_{c}"] = np.log1p(d[c].clip(lower=0))

    # drop ultra-sparse columns (>90% missing)
    miss = d.isna().mean()
    d = d.drop(columns=miss[miss > 0.90].index.tolist())

    # categoricals: id-like columns + leftover strings -> category dtype
    cat_cols = [c for c in d.columns if c.endswith("id") or c in
                {"fips", "propertyzoningdesc", "trans_month", "trans_quarter"}]
    cat_cols = [c for c in cat_cols if c in d.columns]
    obj_cols = [c for c in d.columns
                if pd.api.types.is_string_dtype(d[c].dtype)]
    cat_cols = sorted(set(cat_cols) | set(obj_cols))
    for c in cat_cols:
        d[c] = d[c].astype("category")

    X = d.drop(columns=["parcelid", "transactiondate", "logerror"])
    y = d["logerror"].values
    return X, y, cat_cols
