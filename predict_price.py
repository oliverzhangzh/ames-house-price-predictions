"""Predict house sale prices (Ames, Iowa) with scikit-learn.

Data: Kaggle "House Prices - Advanced Regression Techniques" -> train.csv

Usage:
    python predict_price.py train.csv
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Look for files next to this script, not wherever it was launched from
HERE = Path(__file__).resolve().parent
path = sys.argv[1] if len(sys.argv) > 1 else HERE / "train.csv"

# ---------------------------------------------------------------
# 1. Load and look at the data
# ---------------------------------------------------------------
df = pd.read_csv(path).drop(columns=["Id"], errors="ignore")
print(f"{len(df):,} homes, {df.shape[1] - 1} features")
print(df["SalePrice"].describe().round(0))

# Which columns have the most missing values? (worth mentioning in your README)
missing = df.isna().mean().sort_values(ascending=False)
print("\nMost-missing columns:\n", missing[missing > 0].head(8).round(3))

# Drop columns that are mostly empty (in Ames, 'missing' often means "no pool", "no alley", etc.)
too_empty = missing[missing > 0.5].index.tolist()
df = df.drop(columns=too_empty)
print(f"\nDropped {len(too_empty)} columns with >50% missing: {too_empty}")

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------
# 2. Preprocessing: fill gaps, scale numbers, one-hot encode categories
# ---------------------------------------------------------------
prep = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Missing")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

# Prices are right-skewed, so models learn on log(price) and we convert back to dollars
def wrap(model):
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep), ("model", model)]),
        func=np.log1p, inverse_func=np.expm1)

models = {
    "Baseline (always the average price)": DummyRegressor(strategy="mean"),
    "Ridge regression": Ridge(alpha=10),
    "Random forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
}

# ---------------------------------------------------------------
# 3. Train and compare
# ---------------------------------------------------------------
rows, fitted = [], {}
for name, model in models.items():
    m = wrap(model).fit(X_train, y_train)
    pred = m.predict(X_test)
    rows.append({
        "model": name,
        "MAE ($)": mean_absolute_error(y_test, pred),
        "RMSE ($)": root_mean_squared_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    })
    fitted[name] = m

results = pd.DataFrame(rows).set_index("model").round(3)
print("\n", results.to_string())
results.to_csv(HERE / "model_results.csv")

# ---------------------------------------------------------------
# 4. Charts: predicted vs. actual, and which features matter
# ---------------------------------------------------------------
chart_models = ["Ridge regression", "Random forest"]
lims = [y.min(), y.max()]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
for ax, name in zip(axes, chart_models):
    ax.scatter(y_test, fitted[name].predict(X_test), alpha=0.5)
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual sale price ($)")
    ax.set_title(f"{name}: predicted vs. actual")
    ax.legend()
axes[0].set_ylabel("Predicted sale price ($)")
plt.tight_layout()
plt.savefig(HERE / "predicted_vs_actual.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, name in zip(axes, chart_models):
    imp = permutation_importance(fitted[name], X_test, y_test, n_repeats=5,
                                 random_state=0, scoring="r2")
    top = pd.Series(imp.importances_mean, index=X.columns).nlargest(10).sort_values()
    top.plot.barh(ax=ax)
    ax.set_xlabel("Drop in R2 when feature is shuffled")
    ax.set_title(f"{name}: top 10 features")
plt.tight_layout()
plt.savefig(HERE / "feature_importance.png", dpi=150)
print("\nSaved model_results.csv, predicted_vs_actual.png, feature_importance.png")