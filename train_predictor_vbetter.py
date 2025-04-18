import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score


# === Start timer ===
start_time = time.time()

print("📥 Loading dataset...")
df = pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")
print(f"✅ Loaded {len(df):,} rows.")

# === Prepare dataset ===
df["season"] = df["season"].astype(int)
df = df[df["Valeur marchande (euros)"].notna()]
df = df.sort_values(["player", "season"])
df["next_market_value"] = df.groupby("player")["Valeur marchande (euros)"].shift(-1)

# Drop last known rows
df_model = df[df["next_market_value"].notna()].copy()
print(f"📊 Rows with valid target: {len(df_model):,}")

# === Feature engineering ===

# Lag features
df_model["mv_lag1"] = df_model.groupby("player")["Valeur marchande (euros)"].shift(1)
df_model["rolling_mv_2"] = df_model.groupby("player")["Valeur marchande (euros)"].rolling(2).mean().shift(1).reset_index(0, drop=True)
df_model["rolling_mv_3"] = df_model.groupby("player")["Valeur marchande (euros)"].rolling(3).mean().shift(1).reset_index(0, drop=True)

# Interaction features
df_model["gls_per_90"] = df_model["Performance Gls"] / df_model["90s"]
df_model["market_age_ratio"] = df_model["Valeur marchande (euros)"] / df_model["age"]

# Drop rows with any new NaNs
df_model = df_model.dropna(subset=["mv_lag1", "rolling_mv_2", "rolling_mv_3", "gls_per_90", "market_age_ratio"])

# === Define features and target ===
X = df_model.drop(columns=["next_market_value", "player", "born", "Valeur marchande (euros)"], errors="ignore")
y = df_model["next_market_value"]

# === Column types ===
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# === Preprocessing ===
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ]), cat_cols)
])

# === XGBoost regressor and pipeline ===
xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb)
])

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(f"🔀 Train: {len(X_train):,}, Test: {len(X_test):,}")

# === Hyperparameter tuning ===
param_dist = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [3, 5, 7, 10],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,
    scoring="r2",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("🚀 Tuning and training XGBoost model...")
search.fit(X_train, y_train)

# === Evaluate ===
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Absolute Error (MAE): €{mae:,.0f}")
print(f"📈 R² Score: {r2:.4f}")

# === Save model ===
model_path = "player_value_predictor_final_xgb.joblib"
dump((best_model, cat_cols, num_cols), model_path)
print(f"💾 Model saved to {model_path}")

elapsed = time.time() - start_time
print(f"🎉 All done in {elapsed:.1f} seconds.")
