import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
import time

start_time = time.time()

print("📥 Loading dataset...")
df = pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")
print(f"✅ Loaded {len(df):,} rows.")

print("📤 Preparing features and target...")
X = df.drop(columns=["Valeur marchande (euros)", "player", "born"], errors="ignore")
y = df["Valeur marchande (euros)"]

print("🧠 Identifying column types...")
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
print(f"🔣 Categorical columns: {cat_cols}")
print(f"🔢 Numerical columns: {num_cols}")

print("🧼 Creating preprocessing pipeline...")
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

print("🔁 Building training pipeline with RandomForest...")
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=50, random_state=42))
])

print("🔀 Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(f"📊 Train size: {len(X_train):,}, Test size: {len(X_test):,}")

print("🚀 Training model...")
pipeline.fit(X_train, y_train)
print("✅ Model training complete.")

print("💾 Saving model to 'player_value_predictor.joblib'...")
dump((pipeline, cat_cols, num_cols), "player_value_predictor_bis.joblib")

end_time = time.time()
elapsed = end_time - start_time
print(f"🎉 All done in {elapsed:.1f} seconds.")
