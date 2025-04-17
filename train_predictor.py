import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump

# Load the dataset
df = pd.read_csv("Data_merged_with_market_value/all_leagues_merged_final_df.csv")

# Drop rows with missing target
df = df[df["Valeur marchande (euros)"].notna()]

# Drop player identifiers for model training
X = df.drop(columns=["Valeur marchande (euros)", "player", "born"])
y = df["Valeur marchande (euros)"]

# Identify column types
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=50, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save model
dump((pipeline, cat_cols, num_cols), "player_value_predictor.joblib")
print("✅ Model trained and saved as 'player_value_predictor.joblib'")
