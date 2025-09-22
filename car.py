import pandas as pd
import numpy as np
import re
import pickle
import json
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from category_encoders import TargetEncoder
from xgboost import XGBRegressor

# ----------------------
# Helper cleaning funcs
# ----------------------
def extract_number(value):
    if pd.isna(value):
        return np.nan
    num = re.findall(r"[0-9.]+", str(value))
    return float(num[0]) if num else np.nan

def clean_new_price(val):
    if pd.isna(val):
        return np.nan
    val = str(val)
    if "Lakh" in val:
        return float(val.replace("Lakh", "").strip())
    if "Cr" in val:
        return float(val.replace("Cr", "").strip()) * 100
    return np.nan

# ----------------------
# Load & clean dataset
# ----------------------
df = pd.read_csv("used_cars_data_updated.csv")

# Split Name -> Brand, Model, Variant
name_split = df["Name"].str.split(" ", n=2, expand=True)
df["Brand"], df["Model"], df["Variant"] = name_split[0], name_split[1], name_split[2]

# Clean numeric columns
df["Mileage"] = df["Mileage"].apply(extract_number)
df["Engine"] = df["Engine"].apply(extract_number)
df["Power"] = df["Power"].apply(extract_number)
df["Seats"] = df["Seats"].fillna(df["Seats"].mode()[0])
df["New_Price"] = df["New_Price"].apply(clean_new_price)

# Drop unused + missing price rows
df = df.drop(["S.No.", "Name"], axis=1)
df = df.dropna(subset=["Price"])

# ----------------------
# Feature Engineering
# ----------------------
df["Car_Age"] = 2025 - df["Year"]
df["Kilometers_Driven"] = np.log1p(df["Kilometers_Driven"])
df = df.drop("Year", axis=1)

# ----------------------
# Features & Target
# ----------------------
X = df.drop("Price", axis=1)
y = np.log1p(df["Price"])  # log target

# Feature groups
onehot_cols = ["Location", "Fuel_Type", "Transmission", "Owner_Type"]
targetenc_cols = ["Brand", "Model", "Variant"]
numeric_cols = [col for col in X.columns if col not in onehot_cols + targetenc_cols]

# Preprocessor
numeric_transformer = SimpleImputer(strategy="median")
onehot_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
targetenc_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("targetenc", TargetEncoder())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("onehot", onehot_transformer, onehot_cols),
    ("target", targetenc_transformer, targetenc_cols)
])

# ----------------------
# Optuna objective function
# ----------------------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "n_jobs": -1
    }

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(**params))
    ])

    # Use 3-fold cross-validation on R²
    scores = cross_val_score(model, X, y, cv=3, scoring=make_scorer(r2_score))
    return scores.mean()

# ----------------------
# Run Optuna study
# ----------------------
print(" Running Optuna hyperparameter tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, n_jobs=1)  # you can increase n_trials for better tuning

best_params = study.best_trial.params
print(f"\n Best Params: {best_params}")

# ----------------------
# Train final model with best params
# ----------------------
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(**best_params))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)

preds = np.expm1(final_model.predict(X_val))
true_vals = np.expm1(y_val)

mae, r2 = mean_absolute_error(true_vals, preds), r2_score(true_vals, preds)
print(f"\n Final XGBoost: MAE={mae:.2f}, R²={r2:.3f}")

# ----------------------
# Save outputs
# ----------------------
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("car_price_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("model_metrics.json", "w") as f:
    json.dump({"XGBoost": {"MAE": mae, "R2": r2}}, f, indent=4)

print(" Model + metrics saved")

# ----------------------
# Feature Importance
# ----------------------
feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
importances = final_model.named_steps["model"].feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n Top 20 Most Important Features:")
for idx in sorted_idx[:20]:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")
