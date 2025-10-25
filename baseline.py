import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DATA_PATH = "data/clean/(Clean) USA Housing Dataset.csv"
df = pd.read_csv(DATA_PATH)

print(df.head())

drop_cols = [
    "Lot-Living Ratio","Basement Ratio","Areas Per Bedroom",
    "Bathrooms Per Bedroom","Bedrooms Per Floor",
    "Beds x Baths","Sqft Living x Waterfront","date"
]

df = df.drop(columns=drop_cols, errors="ignore")

y = df.pop("price")
X = df

cat_cols = [
    "waterfront","view","condition","city",
    "Day of Week","Season Sold","Is Renovated","State","ZIP Code"
]

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X = X.astype(np.float32)
y = y.astype(np.float32)

y = y.replace([np.inf, -np.inf], np.nan)
y = y.dropna()                            
X = X.loc[y.index]                        

TRAIN, VAL, TEST = 0.80, 0.10, 0.10

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=(1 - TRAIN), random_state=42, shuffle=True
)

rel_test = TEST / (VAL + TEST)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=rel_test, random_state=42, shuffle=True
)

x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

model = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=1e-4,
    random_state=42,
    warm_start=False,
    max_iter=2000
)

model.fit(X_train_scaled, y_train_scaled)

y_pred_train_scaled = model.predict(X_train_scaled)
y_pred_val_scaled = model.predict(X_val_scaled)
y_pred_test_scaled = model.predict(X_test_scaled)

y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
y_pred_val = y_scaler.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).ravel()
y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

def summarize(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{label} → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
    return mae, rmse, r2

print("\n=== Performance Summary ===")
train_metrics = summarize(y_train, y_pred_train, "Train")
val_metrics   = summarize(y_val,   y_pred_val,   "Validation")
test_metrics  = summarize(y_test,  y_pred_test,  "Test")

joblib.dump(model, "baseline_model.pkl")

loaded_model = joblib.load("baseline_model.pkl")

y_pred_val = loaded_model.predict(X_val_scaled)

y_pred_test_scaled = model.predict(X_test_scaled)

print("Sample predictions:", y_pred_val[:5])

joblib.dump({
    "model": model,
    "scaler": x_scaler
}, "baseline_bundle.pkl")

bundle = joblib.load("baseline_bundle.pkl")
loaded_model = bundle["model"]
loaded_scaler = bundle["scaler"]

sample_pred = loaded_model.predict(X_val_scaled[:5])
print("Bundle verification passed | Sample prediction:", sample_pred)

y_pred_test = y_scaler.inverse_transform(
    model.predict(X_test_scaled).reshape(-1, 1)
).ravel()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.title("Predicted vs Actual (Test Set)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid(True, alpha=0.3)
plt.show()