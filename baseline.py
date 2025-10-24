"""
Baseline Model for Housing Price Prediction Model

Several things need to be done by this script:
    1. Load the data
    2. Drop any data that is nonlinear
    3. One-Hot Encode Categorical Data (includes "rating" columns like 'view')
    4. Split data into training, validation, and testing.
    5. Train SGDRegressor up to 200 epochs (or more), recording MAE, RMSE, and R^2 after every epoch.
    6. Stop training early when validation MAE stops improving.
    7. Evaluate final model on the test set, print metrics
    8. Plot training/validation metrics (3 sub-plots) and save figures to 'results/[metric]_plot.png'
        [metric] = specific metric being plotted
        You can also save them all to one singular png if you can figure that out too.
    9. Save trained model ('models/baseline.joblib') and a JSON file containing all metrics ('results/baseline.json')
"""

# Imports
import json  # To dump metrics into a file
import os  # File managing

import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Math functions
import pandas as pd  # Data processing
from joblib import dump  # Saving model
from sklearn.linear_model import SGDRegressor  # Used for model
from sklearn.metrics import mean_absolute_error  # Metric calc
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # Data splitting

# |--------------|
# | Load Dataset |
# |--------------|
DATA_PATH = "data/clean/(Clean) USA Housing Dataset.csv"  # Path to file
df = pd.read_csv(DATA_PATH)  # Reading file

# |----------------------------|
# | Remove Interaction Columns |
# |----------------------------|
interaction_cols = [
    "Lot-Living Ratio",
    "Basement Ratio",
    "Areas Per Bedroom",
    "Bathrooms Per Bedroom",  # |--- List of interaction columns
    "Bedrooms Per Floor",
    "Beds x Baths",
    "Sqft Living x Waterfront",
]
df = df.drop(columns=interaction_cols)  # Drop interaction_cols
df = df.drop(columns=["date"])  # We cannot convert `date` to float32

# |-------------------------------|
# | Separate target from features |
# |-------------------------------|
y = df.pop("price")  # Pop removes it from dataframe while saving it to `y`

# |---------------------------|
# | One-Hot Encode Categories |
# |---------------------------|
cat_cols = [
    "waterfront",
    "view",
    "condition",
    "city",
    "Day of Week",  # |---    List of categorical columns
    "Season Sold",
    "Is Renovated",
    "State",
    "ZIP Code",
]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # Drop cat_cols

# |--------------------|
# | Convert to float32 |
# |--------------------|
X = df.astype(np.float32)  # Predictors
y = y.astype(np.float32)  # Target

# |---------------------------------|
# | Train/Validation/Testing Splits |
# |---------------------------------|
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42  # 70/30 Split
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42  # 70/15/15 Split
)

# |--------------|
# | SGDRegressor |
# |--------------|
model = SGDRegressor(
    loss="squared_error",  # In the slides I had this as `squared_loss`, I meant this
    penalty=None,  # No penalty to keep it purely linear
    learning_rate="constant",  # Can change this to `optimal` to make smoother changes in metrics
    eta0=1e-3,  # Sweet spot where it doesn't diverge and converges quickly
    max_iter=200,  # 200 epochs
    random_state=42,
    warm_start=True,  # Allows for continual calls of .partial_fit()
)

# |---------------|
# | Training Loop |
# |---------------|
n_epochs = 200  # 200 Epochs

# Initializing lists for metrics
train_mae, val_mae = [], []
train_rmse, val_rmse = [], []
train_r2, val_r2 = [], []

for epoch in range(n_epochs):
    model.partial_fit(
        X_train, y_train
    )  # `warm_state=True` allows this to work on previous epoch

    y_pred_train = model.predict(X_train)  # Training
    y_pred_val = model.predict(X_val)  # Validation

    # Save MAE
    train_mae.append(mean_absolute_error(y_train, y_pred_train))
    val_mae.append(mean_absolute_error(y_val, y_pred_val))

    # Save RMSE
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    val_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    # Save R^2
    train_r2.append(r2_score(y_train, y_pred_train))
    val_r2.append(r2_score(y_val, y_pred_val))

    # Early stopping
    # if epoch > 0 and val_mae[-1] > min(val_mae[:-1]):
    #    print(f"Early stopping after epoch {epoch + 1}")
    #    break

# |------------|
# | Final Eval |
# |------------|
y_pred_test = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("\n=== Test Set Performance ===")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test R^2: {test_r2:.4f}")

# |--------------------------|
# | Plot metrics over epochs |
# |--------------------------|
epochs = range(1, len(train_mae) + 1)

plt.figure(figsize=(12, 4))

# MAE
plt.subplot(1, 3, 1)
plt.plot(epochs, train_mae, label="Train MAE")
plt.plot(epochs, val_mae, label="Val MAE", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Mean Absolute Error")
plt.legend()

# RMSE
plt.subplot(1, 3, 2)
plt.plot(epochs, train_rmse, label="Train RMSE")
plt.plot(epochs, val_rmse, label="Val RMSE", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Root Mean Squared Error")
plt.legend()

# R^2
plt.subplot(1, 3, 3)
plt.plot(epochs, train_r2, label="Train R^2")
plt.plot(epochs, val_r2, label="Val R^2", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("R^2")
plt.title("Coefficient of Determination")
plt.legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/metrics_plot.png")

# |------------------------|
# | Save Model and Metrics |
# |------------------------|
os.makedirs("models", exist_ok=True)
dump(model, "models/baseline.joblib")

baseline_results = {
    "test_mae": test_mae,
    "test_rmse": test_rmse,
    "test_r2": test_r2,
    "train_mae_per_epoch": train_mae,
    "val_mae_per_epoch": val_mae,
    "train_rmse_per_epoch": train_rmse,
    "val_rmse_per_epoch": val_rmse,
    "train_r2_per_epoch": train_r2,
    "val_r2_per_epoch": val_r2,
}

with open("results/baseline.json", "w") as f:
    json.dump(baseline_results, f, indent=4)

print("\nBaseline model & metrics saved in `models/` and `results/`.")
