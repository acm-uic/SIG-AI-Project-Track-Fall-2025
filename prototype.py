# Imports
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set random seed
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load Dataset
DATA_PATH = Path("data/clean/(Clean) USA Housing Dataset.csv")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

SEQ_LEN = 30  # Number of past days used to predict next price
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 8

# Helper Functions


def create_sequences(X, y, seq_len: int):
    n_samples = X.shape[0]
    Xs, ys = [], []
    for i in range(n_samples - seq_len):
        X_slice = X[i : i + seq_len]
        if hasattr(X_slice, "toarray"):
            X_slice = X_slice.toarray()
        else:
            X_slice = np.array(X_slice)

        Xs.append(X_slice)
        ys.append(y[i + seq_len])

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class PriceLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# Load and Preprocess Data

df = pd.read_csv(DATA_PATH)

# Sort values so we dont show the model future dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].isna().any():
    raise ValueError("Some dates could not be parse; check the column format.")

df.sort_values("date", inplace=True)
df.drop(columns=["State", "ZIP Code"], inplace=True)

numeric_cols = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "House Age",
    "Renovation Age",
    "Lot-Living Ratio",
    "Basement Ratio",
    "Areas Per Bedroom",
    "Bathrooms Per Bedroom",
    "Bedrooms Per Floor",
    "Beds x Baths",
    "Sqft Living x Waterfront",
]

categorical_cols = [
    "waterfront",
    "view",
    "condition",
    "city",
    "Day of Week",
    "Season Sold",
    "Is Renovated",
]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_cols,
        ),  # Scales every value to (value - mean) / std
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols,
        ),  # One-hot encodes all categories
    ]
)

y = df["price"]
X = df.drop(columns=["price", "date"])  # We'll keep date for time-index

X_processed = preprocess.fit_transform(X)

# Train/Val Spluit (70/30)
split_date = df["date"].quantile(0.7)
train_mask = df["date"] <= split_date
val_mask = df["date"] > split_date

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]

X_train_raw = X_processed[train_idx]
y_train_raw = y.iloc[train_idx].to_numpy()

X_val_raw = X_processed[val_idx]
y_val_raw = y.iloc[val_idx].to_numpy()

# Create Sequences
X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, SEQ_LEN)
X_val_seq, y_val_seq = create_sequences(X_val_raw, y_val_raw, SEQ_LEN)

# Drop any sample with NaN in the target
train_feat_mask = ~np.isnan(X_train_seq).any(axis=(1, 2))
X_train_seq, y_train_seq = X_train_seq[train_feat_mask], y_train_seq[train_feat_mask]

val_feat_mask = ~np.isnan(X_val_seq).any(axis=(1, 2))
X_val_seq, y_val_seq = X_val_seq[val_feat_mask], y_val_seq[val_feat_mask]

# Make sure directories exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Build Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PriceLSTM(input_dim=X_train_seq.shape[2]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
best_val_loss = float("inf")
counter = 0

train_mae_hist, val_mae_hist = [], []
train_rmse_hist, val_rmse_hist = [], []
train_r2_hist, val_r2_hist = [], []

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()  # Enables dropout
    for i in range(0, len(X_train_seq), BATCH_SIZE):
        xb = torch.tensor(X_train_seq[i : i + BATCH_SIZE], device=device)
        yb = torch.tensor(y_train_seq[i : i + BATCH_SIZE], device=device).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(X_train_seq, device=device))
        train_pred_np = train_pred.cpu().numpy().flatten()
        y_train_np = y_train_seq

        # Metrics
        train_mae = mean_absolute_error(y_train_np, train_pred_np)
        train_rmse = np.sqrt(mean_squared_error(y_train_np, train_pred_np))
        train_r2 = r2_score(y_train_np, train_pred_np)

        val_pred = model(torch.tensor(X_val_seq, device=device))
        val_pred_np = val_pred.cpu().numpy().flatten()
        y_val_np = y_val_seq

        # Metrics
        val_mae = mean_absolute_error(y_val_np, val_pred_np)
        val_rmse = np.sqrt(mean_squared_error(y_val_np, val_pred_np))
        val_r2 = r2_score(y_val_np, val_pred_np)

        # Store
        train_mae_hist.append(train_mae)
        train_rmse_hist.append(train_rmse)
        train_r2_hist.append(train_r2)

        val_mae_hist.append(val_mae)
        val_rmse_hist.append(val_rmse)
        val_r2_hist.append(val_r2)

        # Early Stopping
        val_loss = criterion(
            torch.tensor(val_pred_np, device=device).unsqueeze(1),
            torch.tensor(y_val_np, device=device).unsqueeze(1),
        ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping at {epoch}")
                break

    print(f"Epoch {epoch:02d} | Train RMSE={train_rmse:.2f} | Val RMSE={val_rmse:.2f}")

print("\n=== Test Set Performance ===")
print(f"Val MAE: {val_mae_hist[-1]:.2f}")
print(f"Val RMSE: {val_rmse_hist[-1]:.2f}")
print(f"Val R^2: {val_r2_hist[-1]:.4f}")

n_epochs = range(1, len(train_mae_hist) + 1)

plt.figure(figsize=(12, 4))

# MAE
plt.subplot(1, 3, 1)
plt.plot(n_epochs, train_mae_hist, label="Train MAE")
plt.plot(n_epochs, val_mae_hist, label="Val MAE", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Mean Absolute Error")
plt.legend()

# RMSE
plt.subplot(1, 3, 2)
plt.plot(n_epochs, train_rmse_hist, label="Train RMSE")
plt.plot(n_epochs, val_rmse_hist, label="Val RMSE", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Root Mean Squared Error")
plt.legend()

# R^2
plt.subplot(1, 3, 3)
plt.plot(n_epochs, train_r2_hist, label="Train R^2")
plt.plot(n_epochs, val_r2_hist, label="Val R^2", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("R^2")
plt.title("Coefficient of Determination")
plt.legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/proto_metrics_plot.png")


# Saving Model
torch.save(model.state_dict(), MODEL_DIR / "lstm_prototype.pth")

results = {
    "train_mae": train_mae_hist,
    "train_rmse": train_rmse_hist,
    "train_r2": train_r2_hist,
    "val_mae": val_mae_hist,
    "val_rmse": val_rmse_hist,
    "val_r2": val_r2_hist,
}

with open(RESULTS_DIR / "proto_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nModel & metrics saved.")
