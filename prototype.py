# Imports
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load Dataset
DATA_PATH = "data/clean/(Clean) USA Housing Dataset.csv"
df = pd.read_csv(DATA_PATH)

# Sort values so we dont show the model future dates
df.sort_values("date", inplace=True)
df.drop(
    columns=["State", "ZIP Code"], inplace=True
)  # These two just aren't necessary atm

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

split_date = df["date"].quantile(0.7)
train_mask = df["date"] <= split_date
val_mask = df["date"] > split_date

X_train = X_processed[train_mask]
y_train = y[train_mask]

X_val = X_processed[val_mask]
y_val = y[val_mask]


def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PriceLSTM(input_dim=X_train_seq.shape[2]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
patience = 8

best_val_loss = float("inf")
counter = 0

train_mae_hist = []
val_mae_hist = []

train_rmse_hist = []
val_rmse_hist = []

train_r2_hist = []
val_r2_hist = []

for epoch in range(epochs):
    # Training
    model.train()  # Enables dropout
    for i in range(0, len(X_train_seq), batch_size):
        xb = torch.tensor(X_train_seq[i : i + batch_size], dtype=torch.float32).to(
            device
        )
        yb = (
            torch.tensor(y_train_seq[i : i + batch_size], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(X_train_seq, dtype=torch.float32).to(device))
        train_pred_np = train_pred.cpu().numpy().flatten()
        y_train_np = y_train_seq

        # Metrics
        train_mae = mean_absolute_error(y_train_np, train_pred_np)
        train_rmse = np.sqrt(mean_squared_error(y_train_np, train_pred_np))
        train_r2 = r2_score(y_train_np, train_pred_up)

        # Store
        train_mae_hist.append(train_mae)
        train_rmse_hist.append(train_rmse)
        train_r2_hist.append(train_r2)

        val_pre = model(torch.tensor(X_val_seq, dtype=torch.float32).to(device))
        val_loss = criterion(
            val_pred,
            torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1).to(device),
        ).item()

        # Convert predictions back to NumPy for metrics
        val_pred_np = val_pred.cpu().numpy().flatten()
        y_val_np = y_val_seq

        # Metrics
        val_mae = mean_absolute_error(y_val_np, val_pred_np)
        val_rmse = np.sqrt(mean_squared_error(y_val_np, val_pred_np))
        val_r2 = r2_score(y_val_np, val_pred_np)

        # Store
        val_mae_hist.append(val_mae)
        val_rmse_hist.append(val_rmse)
        val_r2_hist.append(val_r2)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    print(f"Epoch {epoch:02d} | Train RMSE ={train_rmse:.2f} | Val RMSE={val_rmse:.2f}")

# Saving Model
torch.save(model.state_dict(), "models/lstm_prototype.pth")

results = {
    "train_mae_last": train_mae_hist[-1],
    "train_rmse_last": train_rmse_hist[-1],
    "train_r2_last": train_r2_hist[-1],
    "val_mae_last": val_mae_hist[-1],
    "val_rmse_last": val_rmse_hist[-1],
    "val_r2_last": val_r2_hist[-1],
}

with open("results/lstm_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nModel & metrics saved.")
