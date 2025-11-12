"""
Training script for the house‑price LSTM models.
"""

# --------------------------------------------------------------------------- #
# Imports and basic configuration
# --------------------------------------------------------------------------- #

import argparse  # parse command‑line arguments
import copy  # deep‑copy of objects
import json  # read/write JSON files (for results)
from pathlib import Path  # object‑oriented filesystem paths

import numpy as np  # numerical operations
import pandas as pd  # data manipulation (CSV reading, etc.)
import torch  # deep‑learning framework
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Reproducibility: set a fixed random seed
# --------------------------------------------------------------------------- #

SEED = 42  # any number works – we just need it to be the same every run
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Tell PyTorch to use deterministic algorithms (optional but makes results stable)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------- #
# Detect GPU if available
# --------------------------------------------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------- #
# Utility: turn a long time series into overlapping windows (sequences)
# --------------------------------------------------------------------------- #


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    # Total number of samples in the original data
    n_samples = X.shape[0]

    # Containers for the new sequences and their targets
    Xs, ys = [], []

    # Build every possible window of length `seq_len`
    for i in range(n_samples - seq_len):
        # Slice out the window from X
        Xs.append(X[i : i + seq_len])
        # The target is the value that comes immediately after the window
        ys.append(y[i + seq_len - 1])

    # Convert lists to numpy arrays (the LSTM expects float32 tensors)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# --------------------------------------------------------------------------- #
# The LSTM model that will learn from the sequences
# --------------------------------------------------------------------------- #


class PriceLSTM(nn.Module):
    """
    A bidirectional LSTM network with a single output neuron.
    It reads sequences of feature vectors and predicts the next price.
    """

    def __init__(
        self,
        input_dim: int,  # number of features in one time step
        hidden_dim: int = 256,  # size of the LSTM memory cell
        num_layers: int = 2,  # how many stacked LSTM layers
        dropout: float = 0.3,
    ):  # probability of dropping a unit during training
        super().__init__()

        # The core LSTM – bidirectional means it looks at the past *and* future
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Dropout layer to further reduce over‑fitting
        self.dropout = nn.Dropout(dropout)

        # Final linear layer that collapses the hidden state into a single number
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 because bidirectional

    def forward(self, x):
        """
        Forward pass: compute the hidden states and produce a price prediction.
        """
        # Run all time steps through the LSTM
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden*2)

        # Take the last time step – this is what the LSTM predicts for the next day
        out = out[:, -1, :]

        # Apply dropout before the final linear layer
        out = self.dropout(out)

        # Final prediction: a single value per sample
        return self.fc(out)


# --------------------------------------------------------------------------- #
# Training routine for a single duration (1‑week, 4‑weeks or 12‑weeks)
# --------------------------------------------------------------------------- #


def train_model_for_duration(
    df: pd.DataFrame,
    duration: int,
    target_col: str,
    seq_len: int = 12,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 8,
):

    # -----------------------------------------------------------------------
    # Prepare the data
    # -----------------------------------------------------------------------

    # Make sure dates are sorted so we never look into the future
    df = df.sort_values("PERIOD_BEGIN").reset_index(drop=True)
    if "REGION_TYPE" in df.columns:
        df = pd.get_dummies(df, columns=["REGION_TYPE"], drop_first=False)

    # Log‑transform the target – makes the values more normally distributed
    df[target_col] = np.log1p(df[target_col])

    # Remove any rows that still contain missing values
    df = df.dropna().reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Separate features (X) and target (y)
    # -----------------------------------------------------------------------

    feature_cols = [
        c for c in df.columns if c not in {"PERIOD_BEGIN", "DURATION", target_col}
    ]
    X = df[feature_cols].values.astype(np.float32)  # numeric matrix
    y = df[target_col].values.astype(np.float32)  # target vector

    # -----------------------------------------------------------------------
    # Split the data into train / val / test (time‑series split)
    # -----------------------------------------------------------------------

    n = X.shape[0]
    train_end = int(0.7 * n)  # first 70 % for training
    val_end = int(0.85 * n)  # next 15 % for validation

    X_train_raw, y_train_raw = X[:train_end], y[:train_end]
    X_val_raw, y_val_raw = X[train_end:val_end], y[train_end:val_end]
    X_test_raw, y_test_raw = X[val_end:], y[val_end:]

    # -----------------------------------------------------------------------
    # Scale the numeric features (only on training data)
    # -----------------------------------------------------------------------

    scaler = StandardScaler().fit(X_train_raw)  # compute mean & std on train
    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # -----------------------------------------------------------------------
    # Create overlapping sequences for the LSTM
    # -----------------------------------------------------------------------

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_raw, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw, seq_len)

    # -----------------------------------------------------------------------
    # Wrap data in PyTorch DataLoaders (batches)
    # -----------------------------------------------------------------------

    train_ds = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq))
    val_ds = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq))
    test_ds = TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test_seq))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -----------------------------------------------------------------------
    # Build the model
    # -----------------------------------------------------------------------

    model = PriceLSTM(input_dim=X_train_seq.shape[2]).to(device)

    # Loss function (MSE) and optimizer with weight‑decay regularisation
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning‑rate scheduler – reduce LR when validation stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # -----------------------------------------------------------------------
    # Training loop with early stopping
    # -----------------------------------------------------------------------

    best_val_loss = float("inf")
    counter = 0  # counts epochs without improvement

    for epoch in range(1, epochs + 1):
        model.train()  # enable dropout
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.unsqueeze(1).to(device)  # make shape (batch, 1)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation – no gradient needed
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.unsqueeze(1).to(device)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # Early‑stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Print a short progress line
        print(
            f"Epoch {epoch:02d} | Train loss={loss.item():.4f} | Val loss={val_loss:.4f}"
        )

    # -----------------------------------------------------------------------
    # Load the best model (the one with lowest validation loss)
    # -----------------------------------------------------------------------

    if "best_model_state_dict" in locals():
        model.load_state_dict(best_model_state_dict)

    # -----------------------------------------------------------------------
    # Evaluate on train / val / test – produce metrics
    # -----------------------------------------------------------------------

    def _predict(loader):
        """Run the model on a DataLoader and return raw predictions."""
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy().flatten())
        return np.concatenate(preds)

    # Predictions in log‑scale
    train_pred_log = _predict(train_loader)
    val_pred_log = _predict(val_loader)
    test_pred_log = _predict(test_loader)

    # True values (log‑scale) – the target vectors we used
    train_true_log = y_train_seq
    val_true_log = y_val_seq
    test_true_log = y_test_seq

    # Convert back to original price scale (inverse of log1p)
    train_pred = np.expm1(train_pred_log)
    val_pred = np.expm1(val_pred_log)
    test_pred = np.expm1(test_pred_log)

    train_true = np.expm1(train_true_log)
    val_true = np.expm1(val_true_log)
    test_true = np.expm1(test_true_log)

    # Compute metrics
    metrics = {
        "duration": duration,
        "train_mae": mean_absolute_error(train_true, train_pred),
        "train_rmse": np.sqrt(mean_squared_error(train_true, train_pred)),
        "train_r2": r2_score(train_true, train_pred),
        "val_mae": mean_absolute_error(val_true, val_pred),
        "val_rmse": np.sqrt(mean_squared_error(val_true, val_pred)),
        "val_r2": r2_score(val_true, val_pred),
        "test_mae": mean_absolute_error(test_true, test_pred),
        "test_rmse": np.sqrt(mean_squared_error(test_true, test_pred)),
        "test_r2": r2_score(test_true, test_pred),
    }

    return metrics, model.state_dict(), scaler


# --------------------------------------------------------------------------- #
# 7. Main script – orchestrate training for all three durations
# --------------------------------------------------------------------------- #


def main():
    # --------------------------------------------------------------
    # Parse command‑line arguments
    # --------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Train LSTM models for 1‑week, 4‑weeks and 12‑week data."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("clean"),
        help="Folder containing the cleaned CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models"),
        help="Folder where model checkpoints and scalers will be saved.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="MEDIAN_SALE_PRICE",
        help="Name of the target column to predict.",
    )
    parser.add_argument(
        "--seq_len", type=int, default=12, help="Length of the input sequence (weeks)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=8, help="Early‑stopping patience (epochs)."
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Prepare output directories
    # --------------------------------------------------------------

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scalers").mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Find all CSV files in the input folder
    # --------------------------------------------------------------

    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    # --------------------------------------------------------------
    # Train a model for each duration
    # --------------------------------------------------------------

    all_results = []

    for fp in csv_files:
        print(f"\n=== Processing {fp.name} ===")

        # Load the cleaned CSV
        df = pd.read_csv(fp)

        # Every row in a file has the same DURATION value – we can read it once
        if df["DURATION"].nunique() != 1:
            raise ValueError(f"File {fp.name} contains multiple durations.")
        duration = int(df["DURATION"].iloc[0])

        # Train the model for this duration
        metrics, state_dict, scaler = train_model_for_duration(
            df=df,
            duration=duration,
            target_col=args.target,
            seq_len=args.seq_len,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
        )

        # ----------------------------------------------------------
        # Save the trained model and scaler
        # ----------------------------------------------------------

        model_path = args.output_dir / f"lstm_dur{duration}.pth"
        torch.save(state_dict, model_path)

        scaler_path = args.output_dir / "scalers" / f"scaler_dur{duration}.pkl"
        joblib.dump(scaler, scaler_path)

        # ----------------------------------------------------------
        # Record the metrics for later analysis
        # ----------------------------------------------------------

        all_results.append(metrics)
        print(f"Finished duration {duration} – test MAE: {metrics['test_mae']:.2f}")

    # --------------------------------------------------------------
    # Save all metrics to a single JSON file
    # --------------------------------------------------------------

    results_file = args.output_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll models trained. Results written to {results_file}")


# --------------------------------------------------------------------------- #
# 8. Entry point – run the script
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
