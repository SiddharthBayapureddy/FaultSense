import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


SENSOR_COLS = [f"s{i}" for i in range(1, 18)]

def load_data(path: str = "./data/train_FD001.txt") -> pd.DataFrame:
    columns = [
        "unit", "cycle", "os1", "os2", "os3",
        *[f"s{i}" for i in range(1, 18)]
    ]
    df = pd.read_csv(
        path,
        sep=r"\s+", 
        header=None,
        names=columns,
        engine="python",
        index_col=False    # ← key fix
    )
    df = df.iloc[:, :22]   # keep only first 22 columns, drop trailing garbage
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def add_rul_labels(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=rul_cap)
    df = df.drop(columns=["max_cycle"])
    return df


def create_windows(df: pd.DataFrame, window_size: int = 30) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        sensors = unit_df[SENSOR_COLS].values
        rul = unit_df["rul"].values

        for i in range(len(sensors) - window_size + 1):
            X.append(sensors[i:i + window_size])
            y.append(rul[i + window_size - 1])

    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 17, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


def train(path: str = "./data/train_FD001.txt", window_size: int = 30, epochs: int = 50, batch_size: int = 64, lr: float = 0.001):

    print("Loading data...")
    df = load_data(path)
    df = add_rul_labels(df)

    print("Creating windows...")
    X, y = create_windows(df, window_size)
    print(f"  → X shape: {X.shape}, y shape: {y.shape}")
    print(f"  → y min: {y.min()}, y max: {y.max()}, y mean: {y.mean():.2f}")
    print(f"  → X min: {X.min():.4f}, X max: {X.max():.4f}")

    # Train/val split BEFORE normalizing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit scaler ONLY on training data, apply to both
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 17)).reshape(-1, window_size, 17)
    X_val   = scaler.transform(X_val.reshape(-1, 17)).reshape(-1, window_size, 17)

    # Normalize y to 0-1 range
    y_train = y_train / 125.0
    y_val   = y_val / 125.0
    print(f"  → y_train sample (normalized): {y_train[:5]}")

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val,   dtype=torch.float32)

    # DataLoaders
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds   = torch.utils.data.TensorDataset(X_val,   y_val)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size)

    # Model, loss, optimizer
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb)
                val_loss += criterion(preds, yb).item()

        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_dl):.4f} | Val Loss: {val_loss/len(val_dl):.4f}")

    # Save model + scaler
    torch.save(model.state_dict(), "./models/rul_model.pt")
    joblib.dump(scaler, "./models/scaler.pkl")
    print("Model saved to models/rul_model.pt")
    print("Scaler saved to models/scaler.pkl")



if __name__ == "__main__":
    train()