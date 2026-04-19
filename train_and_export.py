"""
Wine Quality Regression DNN
Dataset: UCI Wine Quality (Red Wine) - different from typical iris/MNIST labs
https://archive.ics.uci.edu/ml/datasets/wine+quality

Architecture: Deep MLP with BatchNorm + Dropout
Framework: PyTorch → exported to ONNX → deployed via HTML/JS (GitHub Pages)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import onnx
import onnxruntime as ort
import json, os

# 1. LOAD DATA

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Quality distribution:\n", df["quality"].value_counts().sort_index())

# Features and target
feature_cols = [c for c in df.columns if c != "quality"]
X = df[feature_cols].values.astype(np.float32)
y = df["quality"].values.astype(np.float32)

# 2. PREPROCESS

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

# Save scaler params for the JS frontend
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": feature_cols
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print("Saved scaler_params.json")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64)

# 3. MODEL DEFINITION

class WineDNN(nn.Module):
    """
    Deep MLP: 11 → 128 → 256 → 256 → 128 → 64 → 1
    BatchNorm + Dropout for regularisation
    """
    def __init__(self, n_features=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = WineDNN(n_features=X_train.shape[1])
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 4. TRAINING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

EPOCHS = 150
history = {"train_loss": [], "val_loss": [], "val_r2": []}

best_val_loss = float("inf")
patience, patience_counter = 15, 0

for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)

    # --- validate ---
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            val_preds.append(model(xb).cpu().numpy())
            val_true.append(yb.numpy())
    val_preds = np.concatenate(val_preds)
    val_true  = np.concatenate(val_true)
    val_loss  = mean_squared_error(val_true, val_preds)
    val_r2    = r2_score(val_true, val_preds)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_r2"].append(val_r2)

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val R²: {val_r2:.4f}")

# Load best
model.load_state_dict(torch.load("best_model.pt"))
print(f"\nBest Val MSE: {best_val_loss:.4f}  |  Best Val R²: {max(history['val_r2']):.4f}")


# 5. EXPORT TO ONNX

model.eval()
model.cpu()
dummy = torch.randn(1, X_train.shape[1])

torch.onnx.export(
    model,
    dummy,
    "wine_quality_dnn.onnx",
    export_params=True,
    opset_version=11,
    input_names=["features"],
    output_names=["quality_score"],
    dynamic_axes={"features": {0: "batch_size"}, "quality_score": {0: "batch_size"}},
)
print("Exported wine_quality_dnn.onnx")

# Verify with OnnxRuntime
sess = ort.InferenceSession("wine_quality_dnn.onnx")
ort_out = sess.run(None, {"features": dummy.numpy()})[0]
pt_out  = model(dummy).detach().numpy()
print(f"ONNX vs PyTorch output diff: {abs(ort_out - pt_out).max():.6f} ✓")

# Save training history for the webpage
with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print("Saved training_history.json")

print("\n✅ All done!  Files to copy to your GitHub Pages repo:")
print("   wine_quality_dnn.onnx")
print("   scaler_params.json")
print("   training_history.json")
print("   index.html  (generated separately)")
