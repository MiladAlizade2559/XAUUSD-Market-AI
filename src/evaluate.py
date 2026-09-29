import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from features import create_features
from window import create_windows
from split import split_time_series
from scaler import MarketScaler
from dataset import MarketDataset
from model import MarketAutoEncoder


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "XAUUSD_l_M1.csv"
)

MODEL_PATH = "/content/drive/MyDrive/XAUUSD_models/best_market_model.pt"

SCALER_PATH = "/content/drive/MyDrive/XAUUSD_models/market_scaler.pkl"

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)

ERRORS_PATH = os.path.join(
    REPORT_DIR,
    "test_reconstruction_errors.csv"
)


# --------------------------------------------------
# Settings
# --------------------------------------------------

BATCH_SIZE = 256


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Check paths
# --------------------------------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_PATH}"
    )

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Load raw data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Raw data:", df.shape)


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

data = create_features(df)

print("Features:", data.shape)


# --------------------------------------------------
# Create windows
# --------------------------------------------------

windows = create_windows(data)

print("Windows:", windows.shape)


# --------------------------------------------------
# Time-series split
# --------------------------------------------------

train_data, val_data, test_data = split_time_series(
    windows
)

print("Train:", train_data.shape)
print("Validation:", val_data.shape)
print("Test:", test_data.shape)


# --------------------------------------------------
# Load scaler
# --------------------------------------------------

scaler = MarketScaler()

scaler.load(SCALER_PATH)

test_scaled = scaler.transform(
    test_data
)

print("Test scaled:", test_scaled.shape)


# --------------------------------------------------
# Test Dataset
# --------------------------------------------------

test_dataset = MarketDataset(
    test_scaled
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = MarketAutoEncoder()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Checkpoint loaded."
    )

    if "epoch" in checkpoint:
        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    if "val_loss" in checkpoint:
        print(
            "Checkpoint val loss:",
            checkpoint["val_loss"]
        )

else:

    model.load_state_dict(
        checkpoint
    )

    print(
        "Model state_dict loaded."
    )


model.to(device)
model.eval()


# --------------------------------------------------
# Reconstruction error
# --------------------------------------------------

errors = []

criterion = torch.nn.SmoothL1Loss(
    reduction="none"
)


with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        reconstruction = model(
            batch
        )

        loss = criterion(
            reconstruction,
            batch
        )

        # Mean error for each window
        batch_errors = loss.mean(
            dim=(1, 2)
        )

        errors.extend(
            batch_errors.detach()
            .cpu()
            .numpy()
        )


errors = np.asarray(
    errors,
    dtype=np.float64
)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

mean_error = errors.mean()

median_error = np.median(
    errors
)

std_error = errors.std()

min_error = errors.min()

max_error = errors.max()

percentile_95 = np.percentile(
    errors,
    95
)

percentile_99 = np.percentile(
    errors,
    99
)


print()
print("===================================")
print("Test Reconstruction Error")
print("===================================")

print(
    "Mean:",
    mean_error
)

print(
    "Median:",
    median_error
)

print(
    "Std:",
    std_error
)

print(
    "Min:",
    min_error
)

print(
    "Max:",
    max_error
)

print(
    "95th percentile:",
    percentile_95
)

print(
    "99th percentile:",
    percentile_99
)


# --------------------------------------------------
# Save errors
# --------------------------------------------------

error_df = pd.DataFrame(
    {
        "reconstruction_error": errors
    }
)

error_df.to_csv(
    ERRORS_PATH,
    index=False
)

print()
print(
    "Errors saved to:",
    ERRORS_PATH
)
