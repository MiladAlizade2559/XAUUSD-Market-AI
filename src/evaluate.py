import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = "/content/XAUUSD-Market-AI"

SRC_PATH = "/content/XAUUSD-Market-AI/src"

DATA_PATH = (
    "/content/XAUUSD-Market-AI/data/"
    "XAUUSD_l_M1.csv"
)

MODEL_PATH = (
    "/content/drive/MyDrive/XAUUSD_models/"
    "best_market_model.pt"
)

SCALER_PATH = (
    "/content/drive/MyDrive/XAUUSD_models/"
    "market_scaler_fixed.pkl"
)


# --------------------------------------------------
# Evaluation output paths
# --------------------------------------------------

EVAL_DIR = (
    "/content/drive/MyDrive/XAUUSD_models/"
    "evaluation"
)

ERRORS_PATH = (
    "/content/drive/MyDrive/XAUUSD_models/"
    "evaluation/"
    "test_reconstruction_errors.csv"
)


# --------------------------------------------------
# Create evaluation directory
# --------------------------------------------------

os.makedirs(
    EVAL_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Python path
# --------------------------------------------------

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from features import create_features
from window import create_windows
from split import split_time_series
from dataset import MarketDataset
from model import MarketAutoEncoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 256


# --------------------------------------------------
# Load raw data
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Raw data: {df.shape}"
)


# --------------------------------------------------
# Create features
# --------------------------------------------------

features = create_features(
    df
)

print(
    f"Features: {features.shape}"
)


# --------------------------------------------------
# Create windows
# --------------------------------------------------

windows = create_windows(
    features
)

print(
    f"Windows: {windows.shape}"
)


# --------------------------------------------------
# Time-series split
# --------------------------------------------------

train_data, val_data, test_data = split_time_series(
    windows
)

print(
    f"Train: {train_data.shape}"
)

print(
    f"Validation: {val_data.shape}"
)

print(
    f"Test: {test_data.shape}"
)


# --------------------------------------------------
# Load trained scaler
# --------------------------------------------------

with open(
    SCALER_PATH,
    "rb"
) as f:

    scaler = pickle.load(
        f
    )


# --------------------------------------------------
# Scale test data
# --------------------------------------------------

test_reshaped = test_data.reshape(
    -1,
    test_data.shape[-1]
)

test_scaled = scaler.transform(
    test_reshaped
).reshape(
    test_data.shape
)

print(
    f"Test scaled: {test_scaled.shape}"
)


# --------------------------------------------------
# Test dataset
# --------------------------------------------------

test_dataset = MarketDataset(
    test_scaled
)


# --------------------------------------------------
# Test dataloader
# --------------------------------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = MarketAutoEncoder()

model = model.to(
    DEVICE
)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        None
    )

    checkpoint_val_loss = checkpoint.get(
        "val_loss",
        None
    )

else:

    model.load_state_dict(
        checkpoint
    )

    checkpoint_epoch = None
    checkpoint_val_loss = None


print(
    "Checkpoint loaded."
)


if checkpoint_epoch is not None:

    print(
        f"Checkpoint epoch: {checkpoint_epoch}"
    )


if checkpoint_val_loss is not None:

    print(
        f"Checkpoint val loss: "
        f"{checkpoint_val_loss}"
    )


# --------------------------------------------------
# Evaluation mode
# --------------------------------------------------

model.eval()


# --------------------------------------------------
# Reconstruction errors
# --------------------------------------------------

reconstruction_errors = []


with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(
            DEVICE
        )

        output = model(
            batch
        )

        # --------------------------------------------------
        # Per-window reconstruction error
        #
        # Shape before mean:
        # [batch, sequence, features]
        #
        # Shape after mean:
        # [batch]
        # --------------------------------------------------

        error = torch.mean(
            torch.abs(
                output - batch
            ),
            dim=(1, 2)
        )

        reconstruction_errors.extend(
            error.cpu().numpy()
        )


reconstruction_errors = np.asarray(
    reconstruction_errors
)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

mean_error = np.mean(
    reconstruction_errors
)

median_error = np.median(
    reconstruction_errors
)

std_error = np.std(
    reconstruction_errors
)

min_error = np.min(
    reconstruction_errors
)

max_error = np.max(
    reconstruction_errors
)

p95_error = np.percentile(
    reconstruction_errors,
    95
)

p99_error = np.percentile(
    reconstruction_errors,
    99
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print(
    "==================================="
)

print(
    "Test Reconstruction Error"
)

print(
    "==================================="
)

print(
    f"Mean: {mean_error}"
)

print(
    f"Median: {median_error}"
)

print(
    f"Std: {std_error}"
)

print(
    f"Min: {min_error}"
)

print(
    f"Max: {max_error}"
)

print(
    f"95th percentile: {p95_error}"
)

print(
    f"99th percentile: {p99_error}"
)


# --------------------------------------------------
# Save reconstruction errors
# --------------------------------------------------

errors_df = pd.DataFrame(
    {
        "window_index": np.arange(
            len(reconstruction_errors)
        ),
        "reconstruction_error": reconstruction_errors
    }
)


errors_df.to_csv(
    ERRORS_PATH,
    index=False
)


print()
print(
    f"Errors saved to: {ERRORS_PATH}"
)
