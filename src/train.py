# train.py


import os

import torch
import torch.nn as nn
from torch.optim import Adam

import pandas as pd


from features import create_features

from window import create_windows

from split import split_time_series

from scaler import MarketScaler

from dataset import MarketDataset

from dataloader import create_dataloader

from model import MarketAutoEncoder


# =====================================
# Paths
# =====================================

PROJECT_ROOT = "/content/XAUUSD-Market-AI"

DATA_PATH = (
    "/content/XAUUSD-Market-AI/"
    "data/XAUUSD_l_M1.csv"
)

MODEL_DIR = (
    "/content/drive/MyDrive/"
    "XAUUSD_models"
)

MODEL_PATH = (
    "/content/drive/MyDrive/"
    "XAUUSD_models/best_market_model.pt"
)

SCALER_PATH = (
    "/content/drive/MyDrive/"
    "XAUUSD_models/market_scaler_fixed.pkl"
)


# =====================================
# Create output directory
# =====================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =====================================
# Device
# =====================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)


# =====================================
# Load Data
# =====================================

df = pd.read_csv(
    DATA_PATH
)


print(
    "Raw data:",
    df.shape
)


# =====================================
# Feature Engineering
# =====================================

data = create_features(
    df
)


print(
    "Features:",
    data.shape
)


# =====================================
# Window Creation
# =====================================

windows = create_windows(
    data
)


print(
    "Windows:",
    windows.shape
)


# =====================================
# Split
# =====================================

train_data, val_data, test_data = split_time_series(
    windows
)


print(
    "Train:",
    train_data.shape
)

print(
    "Validation:",
    val_data.shape
)

print(
    "Test:",
    test_data.shape
)


# =====================================
# Scaling
# =====================================

scaler = MarketScaler()


# Fit ONLY on training data
scaler.fit(
    train_data
)


train_scaled = scaler.transform(
    train_data
)


val_scaled = scaler.transform(
    val_data
)


test_scaled = scaler.transform(
    test_data
)


print(
    "Train scaled:",
    train_scaled.shape
)

print(
    "Validation scaled:",
    val_scaled.shape
)

print(
    "Test scaled:",
    test_scaled.shape
)


# =====================================
# Save Scaler
# =====================================

scaler.save(
    SCALER_PATH
)


print(
    "Scaler saved to:",
    SCALER_PATH
)


# =====================================
# Dataset
# =====================================

train_dataset = MarketDataset(
    train_scaled
)


val_dataset = MarketDataset(
    val_scaled
)


# =====================================
# DataLoader
# =====================================

train_loader = create_dataloader(

    train_dataset,

    batch_size=64,

    shuffle=True

)


val_loader = create_dataloader(

    val_dataset,

    batch_size=64,

    shuffle=False

)


# =====================================
# Model
# =====================================

model = MarketAutoEncoder()


model.to(
    device
)


# =====================================
# Loss
# =====================================

criterion = nn.SmoothL1Loss()


# =====================================
# Optimizer
# =====================================

optimizer = Adam(

    model.parameters(),

    lr=0.001

)


# =====================================
# Training
# =====================================

EPOCHS = 30


best_val_loss = float(
    "inf"
)


for epoch in range(EPOCHS):


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )


    # =================================
    # Train
    # =================================

    model.train()


    train_loss = 0.0


    for batch in train_loader:


        batch = batch.to(
            device
        )


        optimizer.zero_grad()


        reconstruction = model(
            batch
        )


        loss = criterion(

            reconstruction,

            batch

        )


        loss.backward()


        optimizer.step()


        train_loss += loss.item()


    train_loss /= len(
        train_loader
    )


    # =================================
    # Validation
    # =================================

    model.eval()


    val_loss = 0.0


    with torch.no_grad():


        for batch in val_loader:


            batch = batch.to(
                device
            )


            reconstruction = model(
                batch
            )


            loss = criterion(

                reconstruction,

                batch

            )


            val_loss += loss.item()


    val_loss /= len(
        val_loader
    )


    # =================================
    # Print Loss
    # =================================

    print(
        "Train Loss:",
        train_loss
    )


    print(
        "Val Loss:",
        val_loss
    )


    # =================================
    # Save Best Model
    # =================================

    if val_loss < best_val_loss:


        best_val_loss = val_loss


        torch.save(

            {

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss":
                    val_loss

            },

            MODEL_PATH

        )


        print(
            "✅ Best model saved"
        )

        print(
            "Model path:",
            MODEL_PATH
        )


# =====================================
# Training Complete
# =====================================

print(
    "\n==================================="
)

print(
    "Training Complete"
)

print(
    "==================================="
)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Best model:",
    MODEL_PATH
)

print(
    "Scaler:",
    SCALER_PATH
)
