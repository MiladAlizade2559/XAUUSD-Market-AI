# train.py


import torch
import torch.nn as nn
from torch.optim import Adam

import joblib
import pandas as pd


from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    SCALER_PATH
)


from features import create_features

from window import create_windows

from split import split_data

from scaler import fit_scaler, transform_data

from dataset import MarketDataset

from dataloader import create_dataloader

from model import MarketAutoEncoder



# =====================================
# Device
# =====================================


device = torch.device(
    "cuda" if torch.cuda.is_available()
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
    "data/XAUUSD.csv"
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


train_data, val_data, test_data = split_data(
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


scaler = fit_scaler(
    train_data
)


train_scaled = transform_data(
    scaler,
    train_data
)


val_scaled = transform_data(
    scaler,
    val_data
)


test_scaled = transform_data(
    scaler,
    test_data
)



# save scaler

joblib.dump(
    scaler,
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

    batch_size=BATCH_SIZE,

    shuffle=True

)


val_loader = create_dataloader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)



# =====================================
# Model
# =====================================


model = MarketAutoEncoder()


model.to(device)



criterion = nn.SmoothL1Loss()


optimizer = Adam(

    model.parameters(),

    lr=LEARNING_RATE

)



# =====================================
# Training
# =====================================


best_val_loss = float(
    "inf"
)


for epoch in range(EPOCHS):


    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
    )


    # -----------------
    # Train
    # -----------------

    model.train()


    train_loss = 0



    for batch in train_loader:


        batch = batch.to(device)


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



    train_loss /= len(train_loader)



    # -----------------
    # Validation
    # -----------------

    model.eval()


    val_loss = 0


    with torch.no_grad():


        for batch in val_loader:


            batch = batch.to(device)



            reconstruction = model(
                batch
            )


            loss = criterion(

                reconstruction,

                batch

            )


            val_loss += loss.item()



    val_loss /= len(val_loader)



    print(
        "Train Loss:",
        train_loss
    )


    print(
        "Val Loss:",
        val_loss
    )



    # -----------------
    # Save Best Model
    # -----------------


    if val_loss < best_val_loss:


        best_val_loss = val_loss


        torch.save(

            {

            "epoch": epoch + 1,


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
