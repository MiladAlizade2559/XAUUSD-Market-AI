
# src/config.py

import os


# =========================
# Project Paths
# =========================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")


# =========================
# Data Settings
# =========================

# تعداد کندل‌هایی که مدل در هر ورودی می‌بیند
SEQ_LENGTH = 60


# تایم فریم
TIMEFRAME = "M1"


# =========================
# Feature Settings
# =========================

# تعداد Featureهای ورودی
NUM_FEATURES = 10


FEATURE_NAMES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "range",
    "body",
    "upper_wick",
    "lower_wick",
    "volatility"
]


# =========================
# Model Settings
# =========================

# اندازه Representation بازار
EMBEDDING_DIM = 256


# CNN
CNN_CHANNELS = 64


# Transformer
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 8
TRANSFORMER_DROPOUT = 0.1


# =========================
# Training Settings
# =========================

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5


# =========================
# Masking Settings
# =========================

MASK_RATIO = 0.15


# =========================
# Files
# =========================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_market_model.pt"
)


REPORT_PATH = os.path.join(
    REPORT_DIR,
    "training_report.json"
)
