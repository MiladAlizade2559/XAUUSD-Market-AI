# config.py


# =========================
# Data Configuration
# =========================

# XAUUSD point size
POINT = 0.01



# =========================
# Feature Configuration
# =========================

FEATURE_NAMES = [

    "range_points",

    "open_position_points",

    "close_position_points",

    "gap_points"

]


NUM_FEATURES = len(
    FEATURE_NAMES
)



# =========================
# Window Configuration
# =========================

# Number of candles in each sequence
SEQ_LENGTH = 60



# =========================
# Train Configuration
# =========================

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 0.001



# =========================
# Split Configuration
# =========================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15



# =========================
# Model Configuration
# =========================

# Transformer / Encoder dimensions
EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 3

DROPOUT = 0.1



# =========================
# Saving Configuration
# =========================

MODEL_PATH = "best_market_model.pt"

SCALER_PATH = "market_scaler.pkl"
