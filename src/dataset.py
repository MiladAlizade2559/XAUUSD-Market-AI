# src/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MarketDataset(Dataset):
    """
    Dataset for XAUUSD market data.

    Converts continuous market features into
    fixed length sequences for Transformer / AutoEncoder.

    Input:
        Sequence of candles

        Shape:
            (seq_length, num_features)

    Example:
        (60, 10)

    Output:
        Same sequence for reconstruction

        Shape:
            (60, 10)
    """

    def __init__(
        self,
        data,
        seq_length=60
    ):

        self.seq_length = seq_length


        # Accept pandas DataFrame or numpy array

        if isinstance(data, pd.DataFrame):
            data = data.values


        # Ensure correct datatype for PyTorch

        self.data = data.astype(np.float32)



    def __len__(self):

        return len(self.data) - self.seq_length



    def __getitem__(self, index):

        # Create sliding window

        sequence = self.data[
            index :
            index + self.seq_length
        ]


        # Convert numpy array to torch tensor

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )


        return sequence
