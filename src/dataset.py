# src/dataset.py

import torch
from torch.utils.data import Dataset



class MarketDataset(Dataset):

    """
    Dataset for pre-created market windows.

    Input:

        windows shape:

        (samples, sequence_length, features)


        Example:

        (70000,60,4)


    Output:

        torch tensor:

        (60,4)

    """


    def __init__(
        self,
        windows
    ):

        self.windows = windows



    def __len__(self):

        return len(
            self.windows
        )



    def __getitem__(
        self,
        index
    ):


        sample = self.windows[
            index
        ]


        sample = torch.tensor(
            sample,
            dtype=torch.float32
        )


        return sample
