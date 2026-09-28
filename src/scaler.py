# src/scaler.py

import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler



class MarketScaler:


    def __init__(self):

        self.scaler = StandardScaler()



    def fit(self, data):

        """
        Learn normalization parameters.

        Input:
            2D array:
            (samples, features)
        """

        self.scaler.fit(data)

        return self



    def transform(self, data):

        """
        Apply learned normalization.
        """

        return self.scaler.transform(data)



    def fit_transform(self, data):

        return self.scaler.fit_transform(data)



    def save(self, path):

        """
        Save trained scaler.
        """

        with open(path, "wb") as f:

            pickle.dump(
                self.scaler,
                f
            )



    def load(self, path):

        """
        Load existing scaler.
        """

        with open(path, "rb") as f:

            self.scaler = pickle.load(f)

        return self



def scale_features(
    train_data,
    val_data=None,
    scaler_path="market_scaler.pkl"
):


    scaler = MarketScaler()


    # فقط Train یاد می‌گیرد

    train_scaled = scaler.fit_transform(
        train_data
    )


    if val_data is not None:

        val_scaled = scaler.transform(
            val_data
        )

    else:

        val_scaled = None



    scaler.save(
        scaler_path
    )



    return (
        train_scaled,
        val_scaled
    )
