# src/scaler.py

import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler



class MarketScaler:


    def __init__(self):

        self.scaler = StandardScaler()



    def fit(self, windows):

        """
        Fit scaler only on training windows.

        Input:

            (samples, sequence, features)

        Example:

            (70000,60,4)

        """


        samples, seq, features = windows.shape


        reshaped = windows.reshape(
            samples * seq,
            features
        )


        self.scaler.fit(
            reshaped
        )


        return self



    def transform(self, windows):

        """
        Normalize windows.

        Keeps original shape.
        """


        samples, seq, features = windows.shape


        reshaped = windows.reshape(
            samples * seq,
            features
        )


        scaled = self.scaler.transform(
            reshaped
        )


        return scaled.reshape(
            samples,
            seq,
            features
        )



    def fit_transform(self, windows):

        self.fit(
            windows
        )

        return self.transform(
            windows
        )



    def save(self, path):

        with open(path, "wb") as f:

            pickle.dump(
                self.scaler,
                f
            )



    def load(self, path):

        with open(path, "rb") as f:

            self.scaler = pickle.load(
                f
            )

        return self
