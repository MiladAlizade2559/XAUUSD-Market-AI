# model.py

import torch
import torch.nn as nn

from config import (
    NUM_FEATURES,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    DROPOUT
)



class MarketAutoEncoder(nn.Module):

    def __init__(self):

        super().__init__()


        # -------------------------
        # Feature Projection
        # -------------------------

        self.input_projection = nn.Linear(
            NUM_FEATURES,
            EMBED_DIM
        )



        # -------------------------
        # Positional Information
        # -------------------------

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                60,
                EMBED_DIM
            )
        )



        # -------------------------
        # Transformer Encoder
        # -------------------------

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=EMBED_DIM,

            nhead=NUM_HEADS,

            dim_feedforward=EMBED_DIM * 4,

            dropout=DROPOUT,

            batch_first=True,

            activation="gelu"

        )


        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=NUM_LAYERS

        )



        # -------------------------
        # Decoder
        # -------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                EMBED_DIM,
                EMBED_DIM
            ),

            nn.GELU(),

            nn.Linear(
                EMBED_DIM,
                NUM_FEATURES
            )

        )




    def forward(self, x):

        """
        Input:

            x:
            (batch, seq, features)

            Example:
            (64,60,4)


        Output:

            reconstruction:

            (batch,60,4)

        """


        # feature embedding

        x = self.input_projection(
            x
        )


        # add position

        x = x + self.position_embedding



        # transformer

        x = self.transformer(
            x
        )



        # reconstruction

        x = self.decoder(
            x
        )


        return x
