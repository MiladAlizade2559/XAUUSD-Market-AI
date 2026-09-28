# src/window.py

import numpy as np



def create_windows(
    data,
    sequence_length=60
):

    """
    Convert time series features into sequences.

    Input:

        data shape:

        (candles, features)

        Example:

        (100000,4)


    Output:

        windows shape:

        (samples, sequence_length, features)

        Example:

        (99940,60,4)

    """



    windows = []


    total_samples = (
        len(data)
        -
        sequence_length
    )


    for i in range(total_samples):


        window = data[
            i :
            i + sequence_length
        ]


        windows.append(
            window
        )


    return np.array(
        windows,
        dtype=np.float32
    )
