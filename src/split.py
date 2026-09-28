# src/split.py

import numpy as np



def split_time_series(
    windows,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):

    """
    Split time series windows without shuffle.

    Input:

        windows:

        (samples, sequence_length, features)


        Example:

        (99940,60,4)


    Output:

        train
        validation
        test

    """



    assert (
        train_ratio
        +
        val_ratio
        +
        test_ratio
        ==
        1.0
    )


    total = len(windows)



    train_end = int(
        total * train_ratio
    )


    val_end = int(
        total *
        (train_ratio + val_ratio)
    )



    train = windows[
        :train_end
    ]


    val = windows[
        train_end:val_end
    ]


    test = windows[
        val_end:
    ]



    return (
        train,
        val,
        test
    )
