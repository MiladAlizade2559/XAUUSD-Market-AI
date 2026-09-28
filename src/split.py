# src/split.py

import numpy as np



def split_time_series(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):

    """
    Split time series data without shuffle.

    Example:

    data:
        [old ---------------- new]

    output:

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


    total = len(data)


    train_end = int(
        total * train_ratio
    )


    val_end = int(
        total *
        (train_ratio + val_ratio)
    )



    train = data[
        :train_end
    ]


    val = data[
        train_end:val_end
    ]


    test = data[
        val_end:
    ]


    return (
        train,
        val,
        test
    )
