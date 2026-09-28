# src/features.py

import pandas as pd
import numpy as np



AVAILABLE_FEATURES = [

    "range_points",
    "open_position_points",
    "close_position_points",
    "gap_points"

]



def create_features(
    df,
    selected_features=None,
    point=0.01
):

    """
    Create price action features for XAUUSD.

    Input:
        DataFrame:
            open
            high
            low
            close

    Output:
        Selected market features
    """


    data = df.copy()



    # اگر چیزی انتخاب نشده بود
    # همه Feature ها ساخته می شوند

    if selected_features is None:

        selected_features = AVAILABLE_FEATURES



    result = pd.DataFrame(
        index=data.index
    )



    # ==========================
    # Candle Range
    # ==========================

    if "range_points" in selected_features:

        result["range_points"] = (

            (data["high"] - data["low"])

            / point

        )



    # ==========================
    # Open Position
    # ==========================

    if "open_position_points" in selected_features:

        result["open_position_points"] = (

            (data["open"] - data["low"])

            / point

        )



    # ==========================
    # Close Position
    # ==========================

    if "close_position_points" in selected_features:

        result["close_position_points"] = (

            (data["close"] - data["low"])

            / point

        )



    # ==========================
    # Gap
    # ==========================

    if "gap_points" in selected_features:

        result["gap_points"] = (

            (data["open"] - data["close"].shift(1))

            / point

        )



    # حذف NaN ایجاد شده توسط shift

    result = result.dropna().reset_index(drop=True)



    return result
