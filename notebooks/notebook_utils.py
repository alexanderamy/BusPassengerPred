import pandas as pd
from typing import List

def resolve_paths_from_parent_directory():
    # Resolve paths from root project directory
    import os
    import sys

    module_path = os.path.abspath(os.path.join(".."))
    if module_path not in sys.path:
        sys.path.append(module_path)


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimize_dataframe(df: pd.DataFrame):
    """ Optimizes dataframe memory usage """
    return optimize_floats(optimize_ints(df))