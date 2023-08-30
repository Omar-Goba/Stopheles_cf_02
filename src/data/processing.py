### importations ###
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


def get_date(path: str) -> pd.DataFrame:
    """
        Get the date from the csv file
        args:
            path: str, default = None
                path to the csv file
        return:
            df: pd.DataFrame
                dataframe with the date
    """
    df = pd.read_csv(path, names=["date", "value", "label", "unused"], header=None)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
        Clean the dataframe
        do the following:
            - drop unused column (last column)
            - rename columns (timestamp, target_value, catigory)
            - set the dtype of the dataframe
            - remove outliers (IQR) on target_value
            - drop duplicate dates by the following aggregation:
                * target_value = sum
                * category = list
            - reindexing to be daily and fill missing days with the following:
                * target_value = 0
                * category = np.nan
        args:
            df: pd.DataFrame
                dataframe with the date
        return:
            df: pd.DataFrame
                cleaned dataframe
    """
    ...

def set_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """
        Set the dtype of the dataframe
        such that:
            - timestamp: datetime64
            - target_value: float64
            - category: list
    """
    ...

def main() -> int:
    """
    """
    ### get the data ###
    root_path = "./"
    df = get_date(f"{root_path}dbs/intermittent/db.csv")
    print(df)

    return 0


if (__name__ == "__main__"):    main()
