### importations ###
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
    

### ~~~ CLEANING ~~~ ###
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

def OHE(df: pd.DataFrame) -> pd.DataFrame:
    """
        Do a one hot encoding on the dataframe for the 
        catigory column. hint: use pd.get_dummies and "|".join()
        df.category[i] = [cat1, cat2, cat3, ...] --> df.cat1[i] = 1, df.cat2[i] = 1, df.cat3[i] = 1, ...
        2023/09/06,100,["coffee", "tea", "milk"] --> 2023/09/06,100,1,1,1,0,0
        2023/09/07,85,["coffee", "milk"] --> 2023/09/06,100,1,0,1,0,0
        2023/09/07,85,["cigar", "gum"] --> 2023/09/06,100,0,0,0,1,1
    """
    ...

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

    ### drop unused ###
    df.drop("unused", axis=1, inplace=True)
    
      ### rename ###
    column_mapping = {'date': 'timestamp', 'value': 'target_value', 'label': 'category'}
    df.rename(columns=column_mapping, inplace=True)
    
    ### remove outliers ###
    Q1 = df['target_value'].quantile(0.25)
    Q3 = df['target_value'].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['target_value']>=lower_bound) & df['target_value']<=upper_bound ]
    
    ### drop duplicate dates & aggregate ###
    df = df.groupby('timestamp').agg({
    'target_value': 'sum',
    'category': list
    }).reset_index()
    
    ### reindex to be daily ###
    try:
      data_range = pd.date_range(start= df['timestamp'].min(), end = df['timestamp'].max())
      df = df.set_index('timestamp').reindex(data_range,fill_value=0)
      df.loc[df['target_value']==0 ,'category'] = pd.NA 
    except ValueError as e:
        # Handle date parsing errors by printing the problematic values
        print(f"Error parsing dates: {e}")
        print("Problematic values:")
    
    return df
    
    

    
    

def set_dtype(df: pd.DataFrame) -> pd.DataFrame:
    # Specify the custom date format
    date_format = '%y/%m/%d'
    df = df.dropna(subset=['date'])

    try:
        df['date'] = pd.to_datetime(df['date'], format=date_format)
    except ValueError as e:
        pass
    
        #print(f"Error parsing dates: {e}")
       # print("Problematic values:")
       # print(df.loc[pd.to_datetime(df['date'], format=date_format, errors='coerce').isna()]['date']) 

    df['value'] = df['value'].astype(float)
    df['label'] = df['label'].apply(lambda x: [x])  
    
    return df


### ~~~ CLEANING ~~~ ###
### ~~~ FEATURE EXTRACTION ~~~ ###
def get_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Get the date features from the dataframe
            - day of week
            - day of month
            - day of year
            - week of year
            - month of year
            - quarter of year
            - is weekend
            - remove the date column
        args:
            df: pd.DataFrame, default = None
                dataframe with the date column
        return:
            df: pd.DataFrame
                dataframe with the date features but without the date column
    """
    ...

def get_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:    
    """
        Get the time features from the dataframe
            - lags, lag = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
            - rolling mean, window = [7, 14, 21, 28]
            - exponential weighted mean, alpha = [0.3, 0.5, 0.7], window = [7, 14, 21, 28]
            - fourier transform, n = [3, 6, 9, 12, 15, 18, 21, 24]
        args:
            df: pd.DataFrame, default = None
        return:
            df: pd.DataFrame
    """
    ...
### ~~~ FEATURE EXTRACTION ~~~ ###



def main() -> int:
    """
    """
    ### get the data ###
    df = get_date("db.csv") 
    df = set_dtype(df)
    df = clean(df)

    print(df)

    return 0


if (__name__ == "__main__"):    main()
