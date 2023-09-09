### importations ###
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from datetime import datetime
    
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
    df['category'] = df['category'].fillna('')
    try :
        df['category'] = df['category'].apply(lambda x: '|'.join(map(str, x)) if isinstance(x, list) else str(x))

    except ValueError:
        pass
    
    unique_categories = set(category for categories in df['category'] if isinstance(categories, str)
                            for category in categories.split('|'))
    for category in unique_categories:
        df[category] = 0
    
    # Populate the columns with 1 if the category is present in the row's string
    for category in unique_categories:
        df[category] = df['category'].apply(lambda x: 1 if isinstance(x, str) and category in x.split('|') else 0)
    
    df.drop(columns=['category'],inplace=True)
    #df = pd.get_dummies(df, columns=['category'])

    return df
    
def reindex_df(df: pd.DataFrame) -> pd.DataFrame:
    
    """reindex the dataframe to have all dates starting from the first to the last and fill the empty dates with 
    value=0 and category = na"""
    try:
      data_range = pd.date_range(start= df['timestamp'].min(), end = df['timestamp'].max())
      df = df.set_index('timestamp').reindex(data_range,fill_value=0)
      df.loc[df['target_value']==0 ,'category'] = pd.NA 
      df['timestamp'] = df.index
      df.index.name = None  # Remove the index name
      df = df[['timestamp'] + [col for col in df.columns if col != 'timestamp']] 
    except ValueError as e:
        pass
    
    return df

def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    
    "aggregates the duplicate dates and adds different categories to a list"
    df = df.groupby('timestamp').agg({
    'target_value': 'sum',
    'category': list
    }).reset_index()
    df['category']= df['category'].apply(lambda x: list(set(x)))
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """removes outliers"""
    Q1 = df['target_value'].quantile(0.25)
    Q3 = df['target_value'].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['target_value']>=lower_bound) & df['target_value']<=upper_bound ]
    return df
    
def clean(df: pd.DataFrame) -> pd.DataFrame:
    
    df = set_dtype(df)
    ### drop unused ###
    df.drop("unused", axis=1, inplace=True)
    
      ### rename ###
    column_mapping = {'date': 'timestamp', 'value': 'target_value', 'label': 'category'}
    df.rename(columns=column_mapping, inplace=True)
    
    ### remove outliers ###
    df = remove_outliers(df)
    
    ### drop duplicate dates & aggregate ###
    df = aggregate_duplicates(df)
    
    ### reindex to be daily ###
    df = reindex_df(df)
    
    return df
    
def set_dtype(df: pd.DataFrame) -> pd.DataFrame:
    
    # Specify the custom date format
    date_format = '%y/%m/%d'
    df = df.dropna(subset=['date'])

    try:
        df['date'] = pd.to_datetime(df['date'], format=date_format)
    except ValueError as e:
        pass
    
    # set the type of the value column to be float
    df['value'] = df['value'].astype(float)
    # set the label column to be 1d list using explode to flatten the list
    df['label'] = df['label'].explode().reset_index(drop=True)

    return df

def is_weekend(day):
    ### check if the provided day is friday or saturday###
    return day.weekday()==4 or day.weekday()==5

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
    #Create a new column which contains the corresponding weekday's name for each timestamp
    #df['day_of_week_name'] = df['timestamp'].dt.day_name
    df['day_of_week_index'] = df['timestamp'].dt.weekday
    
     #Create a new column which contains the corresponding day of the month for each timestamp
    df['day_of_month'] = df['timestamp'].dt.day
    
    #Create a new column which contains the corresponding day of the year for each timestamp
    df['day_of_year'] = df['timestamp'].dt.dayofyear
        
    #Create a new column which contains the corresponding week of the year for each timestamp
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
    #Create a new column which contains the corresponding week of the year for each timestamp
    df['month_of_year'] = df['timestamp'].dt.month
        
    #Create a new column which contains the corresponding quarter of the year for each timestamp
    df['quarter_of_year'] = df['timestamp'].dt.quarter
        
    #Create a new column which tells if this day is a weekend or not , 1 = yes , 0 = no using a helper method
    df['is_weekend'] = df['timestamp'].apply(is_weekend)
        
    #remove the date columns
    df.drop('timestamp',axis=1,inplace=True)
        
    return df
    
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
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    for lag in lags:
        df[f'target_value_lag{lag}'] = df['target_value'].shift(lag)
    
    rolling_windows = [7, 14, 21, 28]
    for window in rolling_windows:
        df[f'target_value_rolling_mean{window}'] = df['target_value'].rolling(window=window).mean()
    
    alpha_values = [0.3, 0.5, 0.7]
    window_sizes = [7, 14, 21, 28]
    
    for alpha in alpha_values:
        for window in window_sizes:
            column_name = f'target_value_ewm_alpha{alpha}_window{window}'
            df[column_name] = df['target_value'].ewm( span=window).mean()

        #f[column_name] = df['target_value'].ewm(alpha=alpha, span=window).mean()

    day_of_year_series = df['day_of_year']

    # Apply Fourier Transform 

    fft_result = np.fft.fft(df.target_value)

    # Store the Fourier coefficients or perform further analysis
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)

    df["fft_mag"] = magnitude
    df["fft_phase"] = phase

    return df

def main() -> int:
    """
    """
    ### get the data ###
    df = get_date("./dbs/intermittent/db.csv") 
    df = clean(df)
    df = get_date_features(df)
    df = get_timeseries_features(df)
    df = OHE(df)
    print(df.columns)
   # df['category'] = df['category'].fillna('')
    #print(df['day_of_week_index'])
    

    return 0

if (__name__ == "__main__"):    main()
