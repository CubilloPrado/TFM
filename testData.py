import yfinance as yf
import pandas as pd


tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "AMD", "NFLX",
    "JPM", "V", "MA", "DIS", "CRM", "HD", 
    "GS", "ORCL", "NKE", "CMCSA",  "T",
    "BAC", "CSCO", "XOM", "PFE", "WMT", "VZ", "IBM", "INTC",
    "KO", "PEP", "CVX", "MRK", "WFC", "TSM", "UNH", "BA", "CAT",
    "MMM", "PG", "C", "ADBE"
]


data = yf.download(tickers,'2010-1-1','2021-1-1', auto_adjust=True)['Close']
print(data.head())

import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_to_zscore(df):
    # Check if the DataFrame is empty
    if df.empty:
        return df
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to the data
    scaler.fit(df)
    
    # Transform the data to the Z-score
    standardized_data = scaler.transform(df)
    
    # Convert the standardized data back to a DataFrame
    standardized_df = pd.DataFrame(standardized_data, columns=df.columns, index=df.index)
    
    return standardized_df

standardized_df = standardize_to_zscore(data)
print(standardized_df)


def check_for_nan(df):
    nan_values = df.isna().sum().sum()
    if nan_values > 0:
        print("There are {} NaN values in the DataFrame.".format(nan_values))
    else:
        print("There are no NaN values in the DataFrame.")

# Example usage:
# Assuming you have a DataFrame named 'df', you can call the function like this:
# check_for_nan(df)

check_for_nan(data)

def check_for_nan(df):
    nan_values = df.isna().sum()
    tickers_with_nan = nan_values[nan_values > 0]
    if len(tickers_with_nan) > 0:
        print("The following tickers have NaN values:")
        for ticker, count in tickers_with_nan.items():
            print("- Ticker '{}': {} NaN values".format(ticker, count))
    else:
        print("There are no NaN values in the DataFrame.")

# Example usage:
# Assuming you have a DataFrame named 'df', you can call the function like this:
# check_for_nan(df)

check_for_nan(data)