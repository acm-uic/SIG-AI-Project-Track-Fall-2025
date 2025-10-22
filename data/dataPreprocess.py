# Importing necessary libraries
import os
import pandas as pd 
import numpy as np 

# Loading the data
df = pd.read_csv('raw/USA Housing Dataset.csv')

# Deletion 
# Removing the columns we don't need
df.drop(columns=['street', 'country'], inplace = True)


# Adjustment
# Adjusting the columns to be easier to work with
df['date'] = pd.to_datetime(df['date'], errors='coerce') # Converts the column to a datetime datatype
df['price'] = np.log(df['price'])

# Calculation
# Creating new calculated columns

# The times are split as numbers as these are easier to train on
# Year not included because dataset limited to 2014
    # Time related
df['Day of Week'] = df['date'].dt.dayofweek
df['Season Sold'] = ((df['date'].dt.month - 1) // 3 + 1)

    # Subtraction
df['House Age']         = df['date'].dt.year - df['yr_built']
df['Renovation Age']    = df['date'].dt.year - df['yr_renovated']
df['Is Renovated']      = df['date'].dt.year - df['yr_renovated']

    # Ratios
df['Lot-Living Ratio']      = np.where(df['sqft_living']    != 0, df['sqft_lot']        / df['sqft_living'],    np.nan)
df['Basement Ratio']        = np.where(df['sqft_living']    != 0, df['sqft_basement']   / df['sqft_living'],    np.nan)
df['Areas Per Bedroom']     = np.where(df['bedrooms']       != 0, df['sqft_living']     / df['bedrooms'],       np.nan)
df['Bathrooms Per Bedroom'] = np.where(df['bedrooms']       != 0, df['bathrooms']       / df['bedrooms'],       np.nan)
df['Bedrooms Per Floor']    = np.where(df['floors']         != 0, df['bedrooms']        / df['floors'],         np.nan)

    # Interactions
df['Beds x Baths']              = df['bedrooms'] * df['bathrooms']
df['Sqft Living x Waterfront']  = df['sqft_living'] * df['waterfront']
df[['State', 'ZIP Code']]       = df['statezip'].str.split(expand=True)

# Deletion Pt 2
# Sometimes its better to have the calculated column over the raw data, so we drop those here
df.drop(columns=['statezip', 'yr_built', 'yr_renovated'], inplace=True)
# Since the data is all from the same year, the year built and renovated isn't necessary, since the model will just be deriving the age from it

# Get Dataframe as a CSV
output_dir = 'clean'
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, '(Clean) USA Housing Dataset.csv'), index=False)

print("✅ Data preprocessing complete – cleaned file saved to", os.path.join(output_dir, '(Clean) USA Housing Dataset.csv'))
