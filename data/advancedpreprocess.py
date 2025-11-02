# Import
import os 
import pandas as pd  
import numpy as np 

# This will NOT be on the GitHub, you must download it from RedFin
df = pd.read_csv("raw/RedFinDataset.tsv", sep='\t')

# Deletion 
df = df.drop(columns=[
    "REGION_TYPE_ID",
    'REGION_NAME', 
    'REGION_ID', 
    'AVERAGE_PENDING_SALES_LISTING_UPDATES',
    'AVERAGE_PENDING_SALES_LISTING_UPDATES_YOY',
    'MEDIAN_DAYS_TO_CLOSE',
    'MEDIAN_DAYS_TO_CLOSE_YOY',
    'MEDIAN_PENDING_SQFT',
    'MEDIAN_PENDING_SQFT_YOY',
    'LAST_UPDATED'])

# Adjustment
df['PERIOD_BEGIN'] = pd.to_datetime(df['PERIOD_BEGIN'], errors="coerce")
df['PERIOD_END'] = pd.to_datetime(df['PERIOD_END'], errors="coerce")
df.astype({
    'REGION_TYPE': 'string',
    'DURATION': 'string',
    'ADJUSTED_AVERAGE_NEW_LISTINGS': 'float64',
    'ADJUSTED_AVERAGE_NEW_LISTINGS_YOY': 'float64',
    'OFF_MARKET_IN_TWO_WEEKS': 'float64',
    'OFF_MARKET_IN_TWO_WEEKS_YOY': 'float64',
    'ADJUSTED_AVERAGE_HOMES_SOLD': 'float64',
    'ADJUSTED_AVERAGE_HOMES_SOLD_YOY': 'float64',
    'MEDIAN_NEW_LISTING_PRICE': 'float64',
    'MEDIAN_NEW_LISTING_PRICE_YOY': 'float64',
    'MEDIAN_SALE_PRICE': 'float64',
    'MEDIAN_SALE_PRICE_YOY': 'float64',
    'MEDIAN_NEW_LISTING_PPSF': 'float64',
    'MEDIAN_NEW_LISTING_PPSF_YOY': 'float64',
    'ACTIVE_LISTINGS': 'float64',
    'ACTIVE_LISTINGS_YOY': 'float64',
    'MEDIAN_DAYS_ON_MARKET': 'float64',
    'MEDIAN_DAYS_ON_MARKET_YOY': 'float64',
    'PERCENT_ACTIVE_LISTINGS_WITH_PRICE_DROPS': 'float64',
    'PERCENT_ACTIVE_LISTINGS_WITH_PRICE_DROPS_YOY': 'float64',
    'AGE_OF_INVENTORY': 'float64',
    'AGE_OF_INVENTORY_YOY': 'float64',
    'WEEKS_OF_SUPPLY': 'float64',
    'WEEKS_OF_SUPPLY_YOY': 'float64',
    'MEDIAN_SALE_PPSF': 'float64',
    'MEDIAN_SALE_PPSF_YOY': 'float64',
    })

# Calculation
df['DATE'] = df['PERIOD_BEGIN'] + (df['PERIOD_END'] - df['PERIOD_BEGIN'])/2
df['SEASON_SOLD'] = (df['DATE'].dt.month - 1) // 3 + 1
df['MONTH_SOLD'] = df['DATE'].dt.month
df['YEAR_SOLD'] = df['DATE'].dt.year 

# >1, sells above asking; <1, sells below asking
df['PPSF_RATIO'] = df['MEDIAN_SALE_PPSF'] / df['MEDIAN_NEW_LISTING_PPSF']

# listings per weeks of supply
df['SUPPLY_DENSITY'] = df['ACTIVE_LISTINGS'] / df['WEEKS_OF_SUPPLY']

df['OFF_MARKET_RATE'] = df['OFF_MARKET_IN_TWO_WEEKS'] / df['ACTIVE_LISTINGS']

# Lagging Function 
def lagData(df, col, n, newColName):
    df = df.copy()
    df = df.sort_values(['REGION_TYPE', 'PERIOD_BEGIN'])
    num = {
            '12 weeks':12,
            '4 weeks':4,
            '1 weeks':1
            }
    df['TARGET_DATE'] = df.apply(lambda r: r['PERIOD_BEGIN'] - pd.Timedelta(weeks=(num[r['DURATION']] * n)), axis=1)

    lookup = df[['REGION_TYPE', 'PERIOD_BEGIN', col]].copy()
    lookup.rename(columns={col: newColName}, inplace=True)
    lookup.rename(columns={'PERIOD_BEGIN': 'lookupDate'}, inplace=True)

    df = pd.merge(
            df, 
            lookup, 
            left_on=['REGION_TYPE', 'TARGET_DATE'], 
            right_on=['REGION_TYPE', 'lookupDate'], 
            how='left', 
            suffixes=('', '_lag')
            )

    df.drop(columns=['TARGET_DATE', 'lookupDate'], inplace=True)

    return df

# Lagged Columns
# MEDIAN_SALE_PRICE_lag1 (1 period)
df = lagData(df, 'MEDIAN_SALE_PRICE', 1, 'MEDIAN_SALE_PRICE_lag1')

# MEDIAN_SALE_PRICE_lag3 (3 periods)
df = lagData(df, 'MEDIAN_SALE_PRICE', 3, 'MEDIAN_SALE_PRICE_lag3')

# MEDIAN_SALE_PRICE_lag12 (12 periods)
df = lagData(df, 'MEDIAN_SALE_PRICE', 12, 'MEDIAN_SALE_PRICE_lag12')

# ACTIVE_LISTINGS_lag1 (1 period)
df = lagData(df, 'ACTIVE_LISTINGS', 1, 'ACTIVE_LISTINGS_lag1')

# WEEKS_OF_SUPPLY_lag1 (1 period)
df = lagData(df, 'WEEKS_OF_SUPPLY', 1, 'WEEKS_OF_SUPPLY_lag1')

# MEDIAN_DAYS_ON_MARKET_lag12 (12 periods)
df = lagData(df, 'MEDIAN_DAYS_ON_MARKET', 12, 'MEDIAN_DAYS_ON_MARKET_lag12')

# Get Dataframe as a CSV 
output_dir = "clean"
df.to_csv(os.path.join(output_dir, "(Clean) RedFinDataset.csv"), index=False)

print(
    "✅ Data preprocessing complete – cleaned file saved to",
    os.path.join(output_dir, "(Clean) RedFinDataset.csv"),
)
