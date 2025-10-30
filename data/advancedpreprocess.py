# Import
import os 
import pandas as pd  
import numpy as np 

df = pd.read_csv("raw/RedFinDataset.csv")

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
# All YOY must be floats
# ADJUSTED_AVERAGE_NEW_LISTINGS (Float)
# OFF_MARKET_IN_TWO_WEEKS (int)
# ADJUSTED_AVERAGE_HOMES_SOLD (float)
# MEDIAN_NEW_LISTING_PRICE (float)
# MEDIAN_SALE_PRICE (float)
# MEDIAN_NEW_LISTING_PPSF (float)
# ACTIVE_LISTINGS (int)
# MEDIAN_DAYS_ON_MARKET (float)
# PERCENT_ACTIVE_LISTINGS_WITH_PRICE_DROP (float)
# AGE_OF_INVENTORY (float)
# WEEKS_OF_SUPPLY (float)
# MEDIAN_SALE_PPSF (float)

# Calculation 

# Lagging Function 

# Lagged Columns
# MEDIAN_SALE_PRICE_lag1 (1 period)
# MEDIAN_SALE_PRICE_lag3 (3 periods)
# MEDIAN_SALE_PRICE_lag12 (12 periods)
# ACTIVE_LISTINGS_lag1 (1 period)
# WEEKS_OF_SUPPLY_lag1 (1 period)
# MEDIAN_DAYS_ON_MARKET_lag12 (12 periods)

# Get Dataframe as a CSV 
output_dir = "clean"
df.to_csv(os.path.join(output_dir, "(Clean) RedFinDataset.csv"), index=False)

print(
    "✅ Data preprocessing complete – cleaned file saved to",
    os.path.join(output_dir, "(Clean) RedFinDataset.csv"),
)
