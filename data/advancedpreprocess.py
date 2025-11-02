# Import
import os

import numpy as np
import pandas as pd

# This will NOT be on the GitHub, you must download it from RedFin
df = pd.read_csv("raw/RedFinDataset.csv", on_bad_lines="skip", engine="python")

# Adjustment
df["PERIOD_BEGIN"] = pd.to_datetime(df["PERIOD_BEGIN"], errors="coerce")
df["PERIOD_END"] = pd.to_datetime(df["PERIOD_END"], errors="coerce")

# Calculation
df["DATE"] = df["PERIOD_BEGIN"] + (df["PERIOD_END"] - df["PERIOD_BEGIN"]) / 2
df["SEASON_SOLD"] = (df["DATE"].dt.month - 1) // 3 + 1
df["MONTH_SOLD"] = df["DATE"].dt.month
df["YEAR_SOLD"] = df["DATE"].dt.year

# >1, sells above asking; <1, sells below asking
df["PPSF_RATIO"] = df["MEDIAN_SALE_PPSF"] / df["MEDIAN_NEW_LISTING_PPSF"]

# listings per weeks of supply
df["SUPPLY_DENSITY"] = df["ACTIVE_LISTINGS"] / df["WEEKS_OF_SUPPLY"]

df["OFF_MARKET_RATE"] = df["OFF_MARKET_IN_TWO_WEEKS"] / df["ACTIVE_LISTINGS"]

# Deletion
df.drop(columns=["DATE", "PERIOD_END"], inplace=True)

# Get Dataframe as a CSV
output_dir = "clean"
df.to_csv(os.path.join(output_dir, "(Clean) RedFinDataset.csv"), index=False)

print(
    "✅ Data preprocessing complete – cleaned file saved to",
    os.path.join(output_dir, "(Clean) RedFinDataset.csv"),
)
