import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/raw/USA Housing Dataset.csv')

df.drop(columns = ['country', 'street'], inplace=True)
df.fillna(method='ffill', inplace=True)

df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
df['price'] = np.log(df['price'])