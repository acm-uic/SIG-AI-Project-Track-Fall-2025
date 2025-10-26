#python Model.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("clean/Clean-USA-Housing-Dataset.csv")

y = df.pop('price')

cat_cols = ['city','state', 'Zip Code']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.astype(np.float32)
y = y.astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(
    #code
)

print(df)