#python Model.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


df = pd.read_csv("clean/Clean-USA-Housing-Dataset.csv")

y = df.pop('price')

cat_cols = ['city','state', 'Zip Code']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.astype(np.float32)
y = y.astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

model = SGDRegressor(
    loss='squared_loss',
    penalty=None,
    learning_rate='constant',
    eta0=1e-3,
    max_iter=200,
    random_state=42,
    warm_start=True
)

n_epochs = 200

train_mae

print(df)