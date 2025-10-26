
#python Model.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

relative_path = "clean/Clean-USA-Housing-Dataset.csv"
absolute_path = os.path.abspath(relative_path)

df = pd.read_csv("clean/Clean-USA-Housing-Dataset.csv")

interaction_cols = [
    "Lot-Living Ratio",
    "Basement Ratio",
    "Areas Per Bedroom",
    "Bathrooms Per Bedroom",  # |--- List of interaction columns
    "Bedrooms Per Floor",
    "Beds x Baths",
    "Sqft Living x Waterfront",
]
df = df.drop(columns=interaction_cols)  # Drop interaction_cols
df = df.drop(columns=["date"])  # We cannot convert `date` to float32



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

train_mae, val_mae = [], []
train_rmse, val_rmse = [], []
train_r2, val_r2 = [], []

for epoch in range(n_epochs):
    model.partial_fit(X_train, y_train)

    #Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Metrics
    train_mae.append(mean_absolute_error(y_train, y_pred_train))
    val_mae.append(mean_absolute_error(y_val, y_pred_val))

    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    val_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    train_r2.append(r2_score(y_train, y_pred_train))
    val_r2.append(r2_score(y_val, y_pred_val))
    if epoch >0 and val_mae[-1] > min(val_mae[:-1]):
        print(f'Early stopping after epoch {epoch+1}')
        break

y_pred_test = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)


epochs = range(1, len(train_mae) + 1)

plt.figure(figsize=(12, 4))

#MAE
plt.subplot(1,3,1)
plt.plot(epochs, train_mae, label="Train MAE")
plt.plot(epochs, val_mae, label="Val MAE", linestyle="--")


plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Mean Absolute Error")
plt.legend()

#RMSE
plt.subplot(1,3,2)
plt.plot(epochs, train_rmse, label="Train RSME")
plt.plot(epochs, val_rmse, label="Val MAE", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("RSME")
plt.title("Root Mean Squared Error")
plt.legend()

#r2
plt.subplot(1,3,3)
plt.plot(epochs, train_r2, label="Train r2")
plt.plot(epochs, val_r2, label="Val r2")
plt.xlabel("Epoch")
plt.ylabel("r2")
plt.title("Coefficent of determination")
plt.legend()

plt.show()