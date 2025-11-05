import pandas as pd 
import numpy as np

df = pd.read_csv('clean/(Clean) USA Housing Dataset.csv')
#dropping the unnecessary columns
features = df.columns.tolist()
features.remove('id')
#The Independent and Dependent Variable 
X=df[features].astype(np.float64)
y=df.pop('price').astype(np.float32)

#TRAINING + TESTING THE MODEL
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Model Type 
model = SGDRegressor(
    loss='squared_loss',
    penalty=None,
    learning_rate='constant',
    eta0=0.001,
    random_state=42,
    max_iter=200,
    warm_start=True
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Training Loop & Metrics
n_epochs = 200
train_mae, val_mae = [], []
train_rsmse, val_rsmse = [], []
train_r2, val_r2 = [], []

for epoch in range(n_epochs):
    model.partial_fit(X_train, y_train)
    
    # Predictions 
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Metrics Calculation
    train_mae.append(mean_absolute_error(y_train, y_pred_train))
    val_mae.append(mean_absolute_error(y_val, y_pred_val))

    train_rsmse.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    val_rsmse.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    train_r2.append(r2_score(y_train, y_pred_train))
    val_r2.append(r2_score(y_val, y_pred_val))

    if epoch > 0 and val_mae[-1] > min(val_mae[:-1]):
        print(f"Early stopping after epoch {epoch+1}")
        break
    
#Final Evaluation on Test Set
y_pred_test = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rsmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

dump(model, 'models/baseline.joblib')
json.dump(baseline_metrics, f, indent=4)


