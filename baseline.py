"""
Baseline Model for Housing Price Prediction Model 

Several things need to be done by this script:
    1. Load the data 
    2. Drop any data that is nonlinear 
    3. One-Hot Encode Categorical Data (includes "rating" columns like 'view') 
    4. Split data into training, validation, and testing.
    5. Train SGDRegressor up to 200 epochs (or more), recording MAE, RMSE, and R^2 after every epoch.
    6. Stop training early when validation MAE stops improving.
    7. Evaluate final model on the test set, print metrics
    8. Plot training/validation metrics (3 sub-plots) and save figures to 'results/[metric]_plot.png'
        [metric] = specific metric being plotted
        You can also save them all to one singular png if you can figure that out too.
    9. Save trained model ('models/baseline.joblib') and a JSON file containing all metrics ('results/baseline.json')
"""

# Imports

# Load Dataset

# Remove Interaction Columns 

# Separate target from features 

# One-Hot Encode Categories (this includes "rating" columns like 'view')

# Convert to float32

# Train/Validation/Testing Splits 

# SGDRegressor

# Training Loop, 200 epochs

# Final Eval

# Plot metrics over epochs

# Save Model and Metrics
