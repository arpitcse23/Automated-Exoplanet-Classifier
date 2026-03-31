import pandas as pd
import xgboost as xgb
import joblib # This is for saving the trained model

print("--- Starting AI Model Training ---")

# 1. Load the training data
print("Loading exoTrain.csv...")
df = pd.read_csv('exoTrain.csv')

# 2. Prepare the data
# X contains the features (the brightness values of the stars)
# y contains the labels (1 for not a planet, 2 for a planet)
print("Preparing data for training...")
X = df.drop('LABEL', axis=1)
y = df['LABEL']

# We need to convert the labels to 0 (not a planet) and 1 (a planet)
y = y.replace({1: 0, 2: 1})

# 3. Train the XGBoost model
# XGBoost is a powerful and fast algorithm for this type of problem
print("Training XGBoost model... (This will take a few minutes)")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

print("Model training complete!")

# 4. Save the trained model to a file
print("Saving trained model to xgboost_model.joblib...")
joblib.dump(model, 'xgboost_model.joblib')

print("--- AI Model Training Finished Successfully ---")