# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
data = pd.read_csv('rainfall in india 1901-2015.csv')
data.dropna(inplace=True)

# Define monthly columns and target variable
monthly_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
target = data['ANNUAL']

# One-hot encode the 'DIVISION' column and concatenate with monthly columns
features = pd.get_dummies(data[['DIVISION'] + monthly_columns], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and feature names
with open("rainfall_prediction_model.pkl", "wb") as file:
    pickle.dump((model, features.columns), file)

print("Model trained and saved as 'rainfall_prediction_model.pkl'")
