from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # You can choose other metrics like R2 score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression


import pandas as pd
import numpy as np
import pickle

# Define the CSV file path and headers to select
filename = "BeamAIData.csv"
headers_to_select_x = ["BeamSpan","BeforeSpan","AfterSpan","Load","Depth","Width","ConcreteFc","RebarsFy"]
headers_to_select_y = ["LeftTop","LeftBot","RightTop","RightBot","AvLeft","AvRight"]

# Read the CSV file using pandas
data = pd.read_csv(filename)

# Select the desired columns
X_raw = data[headers_to_select_x].to_numpy().astype(np.float32) 
y_raw = data[headers_to_select_y].to_numpy().astype(np.float32)

# min_values = X_raw.min(axis=0)
# max_values = X_raw.max(axis=0)
# X = (X_raw - min_values) / (max_values - min_values)

# # Normalize output data (optional)
# y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = X_raw
y = y_raw

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)

# Define and train the MultiOutputRegressor with Linear Regression as the base model
model = MultiOutputRegressor(estimator=LinearRegression())
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print(y_pred.shape)

# Evaluate model performance on the test set using mean squared error (MSE) for each target variable
for i in range(y.shape[1]):
    mse = mean_squared_error(y_test[:, i], y_pred[:, i])
    print("y truth:", y_test[:, i])
    print("y_predicted: ", y_pred[:, i])
    print(f"Mean Squared Error for target {i+1}: {mse:.8f}")

with open('multioutput_model.pkl', 'wb') as f:
  # Pickle the model object
  pickle.dump(model, f)

print("Model saved successfully to multioutput_model.pkl")
print(X_test[1000])
print(y_test[1000])
