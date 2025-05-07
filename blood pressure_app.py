import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Synthetic dataset (replace with real data)
data = {
    'age': [25, 30, 45, 50, 23, 35, 65, 70, 40, 55],
    'weight': [65, 70, 80, 85, 60, 75, 90, 95, 78, 88],
    'height': [165, 170, 175, 180, 160, 168, 172, 176, 174, 182],
    'systolic_bp': [120, 122, 130, 135, 118, 128, 145, 150, 132, 140]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['age', 'weight', 'height']]
y = df['systolic_bp']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plotting
plt.scatter(y_test, y_pred)
plt.xlabel("Actual BP")
plt.ylabel("Predicted BP")
plt.title("Actual vs Predicted Blood Pressure")
plt.grid(True)
plt.show()
