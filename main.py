
# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# IMPORTING THE DATA
BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / "data" / "Salary_Data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# SPLITTING THE DATA INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# TRAINING THE SIMPLE LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(X_test)

# EVALUATING THE MODEL
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Model Evaluation Results")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# VISUALISING THE TRAINING SET RESULTS
plt.scatter(X_train, y_train, color="red")

sorted_indices = np.argsort(X_train[:, 0])
X_sorted = X_train[sorted_indices]

plt.plot(X_sorted, regressor.predict(X_sorted), color="blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.tight_layout()
plt.show()

# VISUALISING THE TEST SET RESULTS
plt.scatter(X_test, y_test, color="red")
plt.plot(X_sorted, regressor.predict(X_sorted), color="blue")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.tight_layout()
plt.show()

# MAKING A SINGLE PREDICTION
print("Predicted Salary for 9.3 years of experience:")
print(regressor.predict([[9.3]])[0])
