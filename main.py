# %%
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# %%
# Loading the dataset
salary_df = pd.read_csv('Salary_Data.csv')

# %%
salary_df.head()

# %%
# Checking the dimensions of the DataFrame
salary_df.shape

# %%
# Checking for any missing values
salary_df.isnull().sum()

# %%
# Separating the features
X = salary_df.drop(['Salary'], axis=1)
y = salary_df['Salary']

# %%
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Defining a custom Linear Regression class
class MyLinearRegression:

  def __init__(self, lr = 0.01, n_iters = 1000):
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None
    self.cost_history = []

  def compute_cost(self, y, y_pred, m):
    return (1/(2*m))*np.sum((y_pred-y)**2)

  def fit(self, X, y):
    m, n = X.shape
    self.weights = np.zeros(n)
    self.bias = 0

    # Gradient Descent loop
    for _ in range(self.n_iters):
      y_pred = np.dot(X, self.weights) + self.bias
      cost = self.compute_cost(y, y_pred, m)
      self.cost_history.append(cost)

      # Calculate gradients for weights (dw) and bias (db)
      dw = (1/m)*np.dot(X.T, (y_pred-y))
      db = (1/m)*np.sum(y_pred - y)

      # Update weights and bias using the learning rate and gradients
      self.weights -= self.lr*dw
      self.bias -= self.lr*db

  def predict(self, X):
    # Make predictions using the learned weights and bias
    y_pred = np.dot(X, self.weights) + self.bias
    return y_pred

# %%
# Creating an instance of the custom MyLinearRegression model with default parameters
mylr = MyLinearRegression()

# %%
# Training the custom linear regression model using the training data
mylr.fit(X_train, y_train)

# %%
# Printing the learned weights, bias and cost history of the trained model
print("Weights:", mylr.weights)
print("Bias:", mylr.bias)
print("Cost History:")
for i, cost in enumerate(mylr.cost_history):
    print(f"Iteration {i+1:4d}: Cost = {cost:,.4f}")

# %%
# Making predictions on the training data using the trained model
y_pred_train = mylr.predict(X_train)
print(y_pred_train)

# %%
# Calculating evaluation metrics (MSE, RMSE, R2 Score) on the training set
mse_train = np.mean((y_train - y_pred_train) ** 2)

rmse_train = np.sqrt(mse_train)

ss_res_train = np.sum((y_train - y_pred_train) ** 2)
ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
r2_train = 1 - (ss_res_train / ss_tot_train)


print("MSE on Training Set:", mse_train)
print("RMSE on Training Set:", rmse_train)
print("R2 Score on Training Set:", r2_train)

# %%
# Making predictions on the testing data using the trained model
y_pred_test = mylr.predict(X_test)
print(y_pred_test)

# %%
# Calculating evaluation metrics (MSE, RMSE, R2 Score) on the testing set
mse_test = np.mean((y_test - y_pred_test) ** 2)

rmse_test = np.sqrt(mse_test)

ss_res_test = np.sum((y_test - y_pred_test) ** 2)
ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
r2_test = 1 - (ss_res_test / ss_tot_test)

print("MSE on Test Set:", mse_test)
print("RMSE on Test Set:", rmse_test)
print("R2 Score on Test Set:", r2_test)

# %%
# Visualizing the training and testing data along with the best-fit line from the trained model
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, marker='x', color='blue', label='Training data')
plt.scatter(X_test, y_test, marker='o', color='green', label='Testing data')
plt.plot(X_train, mylr.predict(X_train), color='red', linewidth=2, label='Best fit line')
plt.title('Salary vs Experience with Linear Regression Fit')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# %%
# Plotting the cost history to visualize how the cost function decreased during training
plt.figure(figsize=(8, 6))
plt.plot(mylr.cost_history)
plt.title('Cost Function History')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.show()

# %%
# Example prediction-1
predicted_salary_9_3 = 9.3 * mylr.weights[0] + mylr.bias
print(f"Predicted salary for 9.3 years of experience will be ${predicted_salary_9_3}")

# %%
# Example prediction-2
predicted_salary_5 = 5 * mylr.weights[0] + mylr.bias
print(f"Predicted salary for 5 years of experience will be ${predicted_salary_5}")


