# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
health_data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv')

# 🔹 Basic Overview
print("\n🔹 Dataset Shape:", health_data.shape)
print("\n🔹 Dataset Info:\n")
print(health_data.info())

# 🔹 Handle Missing Values & Duplicates
health_data.drop_duplicates(inplace=True) 
health_data.fillna(health_data.median(numeric_only=True), inplace=True)  

# Convert categorical features using factorize() (instead of get_dummies)
categorical_cols = health_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    health_data[col], _ = pd.factorize(health_data[col])

# 🔹 Drop Highly Correlated Features (Threshold > 0.9)
corr_matrix = health_data.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

health_data.drop(columns=to_drop, inplace=True)
print(f"\n🔹 Dropped {len(to_drop)} highly correlated features.")

# 🔹 Normalize Data for Regularization
scaler = StandardScaler()
health_data_scaled = scaler.fit_transform(health_data)
health_data = pd.DataFrame(health_data_scaled, columns=health_data.columns)

# 🔹 Final Dataset Info
print("\n🔹 Final Dataset Shape:", health_data.shape)
print("\n🔹 Sample Processed Data:\n", health_data.head())
print("\n🔹 Column names:\n", health_data.columns)

# Divide Train and Test Datasets
from sklearn.model_selection import train_test_split

# Attempting to predict Obesity Rate
y = health_data['Obesity_prevalence']
X = health_data.drop(columns=['fips', 'CNTY_FIPS', 'COUNTY_NAME', 'Obesity_prevalence'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=117)

# Performing linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(f'Model intercept: {model.intercept_}')
print(f'Model coefficients: {model.coef_}')

predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Model mean squared error: {mse}')
print(f'Model r2 score: {r2}')

# Apply L1
from sklearn.linear_model import Lasso

l1_model = Lasso(alpha=0.1,max_iter=200)
l1_model.fit(X_train, y_train)

l1_predictions = l1_model.predict(X_test)

l1_mse = mean_squared_error(y_test, l1_predictions)
l1_r2 = r2_score(y_test, l1_predictions)
print(f'L1 model mean squared error: {l1_mse}')
print(f'L1 model r2 score: {l1_r2}')

# Fine-tune hyperparameters with GridSearchCV

from sklearn.model_selection import GridSearchCV

hyperparams = {
    'alpha': np.arange(0.0001, 1.0, 0.01)
}

grid = GridSearchCV(l1_model, hyperparams, scoring='r2', cv=5)

# Suppress warnings due to incopatibilities or converges
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)
best_alpha = grid.best_params_['alpha']
print(f'Best alpha: {best_alpha}')

# Re-run optimized Lasso model
l1_model = Lasso(alpha=best_alpha,max_iter=200)
l1_model.fit(X_train, y_train)

l1_predictions = l1_model.predict(X_test)

l1_mse = mean_squared_error(y_test, l1_predictions)
l1_r2 = r2_score(y_test, l1_predictions)
print(f'L1 model mean squared error: {l1_mse}')
print(f'L1 model r2 score: {l1_r2}')

