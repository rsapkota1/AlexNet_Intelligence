import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np

#Read gray matter and white matter
df_gray=pd.read_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/Gray_fluid_7874.csv').iloc[:,2:]
df_white=pd.read_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/White_fluid_7874.csv').iloc[:,2:]
df_gray_white_concat_all=pd.concat([df_gray, df_white],axis=1)

#Read cognition data
df_cognition=pd.read_csv('/data/users4/rsapkota/SCCA/Gray_Matter/Cognition_Uncorrected_Baseline_7874.csv')
df_cognition_final=df_cognition.drop(['Unnamed: 0','ID'],axis=1)

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(df_gray_white_concat_all,  df_cognition_final.iloc[:,6], test_size=0.2, random_state=42)

#Implement Lasso Regression
lasso=Lasso()
parameters={'alpha':[0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='r2',cv=5)    
lasso_regressor.fit(X_train,y_train)
# Best hyperparameters
best_alpha = lasso_regressor.best_params_['alpha']
print(best_alpha)
# Creating a new Lasso regression model with the best hyperparameters
lasso_best = Lasso(alpha=best_alpha)
# Training the model with the best hyperparameters
lasso_best.fit(X_train, y_train)
# Predicting on the data
y_pred = lasso_best.predict(X_test)
# Calculating R-squared
r2 = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)

print("Best Alpha:", best_alpha)
print("Mean Square Error:",mse_test)
print("R-squared:", r2)

# Retrieving the coefficient values
coefficients = lasso_best.coef_
intercept = lasso_best.intercept_

print("Intercept:", intercept)
print("Coefficients:", coefficients)

#Getting the important features
print(np.sum(coefficients!=0))