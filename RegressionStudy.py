# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:34:56 2019

@author: Acer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, [1,2,3,4,6,7,8,9,10,11,12]].values
Y = dataset.iloc[:, [5,1]].values

#filling in missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy= 'mean', axis = 0)
imputer = imputer.fit(Y[:, :-1])
Y[:, :-1] = imputer.transform(Y[:, :-1])
Y = Y[:, :-1]
Y = Y[:43824 , :]
X = X[:43824 , :]

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
onehotencoder = OneHotEncoder(categorical_features = [7])
X = onehotencoder.fit_transform(X).toarray()

#dummy variable trap 
X = X[:, 1:]



#splitting data into test and train sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#fitting simple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

plt.plot(Y_test, Y_pred)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((43824,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11, 12]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()]

X_opt = X[:, [0, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11, 12]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 5, 6, 7, 8, 9 , 10, 11, 12]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


# TAKEN X_OPT AS optimum value for X and BUILDING THE MODEL AGAIN

X_train, X_test, Y_train, Y_test = train_test_split(X_opt,Y, test_size = 0.2, random_state = 0)

#fitting simple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred_SLR = regressor.predict(X_test)

plt.plot(Y_test, Y_pred)

#applying polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly_train = poly_reg.fit_transform(X_train)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly_train, Y_train)

Y_pred_PolyREG = lin_reg_poly.predict(poly_reg.fit_transform(X_test))

plt.plot(Y_test, Y_pred_PolyREG)

#applying random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor_RF = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor_RF.fit(X_train,Y_train)
# Predicting a new result
y_pred_RF = regressor_RF.predict(X_test)



