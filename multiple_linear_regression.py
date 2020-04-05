#Multiple Linear Regression
# In this program we determine which factors influence profitability the most when it comes to various expenditures like R&D spend, admin spend etc based on data collected among 50 startups.
# The aim is to give an indication to potential investors about the companies which are likely to give out the highest returns.
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the datasets
dataset = pd.read_csv('50_Startups.csv') #Run the entire program if it's causing problem here
#To get a matrix of data
X = dataset.iloc[:, :-1].values #All data except for the last coloumn; : to get the entire coloumn
y = dataset.iloc[:, 4].values #Here 4 is the index of the coloumn

# Data Preprocessing Template
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#Encode third index
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3]) #One hot encoding on the third coloumn
X = onehotencoder.fit_transform(X).toarray() 

#Avoid Dummy variable trap
X = X[:, 1:]

#Splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #Test size is 20% of the whole dataset, random size used to generate psuedo random number


"""#Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() #Creating a standard scaler object
X_train = sc_X.fit_transform(X_train) #For training dataset we need to fit and transform
X_test = sc_X.transform(X_test) #For testing dataset we need to transform"""

#Fitting multiple linear regression into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Building optimal solution using Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#Original matrix of features
X_opt = X[:, [0,1,2,3,4,5]]
#OLS (Ordinary least squares)
#Parameters - endog (dependant variable), exog(number of obs x K no of regressors. Intercept added by the user(X)), 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Remove index based on the significance level
X_opt = X[:, [0,1,2,3,4,5]]
#OLS (Ordinary least squares)
#Parameters - endog (dependant variable), exog(number of obs x K no of regressors. Intercept added by the user(X)), 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
#OLS (Ordinary least squares)
#Parameters - endog (dependant variable), exog(number of obs x K no of regressors. Intercept added by the user(X)), 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
#OLS (Ordinary least squares)
#Parameters - endog (dependant variable), exog(number of obs x K no of regressors. Intercept added by the user(X)), 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
#OLS (Ordinary least squares)
#Parameters - endog (dependant variable), exog(number of obs x K no of regressors. Intercept added by the user(X)), 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
#OLS (Ordinary least squares)
#Parameters - endog (dependant variable), exog(number of obs x K no of regressors. Intercept added by the user(X)), 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()