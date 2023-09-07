# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MONICKA S
RegisterNumber: 212221220033 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
print("df.head():")
df.head()

print("df.tail():")
df.tail()

#Segregating data to variables
print("Array value of X:")
X=df.iloc[:,:-1].values
X

print("Array value of X:")
Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print("Values of Y prediction:")
Y_pred

#displaying actual values
print("Array values of Y test:")
Y_test

#graph plot for training data
print("Training set graph:")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
print("Test set graph:")
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Values of MSE,MAE and RMSE:")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE')
```

## Output:
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/cdc9dcf8-6803-4d23-99b3-e5e44a2cafbd)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
