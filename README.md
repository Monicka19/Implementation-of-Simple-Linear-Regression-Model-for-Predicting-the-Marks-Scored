# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given data.


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
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/58070106-066f-4e2a-942b-45c3e14f31d9)
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/748b2b7e-3032-4059-9f51-fe254b078af0)
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/cdc9dcf8-6803-4d23-99b3-e5e44a2cafbd)
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/8ce69f68-d9cc-4c21-9205-9afc7d586c53)
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/c2ad4933-476a-471f-a799-b91e497b6d40)
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/8dfc3551-f117-4e33-b79e-b1f13a99863e)
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/7fa90424-4755-4f9d-85a7-9ec810bbbf63)
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/bf85e4ce-07a4-495a-bdb3-59f7b9271e2b)
![image](https://github.com/Monicka19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143497806/4c17ecef-55df-4ce7-8495-2446cf53f1bf)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
