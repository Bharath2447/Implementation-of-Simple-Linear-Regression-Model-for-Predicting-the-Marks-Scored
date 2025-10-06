# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Bharath K
RegisterNumber: 212224230036
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
print("Name: Bharath K")
print("Reg.No: 212224230036")
plt.scatter(X_train,Y_train,color="brown")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name: Bharath K")
print("Reg.No: 212224230036")
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name: Bharath K")
print("Reg.No: 212224230036")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae )
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:

Head:

<img width="168" height="207" alt="image" src="https://github.com/user-attachments/assets/21e58e34-e225-41cf-99e4-63ec406d3e7e" />

Tail:

<img width="178" height="201" alt="image" src="https://github.com/user-attachments/assets/6e142e4e-890d-4348-aa64-782320811f72" />

X:

<img width="147" height="541" alt="image" src="https://github.com/user-attachments/assets/fe584bbd-b0d6-407c-aafa-a57ab648f75c" />

Y:

<img width="716" height="47" alt="image" src="https://github.com/user-attachments/assets/31e1df36-1a12-4d21-9468-1cc54924f6bd" />

Y_pred:

<img width="708" height="53" alt="image" src="https://github.com/user-attachments/assets/5907eed9-e090-4a63-a156-f94fe6dd263c" />

Y_test:

<img width="562" height="27" alt="image" src="https://github.com/user-attachments/assets/03de139a-84f1-4fee-a47e-67aa5e6b116c" />

Training Set:

<img width="753" height="636" alt="image" src="https://github.com/user-attachments/assets/e5ece3f4-a5d6-4722-a032-9de63f13c2c1" />

Test set:

<img width="747" height="623" alt="image" src="https://github.com/user-attachments/assets/9ba20566-52ea-494b-b3b1-c9338499c6a3" />

Values:

<img width="253" height="122" alt="image" src="https://github.com/user-attachments/assets/058354f6-c631-4d2f-a0dd-5ae20b3b16f5" />










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
