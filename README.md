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
# Aim:         

## To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
### 1.Hardware – PCs
### 2.Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm: Predicting Student Marks Using Simple Linear Regression
#### 1. Start

#### 2. Input: Dataset containing number of study hours (X) and marks scored (Y).

#### 3. Load Dataset into memory.

#### 4. Preprocess Data:
  ##### Separate features (X = hours) and target variable (Y = marks).

#### 5. Split Dataset: Divide the dataset into training set and test set (e.g., 80% training, 20% testing).

#### 6. Train Model:
  ##### a. Initialize a Linear Regression model.
  ##### b. Fit the model on the training set (X_train, Y_train).

#### 7. Evaluate Model:
  ##### a. Use the trained model to predict marks on the test set (X_test).
  ##### b. Calculate accuracy metrics (Mean Squared Error, R² Score, etc.).

#### 8. Make Predictions:
  ##### a. Take user input (study hours).
  ##### b. Use the trained model to predict marks for the given input.

#### 9. Output: Display predicted marks.

#### 10. Stop

# Program
 python
#step 1: Import libraries and load the dataset (Hours vs Marks)
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

#step2 Create or Load data set 
df= pd.read_csv('student_scores.csv')
print('Dataset:\n',df.head(10))
df

#Step3 Separate features and target
x= df[['Hours']] #independent variable(2d)
y=df['Scores'] #dependent variable(1d)

#Step4 Train Test Split
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2,random_state=42)

#step5 Train linear regression model
model = lr()
model.fit(xtrain,ytrain)

#step6 prediction variable creation
ypred=model.predict(xtest)

#step7 Model evaluation
print("\nModel Parameters")
print('b0-intercept=',model.intercept_)
print('b1-Slope=',model.coef_[0])

print('\nEvaluation Metrics')
print('Mean Squared Errors:',mse(ytest,ypred))
print('R^2 Score:',r2(ytest,ypred))

#step 8 Visualisation
plt.figure(figsize=(12,8))
plt.scatter(x,y,color='red',label='Data point')
plt.plot(x,model.predict(x),color='green',linewidth=3,label='Regression Line')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Simple Linear Regression - Predictions of Marks')
plt.legend()
plt.grid(True)
plt.show()

#step 9 perfomance for given input
hours=float(input("Enter your studied hour:"))
prediction=model.predict([[hours]])
print(f'\nPredicted marks for you with {hours} hours of study = {prediction[0]:.2f}')


*/
```

## Output:

<img width="614" height="108" alt="image" src="https://github.com/user-attachments/assets/e0f9867b-7005-4fd7-ad35-9f816c42890a" />

<img width="1123" height="699" alt="image" src="https://github.com/user-attachments/assets/3cc45ba8-2b76-4705-abe3-0918f0d113fe" />


<img width="550" height="527" alt="image" src="https://github.com/user-attachments/assets/1b33f1c9-b497-4dfd-9b68-88ce0e527cee" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
