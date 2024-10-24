# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program

Step 2: Import the required packages.

Step 3: Import the dataset to operate on.

Step 4: Split the dataset.

Step 5: Predict the required output.

Step 6: Stop the program.

## Program:

Program to implement the SVM For Spam Mail Detection..5

Developed by: SATHYAA R

RegisterNumber: 212223100052

```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
#Countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:

### head function
![Screenshot 2024-10-24 090242](https://github.com/user-attachments/assets/0d611ebe-0cdd-4483-8b55-91580fc911d0)

### info()
![Screenshot 2024-10-24 090256](https://github.com/user-attachments/assets/50ac310b-7aec-4fdc-baa7-60c532f6c428)

### Y prediction
![Screenshot 2024-10-24 090305](https://github.com/user-attachments/assets/1bc99e94-69bd-4ed0-9295-3fa7c1e9f702)

### Accuracy
![Screenshot 2024-10-24 090310](https://github.com/user-attachments/assets/f79527e2-099f-4bf9-869c-f11fbbf7aa8b)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
