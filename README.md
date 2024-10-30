# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Import Necessary Libraries and Load Data
3. Split Dataset into Training and Testing Sets
4. Train the Model Using Stochastic Gradient Descent (SGD)
5. Make Predictions and Evaluate Accuracy
6. Generate Confusion Matrix
7. End

## Program & Output :
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: ARIVAZHAGAN G R
RegisterNumber:  212223040020
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
```
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
```
![o1](https://github.com/user-attachments/assets/4fa703d6-0ec6-425f-977a-9f0f24447c00)

```
X=df.drop('target',axis=1)
Y=df['target']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
```
```
sgd_clf.fit(X_train,Y_train)
```

![image](https://github.com/user-attachments/assets/9b82391d-59dc-4cce-887a-d4753da4cf82)
```
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(Y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

![image](https://github.com/user-attachments/assets/174c4930-15b0-442c-80b1-6c9eb7dc85a6)

```
cm=confusion_matrix(Y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```

![image](https://github.com/user-attachments/assets/1ec33122-91eb-4bd4-8c00-78fe676b9f89)

## Result :
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
