

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#Reading Dataset
df = pd.read_csv('new1.csv')
data=df.head()
print(data)
tail=df.tail(7)
print(tail)

plt.plot(df['severity_of_illness'],df['stay'])

X= df.iloc[:, 0:-2 ].values #Features

y= df.iloc[:, -1].values  #Labels

#Converting to numeric value using OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:,0])
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:,1])
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:,2])
#labelencoder_X = LabelEncoder()
#X[:, 3] = labelencoder_X.fit_transform(X[:,3])
#labelencoder_X = LabelEncoder()
#X[:, 4] = labelencoder_X.fit_transform(X[:,4])

onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]





#print(df.dtypes)



# Split the data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)



#KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test) 

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy knn:",result2)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver="liblinear")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
result3=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(result3)
result4=classification_report(y_test,y_pred)
print("Classification Report:")
print(result4)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy logistic Regression:", accuracy)


#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

result5 = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result5)
result6 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result6)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy Random:",result2)

