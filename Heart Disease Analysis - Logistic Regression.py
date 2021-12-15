import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#Loading Data
pd.set_option("display.width", 3000)
pd.set_option('display.max_rows', 1000)
data = pd.read_csv("heart.csv")
print(data)


#Splitting Data

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data["target"], test_size=0.25, random_state=4)

X_train.to_csv("X_train.csv")
X_test.to_csv("X_test.csv")
y_train.to_csv("y_train.csv")
y_test.to_csv("y_test.csv")


#Logisitic Regression Model

logisticRegr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state = 4) #Making an instance of the model.

logisticRegr.fit(X_train, y_train) #Fitting the model on the training data. 


#Equation for Logistic Regression
print("Logistic Regression Model: \n")

print(f"Intercept: {logisticRegr.intercept_}")
print("")
print(f"Coefficents: {logisticRegr.coef_}")


#Testing on the Training Set
print("\nTraining Set: \n")
y_pred_train = logisticRegr.predict(X_train) #Testing model on training set. 


#Classification Matrix
print(metrics.classification_report(y_train, y_pred_train))

#Confusion Matrix
cnf_train = metrics.confusion_matrix(y_train, y_pred_train)
cnf_train
sns.heatmap(cnf_train, annot=True, fmt="d")

b, t = plt.ylim()
plt.savefig("trainconf.png")
plt.title("Confusion Matrix for Training Set")
plt.show()



#Testing on the Testing Set
print("Testing Set: \n")
y_pred_test = logisticRegr.predict(X_test) #Testing model on testing set. 


#Classification Matrix
print(metrics.classification_report(y_test, y_pred_test))

#Confusion Matrix
cnf_test = metrics.confusion_matrix(y_test, y_pred_test)
cnf_test
sns.heatmap(cnf_test, annot=True, fmt="d")

b, t = plt.ylim()
plt.title("Confusion Matrix for Testing Set")
plt.savefig("testconf.png")
plt.show()