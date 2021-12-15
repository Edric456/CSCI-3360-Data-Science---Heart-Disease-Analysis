import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split


# In[ ]:


#Loading Data
pd.set_option("display.width", 3000)
pd.set_option('display.max_rows', 1000)
data = pd.read_csv("heart.csv")
print(data)


# In[ ]:


# Splitting Data

X_train, X_test, y_train, y_test =  train_test_split(data.iloc[:,:-1].values, data.iloc[:,13].values, test_size=0.25, random_state=0)

st_x=StandardScaler()
X_train=st_x.fit_transform(X_train)    
X_test=st_x.transform(X_test)


# In[ ]:


def KNN_training(num_neighbor):
    print(f'K = {num_neighbor}\n')
    model = KNeighborsClassifier(n_neighbors=num_neighbor)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print(metrics.classification_report(y_train, y_pred))
    print(metrics.confusion_matrix(y_train, y_pred), '\n')


# In[ ]:


KNN_training(1)
KNN_training(3)
KNN_training(5)
KNN_training(7)
KNN_training(9)
KNN_training(50)
KNN_training(100)


# In[ ]:


def KNN_testing(num_neighbor):
    print(f'K = {num_neighbor}\n')
    model = KNeighborsClassifier(n_neighbors=num_neighbor)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred), '\n')


# In[ ]:


KNN_testing(1)
KNN_testing(3)
KNN_testing(5)
KNN_testing(7)
KNN_testing(9)
KNN_testing(50)
KNN_testing(100)


# In[ ]:




