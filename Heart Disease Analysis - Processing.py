import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split


#Loading Data
pd.set_option("display.width", 3000)
pd.set_option('display.max_rows', 1000)
data = pd.read_csv("heart.csv")
print(data)


#Correlation Heatmap
corr = data.corr()
plt.figure(figsize=(18,14))
sns.heatmap(corr, annot=True, cmap='Reds')
b, t = plt.ylim()
plt.title("Feature Correlation Heatmap")
plt.show()


#Summary Statistics for Numerical Attributes:
for column in data.columns:
    if column != "sex" and column != "fbs" and column != "exang" and column != "target":
            print("")
            print(f"{column} Summary:")
            print(f"\tMinimum Value: {data[column].min()}")
            print(f"\tMaximum Value: {data[column].max()}")
            print(f"\tMedian: {data[column].median()}")
            print(f"\tMean: {data[column].mean()}")
            print(f"\tStandard Deviation: {data[column].std()}")
            
            
            
#Grouped Bar Graphs for Categorical Attributes 
#Setting up graphs
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['figure.dpi'] = 100


fig, ax = plt.subplots(1, 3)


#List of categorical attributes 
categories = ["sex", "fbs", "exang"]

#Variable that keeps track of which graph is being made.
graphNum = 0


#For-loop. Creates grouped bar graphs of categorical attributes. 
for category in categories:
    data_group = data.groupby(category) #Seperates data by current attribute.
    
    
    
    
    #row = int(graphNum / 2) #The row of the current graph.
    #col = graphNum % 2 #The column of the current graph.
    
    #Since all categorical attributes are binary values represented by 0s and 1s, they can all be graphed as followed:
    ax[graphNum].bar([-0.5, 0.5], [len(data_group.get_group(0).groupby("target").get_group(0)), len(data_group.get_group(0).groupby("target").get_group(1))], 0.35, color="red", label=f"{category} = 0")
    ax[graphNum].bar([-0.15, 0.85], [len(data_group.get_group(1).groupby("target").get_group(0)), len(data_group.get_group(1).groupby("target").get_group(1))], 0.35, color="green", label=f"{category} = 1")
    
    #X-labels.
    ax[graphNum].set_xticks([-0.37, 0.62]) 
    ax[graphNum].set_xticklabels(["target = 0", "target = 1"])
    
    #Legend
    ax[graphNum].legend() #Legend.
    
    #Title
    ax[graphNum].set_title(f"Distribution of data across {category}")
    
    graphNum += 1 #Increment graphnum by 1.
    
plt.show()



#Grouped Histograms for Numerical Attributes 

data_target = data.groupby("target") #Grouping data frame by target.

data_group_0 = data_target.get_group(0) #All the entries in which the response variable is 0.
data_group_1 = data_target.get_group(1) #All the entries in which the reponse variable is 1. 

fig_2, ax_2 = plt.subplots(2, 5) #There are 10 numerical attributes. 

histNum = 0 #Keeps track of which specific graph is being made. 



#for-loop to create Histograms. 
for column in data.columns:
    if column != "sex" and column != "fbs" and column != "exang" and column != "target":
        
        #Location of the current graph.
        row = int(histNum / 5) 
        col = histNum % 5
        
        #Making the graph.
        ax_2[row][col].hist(data_group_0[column], histtype="step", color="red", edgecolor="red", label="target = 0", fill=False)
        ax_2[row][col].hist(data_group_1[column], histtype="step", color="green", edgecolor="green", label="target = 1", fill=False)
        
        ax_2[row][col].set_title(f"Distribution of {column}")
        ax_2[row][col].legend()
        
        histNum += 1 #Increment histNum by 1. 

plt.show()


#Histograms for Numerical Attributes
sns.set()

# https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html
fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5)

ax0.hist(data["age"], histtype='bar')
ax0.set_title('Age')

ax1.hist(data["cp"], histtype='bar')
ax1.set_title('Cp')

ax2.hist(data["trestbps"], histtype='bar')
ax2.set_title('Trestbps')

ax3.hist(data["chol"], histtype='bar')
ax3.set_title('Chol')

ax4.hist(data["restecg"], histtype='bar')
ax4.set_title('Restecg')

ax5.hist(data["thalach"], histtype='bar')
ax5.set_title('Thalach')

ax6.hist(data["oldpeak"], histtype='bar')
ax6.set_title('Oldpeak')

ax7.hist(data["slope"], histtype='bar')
ax7.set_title('Slope')

ax8.hist(data["ca"], histtype='bar')
ax8.set_title('Ca')

ax9.hist(data["thal"], histtype='bar')
ax9.set_title('Thal')

fig.tight_layout()
plt.show()


# Splitting Data

X_train, X_test, y_train, y_test =  train_test_split(data.iloc[:,:-1].values, data.iloc[:,13].values, test_size=0.25, random_state=0)

st_x=StandardScaler()
X_train=st_x.fit_transform(X_train)    
X_test=st_x.transform(X_test)

