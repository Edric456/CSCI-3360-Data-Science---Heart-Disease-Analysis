import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


#Loading Data
pd.set_option("display.width", 3000)
pd.set_option('display.max_rows', 1000)
data = pd.read_csv("heart.csv")
print(data)


def KMeans_Clustering(dataframe, var1, var2):
    df = dataframe[[var1, var2]]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
    sns.scatterplot(data=df, x=var1, y=var2, hue=kmeans.labels_)
    plt.title("" + var1 + " & " + var2)
    plt.savefig("./" + var1 + "-" + var2 + ".png")
    plt.close()


KMeans_Clustering(data, "target", "cp")
KMeans_Clustering(data, "cp", "target")
KMeans_Clustering(data, "target", "thalach")
KMeans_Clustering(data, "target", "slope")
KMeans_Clustering(data, "thalach", "cp")
KMeans_Clustering(data, "thalach", "slope")
#KMeans_Clustering(data, "target", "age")






