import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


# Datos
df  = pd.read_csv("Mall_Customers.csv")
print (df.head())
print (df.info())
datos = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

plt.figure(figsize=(8, 5))
sch.dendrogram(sch.linkage(datos, method='ward'))
plt.title('Dendrograma para segmentación de clientes')
plt.xlabel('Clientes')
plt.xticks(rotation=45)
plt.ylabel('Distancias')
plt.show()

cj = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
clusters = cj.fit_predict(datos)

df['cluster'] = clusters

print (df[['CustomerID', 'Annual Income (k$)', 'Spending Score (1-100)', 'cluster']].head(20))
