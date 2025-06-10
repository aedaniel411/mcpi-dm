import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Datos
clientes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
datos = np.array([[10, 500], [15, 700], [3, 100], [5, 200], [7, 250], [20, 1200], [25, 1500]])

plt.figure(figsize=(8, 5))
sch.dendrogram(sch.linkage(datos, method='complete'), labels=clientes)
plt.title('Dendrograma para segmentación de clientes')
plt.xlabel('Clientes')
# plt.xticks(rotation=45)
plt.ylabel('Distancias')
plt.show()

cj = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
clusters = cj.fit_predict(datos)

for cliente, cluster in zip(clientes, clusters):
    print (f'Clusters {cliente} -> cluster {cluster}')