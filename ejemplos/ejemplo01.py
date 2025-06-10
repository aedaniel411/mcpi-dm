import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Datos
datos = np.array([[1, 2], [2, 3], [5, 8], [6, 9], [7, 10]])

plt.figure(figsize=(8, 5))
sch.dendrogram(sch.linkage(datos, method='ward'))
plt.title('Dendrograma de CJ')
plt.xlabel('Puntos de datos')
plt.ylabel('Distancias')
plt.show()

cj = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
clusters = cj.fit_predict(datos)

print ('Clusters:',clusters)