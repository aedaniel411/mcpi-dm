from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
banknote_authentication = fetch_ucirepo(id=267) 
  
# data (as pandas dataframes) 
X = banknote_authentication.data.features 
y = banknote_authentication.data.targets 
  
# metadata 
print(banknote_authentication.metadata) 
  
# variable information 
print(banknote_authentication.variables) 

#------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D

# Verificar valores nulos en las características (features)
print("Valores nulos en X:")
print(X.isnull().sum())

# Verificar valores nulos en la variable objetivo (target)
print("\nValores nulos en y:")
print(y.isnull().sum())

# Renombrar columnas si es necesario
df = X.copy()
df['class'] = y['class']
df['class'] = df['class'].astype('category')

# Resumen del dataset
print(df.info())
print(df.describe())
print(df['class'].value_counts())

# Visualizaciones
df['class'].value_counts().plot(kind='bar', title='Distribución de clases')
plt.show()

for col in ['variance', 'entropy', 'skewness', 'curtosis']:
    plt.hist(df[col], bins=20, edgecolor='black')
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

    plt.boxplot(df[col])
    plt.title(f'Boxplot de {col}')
    plt.show()

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('class', axis=1), df['class'],
    test_size=0.3, random_state=123456, stratify=df['class']
)

# Entrenar modelo SVM
modelo = SVC(C=5, kernel='rbf')  # rbf por defecto
modelo.fit(X_train, y_train)

# Predicciones y matrices de confusión
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

print("Matriz de confusión (entrenamiento):")
print(confusion_matrix(y_train, y_pred_train))
print("Matriz de confusión (prueba):")
print(confusion_matrix(y_test, y_pred_test))

# Graficar con dos variables (entrenamiento)
plot_data_train = X_train.copy()
plot_data_train['Clase'] = y_train.values
sns.scatterplot(data=plot_data_train, x='skewness', y='variance', hue='Clase')
plt.title("SVM en entrenamiento: skewness vs variance")
plt.show()

# Graficar con dos variables (prueba)
plot_data_test = X_test.copy()
plot_data_test['Clase'] = y_test.values
sns.scatterplot(data=plot_data_test, x='entropy', y='curtosis', hue='Clase')
plt.title("SVM en prueba: entropy vs curtosis")
plt.show()

# Gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['skewness'], X_train['variance'], X_train['curtosis'], c=y_train.cat.codes, cmap='viridis')
ax.set_xlabel('Skewness')
ax.set_ylabel('Variance')
ax.set_zlabel('Curtosis')
plt.title("Gráfico 3D de entrenamiento")
plt.show()

# Búsqueda de hiperparámetros
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100, 150, 200]}
#grid = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
#grid.fit(df.drop('class', axis=1), df['class'])

# Resultados del tuning
#print("Mejor parámetro encontrado:", grid.best_params_)
#print("Mejor modelo:", grid.best_estimator_)

# Mostrar los resultados del tuning
#results = pd.DataFrame(grid.cv_results_)

# import ace_tools as tools 
# tools.display_dataframe_to_user(name="Resultados tuning SVM", dataframe=results[['param_C', 'mean_test_score']])
#print("Resultados tuning SVM:")
#print(results[['param_C', 'mean_test_score']])