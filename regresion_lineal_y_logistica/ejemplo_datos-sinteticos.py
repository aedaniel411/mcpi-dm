import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
from scipy.stats import shapiro

np.random.seed(42)
n = 100
size = np.random.uniform(50, 250, n)
habitaciones = np.random.randint(1, 6, n)
edad = np.random.randint(1, 50, n)

precio = 50000 + (size * 300) + (habitaciones * 10000) - (edad * 500) + np.random.normal(0, 10000, n)

df = pd.DataFrame({'Tamaño': size,
                   'Habitaciones': habitaciones, 
                   'Edad': edad, 
                   'Precio': precio})

x = df[['Tamaño', 'Habitaciones', 'Edad']]
y = df['Precio']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

print(f'Intercepto: {modelo.intercept_:.2f}')
print(f'Coeficientes: {modelo.coef_}')

print(f'Error cuadrático medio (MSE): {mean_squared_error(y_test, y_pred):.2f}')
print(f'R^2: {r2_score(y_test, y_pred):.2f}')

# Supuesto de Linealidad
sns.pairplot(df, x_vars=['Tamaño', 'Habitaciones', 'Edad'], y_vars='Precio', height=4, aspect=1, kind='reg')
plt.show()

# No autocorrelación (independencia de los errores)
dw_stat = durbin_watson(y_test - y_pred)
print(f'Estadística de Durbin-Watson: {dw_stat:.2f}')

# Supuesto de Homocedasticidad
# Grafica de residuos vs valores predichos
residuos = y_test - y_pred
plt.scatter(y_pred, residuos)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Verificación de Homocedasticidad')
plt.show()

x_test_const = sm.add_constant(x_test)
_, pval, _, _ = het_breuschpagan(residuos, x_test_const)
print(f'Valor p de Breusch-Pagan: {pval:.4f}')

# Supuesto de Normalidad de los Errores
sns.histplot(residuos, kde=True, bins=20)
plt.title('Distribución de errores')
plt.show()

# Prueba de Shapiro-Wilk
stat, pval = shapiro(residuos)
print(f'P-valor de Shapiro-Wilk: {pval:.4f}')

# Supuesto de No Multicolinealidad
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()