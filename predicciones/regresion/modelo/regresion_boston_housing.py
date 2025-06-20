import numpy as np
import pandas as pd
import seaborn as sns
import joblib, os, io, base64
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm


# boston = fetch_openml(name='boston', version=1, as_frame=True)
# df = boston.frame
def regresion():
    dataset = os.path.join(os.path.dirname(__file__), "housing.csv")
    try:
        df = pd.read_csv(dataset)
    except:
        return {'error': 'No se encuentra el archivo de entrenamiento'}

    x = df[['RM', 'LSTAT', 'PTRATIO']]
    y = df['MEDV']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(x_train, y_train)

    joblib.dump(modelo, 'modelo_regresion.pkl')

    y_pred = modelo.predict(x_test)

    matplotlib.use('Agg')

    # Supuesto de Linealidad
    pair_plot = sns.pairplot(df, x_vars=['RM', 'LSTAT', 'PTRATIO'], y_vars=['MEDV'], kind='reg')
    img = io.BytesIO()
    pair_plot.savefig(img, format='png')
    img.seek(0)

    plot_linealidad = base64.b64encode(img.getvalue()).decode()
    plt.close(pair_plot.figure) 
    
    #  No autocorrelación (independencia de los errores)
    dw_stat = durbin_watson(y_pred -  y_test)

    # Grafica de residuos vs valores predichos (Homosceadastisidad)
    residuos = y_test - y_pred

    plt.scatter(y_pred, residuos)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Valores predichos')
    plt.ylabel('Residuos')
    plt.title('Verificación de Homosceadastisidad')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_residuos = base64.b64encode(img.getvalue()).decode()
    plt.close(plt.gcf()) # Cierra la figura actual.

    # Agrega la constante (intercepto)
    x_test_const = sm.add_constant(x_test)
    
    _, pval_bp, _, _ = het_breuschpagan(residuos, x_test_const)
    # ¿Por qué se necesita una constante?
    # Porque el test asume un modelo con intercepto. Si no se incluye, los resultados no son válidos estadísticamente.

    # Suspuesto de Normalidad de los errores
    sns.histplot(residuos, kde=True, bins=20)
    plt.title('Distribución de errores')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_normalidad_errores = base64.b64encode(img.getvalue()).decode()
    plt.close(plt.gcf()) # Cierra la figura actual.

    # Prueba Shapiro-Wilk
    stat, pval_sw=shapiro(residuos)

    # Suspuesto de No Colinealidad
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_matriz_correlacion = base64.b64encode(img.getvalue()).decode()
    plt.close(plt.gcf()) # Cierra la figura actual.


    resultados = {
        'intercepto': f'{modelo.intercept_:.2f}',
        'coeficientes': modelo.coef_,
        'mse': f'{mean_squared_error(y_test, y_pred):.2f}',
        'r_cuadrada': f'{r2_score(y_test, y_pred)}',
        'modelo': 'modelo_regresion.pkl',
        'plot_linealidad': plot_linealidad,
        'durbin_watson': f'{dw_stat:.2f}',
        'plot_residuos': plot_residuos,
        'breusch_pagan': f'{pval_bp:.6f}',
        'plot_normalidad_errores': plot_normalidad_errores,
        'shapiro_wilk': f'{pval_sw:.6f}',
        'plot_matriz_correlacion': plot_matriz_correlacion,

    }
    return resultados






