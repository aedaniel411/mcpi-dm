import numpy as np
import pandas as pd
import joblib, io, base64, os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar el dataset
def funcion_logica_regresion_logistica():
    ruta_actual = os.path.dirname(__file__)
    
    ######## en esta sección se cargan los datos desde la función  y se guardan en un csv
    # data = ©()
    # df = pd.DataFrame(data.data, columns=data.feature_names)
    # df['target'] = data.target
    # file_csv = os.path.join(ruta_actual, 'datos_cancer.csv')
    # df.to_csv(file_csv, index=False)
    # clases = data.target_names
    # X = pd.DataFrame(data.data, columns=data.feature_names)
    # y = pd.Series(data.target)  # 0 = maligno, 1 = benigno

    try:
        df = pd.read_csv(f'{ruta_actual}/datos_cancer.csv')
    except:
        return {'error': 'No se encuentra el archivo de entrenamiento'}

    X = df.drop(columns=['target'])
    y = df['target']

    clases = ('maligno', 'beningo')
    shape = X.shape

    # print("Clases:", 'data.target_names')
    # print("Shape X:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LogisticRegression(max_iter=10000)
    modelo.fit(X_train, y_train)

    joblib.dump(modelo, f'{ruta_actual}/modelo_regresion_logistica.pkl')

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1

    # Reporte general
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    reporte['0']['f1score'] = reporte['0']['f1-score']
    reporte['1']['f1score'] = reporte['1']['f1-score']
    reporte['macroavg'] = reporte['macro avg']
    reporte['weightedavg'] = reporte['weighted avg']
    reporte['macroavg']['f1score'] = reporte['macroavg']['f1-score']
    reporte['weightedavg']['f1score'] = reporte['weightedavg']['f1-score']
    # print(classification_report(y_test, y_pred))
    # print(reporte)

    matplotlib.use('Agg')

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_matriz_confusion= base64.b64encode(img.getvalue()).decode()
    plt.close(plt.gcf()) # Cierra la figura actual.

    resultados = {
        'clases': clases,
        'shape': shape,
        'accuracy': accuracy,
        'reporte': reporte,
        'plot_matriz_confusion': plot_matriz_confusion
    }
    return resultados

    # fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    # roc_auc = roc_auc_score(y_test, y_proba)

    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    # plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    # plt.xlabel("Tasa de falsos positivos (FPR)")
    # plt.ylabel("Tasa de verdaderos positivos (TPR)")
    # plt.title("Curva ROC")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
