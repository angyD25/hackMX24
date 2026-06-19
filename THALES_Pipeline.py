# --- Celda 1 ---
# region IMPORTACIÓN DE LIBRERÍAS 
import os
import warnings
import logging

# Desactivar oneDNN para reducir mensajes innecesarios de TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings('ignore')  # Ignorar todas las advertencias

# Establecer el nivel de logging en CRITICAL para evitar mensajes innecesarios
logging.basicConfig(level=logging.CRITICAL)

import time
import requests
from io import StringIO
import pandas as pd
import tensorflow as tf
from pycaret.classification import * # Asegúrate de que estamos importando desde pycaret.classification
import matplotlib.pyplot as plt
#endregion

# region URL del archivo CSV
url = 'https://archivo.datos.cdmx.gob.mx/FGJ/carpetas/carpetasFGJ_2024.csv'
#endregion

# region Descargar archivo con reintentos
for i in range(3):  # Intentar 3 veces
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        print("Archivo descargado con éxito")
        break
    except requests.exceptions.RequestException as e:
        print(f"Intento {i+1} fallido: {e}")
else:
    print("Error: No se pudo descargar el archivo después de varios intentos")
    exit()
#endregion

# region Visualización inicial
print("Primeras filas del conjunto de datos:")
print(df.head())
print("\nInformación del conjunto de datos:")
print(df.info())
print("\nDescripción estadística de los datos numéricos:")
print(df.describe())
#endregion

# region Manejo de valores nulos con promedios según categoría
print("\nValores nulos por columna antes de la limpieza:")
print(df.isnull().sum())

# Rellenar valores nulos en columnas numéricas (latitud, longitud) con el promedio por categoria_delito
df['latitud'] = df.groupby('categoria_delito')['latitud'].transform(lambda x: x.fillna(x.mean()))
df['longitud'] = df.groupby('categoria_delito')['longitud'].transform(lambda x: x.fillna(x.mean()))

# Rellenar otros valores faltantes con opciones relevantes
df['colonia_hecho'].fillna('desconocido', inplace=True)
df['alcaldia_hecho'].fillna('desconocido', inplace=True)
df['municipio_hecho'].fillna('desconocido', inplace=True)

# Para columnas de tiempo, asignar "00:00:00" en valores nulos
df['hora_inicio'].fillna("00:00:00", inplace=True)
df['hora_hecho'].fillna("00:00:00", inplace=True)

# Para columna de año y mes, rellenar con valores más frecuentes
df['anio_hecho'].fillna(df['anio_hecho'].mode()[0], inplace=True)
df['mes_hecho'].fillna(df['mes_hecho'].mode()[0], inplace=True)
df['fecha_hecho'].fillna(method='ffill', inplace=True)  # Usar 'ffill' para rellenar fechas consecutivas
#endregion

# region Seleccionar columnas de interés
columns_of_interest = ['mes_hecho', 'anio_hecho', 'fecha_hecho', 'hora_hecho', 
                       'categoria_delito', 'alcaldia_hecho', 
                       'colonia_hecho', 'municipio_hecho']
df = df[columns_of_interest]
#endregion

# region Revisar y convertir tipos de datos
df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'], errors='coerce')
df['categoria_delito'] = df['categoria_delito'].astype('category')
df['alcaldia_hecho'] = df['alcaldia_hecho'].astype('category')
#endregion

# region Normalización y Filtrado
# Convertir texto a minúsculas y eliminar espacios en blanco adicionales
df['categoria_delito'] = df['categoria_delito'].str.lower().str.strip()
df['alcaldia_hecho'] = df['alcaldia_hecho'].str.lower().str.strip()
df['colonia_hecho'] = df['colonia_hecho'].str.lower().str.strip()
df['municipio_hecho'] = df['municipio_hecho'].str.lower().str.strip()
#endregion

# region Eliminar duplicados
df.drop_duplicates(inplace=True)
#endregion

# region Mostrar resumen de columnas clave y valores nulos
print("\nResumen de columnas clave y valores nulos:")
print(df.info())
print("\nValores nulos restantes en las columnas clave después de la limpieza:")
print(df.isnull().sum())

# Mostrar la cantidad de datos en la base de datos limpia (solo las columnas de interés)
rows, columns = df.shape
print(f"\nCantidad de datos en la base de datos limpia (solo columnas de interés): {rows} filas y {columns} columnas.")
#endregion

# region Conteo de tipos de delito
print("\nConteo de tipos de delito en 'categoria_delito':")
delito_counts = df['categoria_delito'].value_counts()
print(delito_counts)
#endregion

# region Configuración de clasificación con PyCaret
# Seleccionar solo las columnas necesarias para la clasificación
data_c = df[['fecha_hecho', 'hora_hecho', 'categoria_delito', 'alcaldia_hecho']].copy()

# Combinar 'fecha_hecho' con 'hora_hecho' en una sola columna y convertirla en datetime
data_c['fecha_hecho'] = data_c['fecha_hecho'].astype(str).str.cat(data_c['hora_hecho'].astype(str), sep=',')
data_c.drop(['hora_hecho'], axis=1, inplace=True)
data_c['fecha_hecho'] = pd.to_datetime(data_c['fecha_hecho'], format='%Y-%m-%d,%H:%M:%S', errors='coerce')

# Crear columnas adicionales para día, mes, año y hora
data_c['day'] = data_c['fecha_hecho'].dt.day
data_c['month'] = data_c['fecha_hecho'].dt.month
data_c['year'] = data_c['fecha_hecho'].dt.year
data_c['hour'] = data_c['fecha_hecho'].dt.hour

# Asegurarse de que no hay valores nulos en las columnas críticas para PyCaret
data_c.dropna(subset=['fecha_hecho', 'categoria_delito', 'alcaldia_hecho'], inplace=True)

# Configuración de PyCaret para la tarea de clasificación
print("\nConfiguración de PyCaret para clasificación:")
s_c = setup(data=data_c, 
            target='categoria_delito',  # Cambiar a 'categoria_delito' en lugar de 'delito'
            session_id=123,  # Para reproducibilidad
            verbose=False)  # Sin salida detallada en la consola

# Comparar modelos de clasificación y seleccionar el mejor según F1-score
print("\nComparando modelos...")
best_c = compare_models(sort='F1')

# Visualizar rendimiento del modelo seleccionado en el conjunto de validación (holdout)
print("\nGraficando el rendimiento del mejor modelo:")
plot_model(best_c, plot='confusion_matrix')


# --- Celda 2 ---
# Optimización de hiperparámetros del mejor modelo
print("\nOptimizando hiperparámetros del mejor modelo:")
best_c_tuned = tune_model(best_c, optimize='F1')

# Visualizar rendimiento del modelo optimizado
print("\nGraficando el rendimiento del modelo optimizado:")
plot_model(best_c_tuned, plot='confusion_matrix')

# Evaluación final del modelo con predicciones en el conjunto de retención
print("\nPredicciones del modelo en el conjunto de retención:")
predictions = predict_model(best_c_tuned)
print(predictions.head())


# --- Celda 3 ---
# Guardar el modelo
save_model(best_c_tuned, 'modelo_criminalidad')

# Cargar el modelo
modelo = load_model('modelo_criminalidad')


# --- Celda 4 ---
# region URL del archivo CSV para datos de prueba 2023
url_test = 'https://archivo.datos.cdmx.gob.mx/FGJ/carpetas/carpetasFGJ_2023.csv'
#endregion

# region Descargar archivo con reintentos para 2023
for i in range(3):  # Intentar 3 veces
    try:
        response = requests.get(url_test, timeout=10)
        response.raise_for_status()
        df_test = pd.read_csv(StringIO(response.text))
        print("Archivo de prueba 2023 descargado con éxito")
        break
    except requests.exceptions.RequestException as e:
        print(f"Intento {i+1} fallido: {e}")
else:
    print("Error: No se pudo descargar el archivo después de varios intentos")
    exit()
#endregion

# region Visualización inicial para 2023
print("Primeras filas del conjunto de datos de prueba:")
print(df_test.head())
print("\nInformación del conjunto de datos de prueba:")
print(df_test.info())
print("\nDescripción estadística de los datos numéricos de prueba:")
print(df_test.describe())
#endregion

# region Manejo de valores nulos con promedios según categoría
print("\nValores nulos por columna antes de la limpieza:")
print(df_test.isnull().sum())

# Rellenar valores nulos en columnas numéricas (latitud, longitud) con el promedio por categoria_delito
df_test['latitud'] = df_test.groupby('categoria_delito')['latitud'].transform(lambda x: x.fillna(x.mean()))
df_test['longitud'] = df_test.groupby('categoria_delito')['longitud'].transform(lambda x: x.fillna(x.mean()))

# Rellenar otros valores faltantes con opciones relevantes
df_test['colonia_hecho'].fillna('desconocido', inplace=True)
df_test['alcaldia_hecho'].fillna('desconocido', inplace=True)
df_test['municipio_hecho'].fillna('desconocido', inplace=True)

# Para columnas de tiempo, asignar "00:00:00" en valores nulos
df_test['hora_inicio'].fillna("00:00:00", inplace=True)
df_test['hora_hecho'].fillna("00:00:00", inplace=True)

# Para columna de año y mes, rellenar con valores más frecuentes
df_test['anio_hecho'].fillna(df_test['anio_hecho'].mode()[0], inplace=True)
df_test['mes_hecho'].fillna(df_test['mes_hecho'].mode()[0], inplace=True)
df_test['fecha_hecho'].fillna(method='ffill', inplace=True)  # Usar 'ffill' para rellenar fechas consecutivas
#endregion

# region Seleccionar columnas de interés para 2023
columns_of_interest_test = ['mes_hecho', 'anio_hecho', 'fecha_hecho', 'hora_hecho', 
                            'categoria_delito', 'alcaldia_hecho', 
                            'colonia_hecho', 'municipio_hecho']
df_test = df_test[columns_of_interest_test]
#endregion

# region Revisar y convertir tipos de datos para 2023
df_test['fecha_hecho'] = pd.to_datetime(df_test['fecha_hecho'], errors='coerce')
df_test['categoria_delito'] = df_test['categoria_delito'].astype('category')
df_test['alcaldia_hecho'] = df_test['alcaldia_hecho'].astype('category')
#endregion

# region Normalización y Filtrado para 2023
# Convertir texto a minúsculas y eliminar espacios en blanco adicionales
df_test['categoria_delito'] = df_test['categoria_delito'].str.lower().str.strip()
df_test['alcaldia_hecho'] = df_test['alcaldia_hecho'].str.lower().str.strip()
df_test['colonia_hecho'] = df_test['colonia_hecho'].str.lower().str.strip()
df_test['municipio_hecho'] = df_test['municipio_hecho'].str.lower().str.strip()
#endregion

# region Eliminar duplicados en 2023
df_test.drop_duplicates(inplace=True)
#endregion

# region Mostrar resumen de columnas clave y valores nulos para 2023
print("\nResumen de columnas clave y valores nulos:")
print(df_test.info())
print("\nValores nulos restantes en las columnas clave después de la limpieza:")
print(df_test.isnull().sum())

# Mostrar la cantidad de datos en la base de datos limpia (solo las columnas de interés)
rows_test, columns_test = df_test.shape
print(f"\nCantidad de datos en la base de datos limpia (solo columnas de interés): {rows_test} filas y {columns_test} columnas.")
#endregion

# region Conteo de tipos de delito para 2023
print("\nConteo de tipos de delito en 'categoria_delito':")
delito_counts_test = df_test['categoria_delito'].value_counts()
print(delito_counts_test)
#endregion

# region Configuración de clasificación con PyCaret para datos de prueba 2023
# Seleccionar solo las columnas necesarias para la clasificación
data_test_c = df_test[['fecha_hecho', 'hora_hecho', 'categoria_delito', 'alcaldia_hecho']].copy()

# Combinar 'fecha_hecho' con 'hora_hecho' en una sola columna y convertirla en datetime
data_test_c['fecha_hecho'] = data_test_c['fecha_hecho'].astype(str).str.cat(data_test_c['hora_hecho'].astype(str), sep=',')
data_test_c.drop(['hora_hecho'], axis=1, inplace=True)
data_test_c['fecha_hecho'] = pd.to_datetime(data_test_c['fecha_hecho'], format='%Y-%m-%d,%H:%M:%S', errors='coerce')

# Crear columnas adicionales para día, mes, año y hora
data_test_c['day'] = data_test_c['fecha_hecho'].dt.day
data_test_c['month'] = data_test_c['fecha_hecho'].dt.month
data_test_c['year'] = data_test_c['fecha_hecho'].dt.year
data_test_c['hour'] = data_test_c['fecha_hecho'].dt.hour

# Asegurarse de que no hay valores nulos en las columnas críticas para PyCaret
data_test_c.dropna(subset=['fecha_hecho', 'categoria_delito', 'alcaldia_hecho'], inplace=True)

# Configuración de PyCaret para la tarea de clasificación

# --- Celda 5 ---
data_test_c

# --- Celda 6 ---
data_test=data_test_c.drop("categoria_delito",axis=1)
final_model=finalize_model(best_c_tuned)
predictions=predict_model(final_model,data_test_c)

# --- Celda 7 ---
predictions

# --- Celda 8 ---
actual=data_test_c["categoria_delito"]
comparison_df=pd.DataFrame({"acual":actual,"predicted":predictions["prediction_label"]})
print(comparison_df.head())

# --- Celda 9 ---
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calcula la matriz de confusión
actual = comparison_df["acual"]
predicted = comparison_df["predicted"]
conf_matrix = confusion_matrix(actual, predicted)

# Crea un heatmap para la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=sorted(actual.unique()), 
            yticklabels=sorted(actual.unique()))

# Etiquetas y título
plt.xlabel("Predicción")
plt.ylabel("Actual")
plt.title("Matriz de Confusión")
plt.show()