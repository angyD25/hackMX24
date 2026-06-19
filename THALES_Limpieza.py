# --- Celda 1 ---
# Install pycaret
!pip install pycaret
import pycaret

# --- Celda 2 ---
import pandas as pd

# Get data (note: this is only the 2024 data)
df = pd.read_csv('https://archivo.datos.cdmx.gob.mx/FGJ/carpetas/carpetasFGJ_2024.csv')
df.head()

# --- Celda 3 ---
import pandas as pd

# Obtener los datos
df = pd.read_csv('https://archivo.datos.cdmx.gob.mx/FGJ/carpetas/carpetasFGJ_2024.csv')

# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Comprobar valores nulos (NA, Nulls)
print("\nValores faltantes antes de la imputación:")
print(df.isnull().sum())

# Imputar valores nulos en columnas numéricas con el promedio
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Para columnas de tipo objeto, usar el modo (más frecuente)
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Comprobar valores nulos después de la imputación
print("\nValores faltantes después de la imputación:")
print(df.isnull().sum())

# Comprobar filas duplicadas
duplicate_rows = df[df.duplicated()]
print("\nFilas duplicadas:")
print(duplicate_rows)

# Comprobar filas sin datos (todas las columnas son NA o vacías)
empty_rows = df[df.isnull().all(axis=1)]
print("\nFilas sin datos:")
print(empty_rows)

# Eliminar filas duplicadas
cleaned_df = df.drop_duplicates()

# Mostrar el DataFrame limpio en un formato de tabla
print("\nDataFrame limpio:")
print(cleaned_df)


# --- Celda 4 ---
# prompt: para la dataframe limpia solo quedate con las variables: mes_hecho, anio_hecho y fecha_hecho, hora_hecho, delito, categoria_delito, alcaldia_hecho, colonia_hecho, municipio_hecho, latitud y longitud

# Keep only the specified columns
selected_columns = ['mes_hecho', 'anio_hecho', 'fecha_hecho', 'hora_hecho', 'delito', 'categoria_delito', 'alcaldia_hecho', 'colonia_hecho', 'municipio_hecho', 'latitud', 'longitud']
cleaned_df = cleaned_df[selected_columns]

# Display the cleaned DataFrame
print("\nDataFrame limpio con columnas seleccionadas:")
cleaned_df.to_csv("aggregated.csv",index=False)

# --- Celda 5 ---
import pandas as pd

# Obtener los datos desde un archivo CSV disponible en línea
df = pd.read_csv('https://archivo.datos.cdmx.gob.mx/FGJ/carpetas/carpetasFGJ_2024.csv')

# Seleccionar solo las columnas especificadas para nuestro analisis en la interfaz antes de limpiar los datos
selected_columns = ['mes_hecho', 'anio_hecho', 'fecha_hecho', 'hora_hecho', 'delito',
                   'categoria_delito', 'alcaldia_hecho', 'colonia_hecho',
                   'municipio_hecho', 'latitud', 'longitud']
df = df[selected_columns]

# Comprobar valores nulos (NA, Nulls) en el DataFrame
print("\nValores faltantes antes de la imputación:")
print(df.isnull().sum())

# Imputar valores nulos en columnas numéricas con el promedio de cada columna, ya que si
#eliminamos estos valores perderiamos datos por lo que es mejor encontrar un promedio
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)  # Rellena con la media

# Para columnas de tipo objeto, usar el modo (el valor más frecuente)
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)  # Rellena con el modo

# Comprobar valores nulos después de la imputación
print("\nValores faltantes después de la imputación:")
print(df.isnull().sum())

# Comprobar filas duplicadas en el DataFrame
duplicate_rows = df[df.duplicated()]
print("\nFilas duplicadas:")
print(duplicate_rows)

# Comprobar filas sin datos (todas las columnas son NA o vacías)
empty_rows = df[df.isnull().all(axis=1)]
print("\nFilas sin datos:")
print(empty_rows)

# Eliminar filas duplicadas del DataFrame
cleaned_df = df.drop_duplicates()

# Convertir 'fecha_hecho' a formato datetime (ignorando la hora en este ejemplo)
# Manejar errores potenciales de manera controlada
cleaned_df['fecha_hecho'] = pd.to_datetime(cleaned_df['fecha_hecho'], format='%Y-%m-%d', errors='coerce')

# Ordenar el DataFrame y establecer 'fecha_hecho' como índice después de la conversión a datetime
cleaned_df = cleaned_df.sort_values('fecha_hecho').set_index('fecha_hecho')

# Mostrar el DataFrame limpio en un formato de tabla
print("\nDataFrame limpio:")
print(cleaned_df)

# --- Celda 6 ---
# Agrupar por 'alcaldia_hecho' y contar el número de delitos
# Utilizamos el método groupby para agrupar los datos por la columna 'alcaldia_hecho'
# Luego contamos el número de delitos en cada alcaldía utilizando la columna 'delito'
crime_counts = cleaned_df.groupby('alcaldia_hecho')['delito'].count().reset_index()

# Renombrar la columna 'delito' a 'numero_delitos'
# Esto facilita la comprensión de la tabla resultante al indicar claramente que se trata del número de delitos
crime_counts = crime_counts.rename(columns={'delito': 'numero_delitos'})

# Mostrar la tabla resultante
# Imprimimos la tabla que muestra el número de delitos por cada alcaldía
print("\nTabla de alcaldías y número de delitos:")
crime_counts

# --- Celda 7 ---
# prompt: dame un dataframe (grouphy) agrupado por alcaldía_hecho y que cuente el numero de delito por cada alcaldía_hecho

# Group by 'alcaldia_hecho' and count the number of crimes
delitos_por_alcaldia = cleaned_df.groupby('alcaldia_hecho')['delito'].count().reset_index()

# Rename the 'delito' column to 'numero_delitos'
delitos_por_alcaldia = delitos_por_alcaldia.rename(columns={'delito': 'numero_delitos'})

# Display the resulting DataFrame
delitos_por_alcaldia

# --- Celda 8 ---
cleaned_df.info()

# --- Celda 9 ---
cleaned_df.dtypes