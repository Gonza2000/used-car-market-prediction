# ==========================================
# 6. PIPELINE DE PREPROCESAMIENTO Y MODELADO (CORREGIDO)
# ==========================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
# Code Aprobed by QA
# 1. Cargar y limpiar datos para obtener df_clean
if os.path.exists('vehicles.csv'):
    print("Cargando datos desde 'vehicles.csv'...")
    df = pd.read_csv('vehicles.csv', encoding='latin1', on_bad_lines='skip', low_memory=False)
else:
    print("El archivo 'vehicles.csv' no existe localmente. Intentando descargar de Kaggle...")
    try:
        os.environ['KAGGLE_API_TOKEN'] = "KGAT_a0fcdeed3125aab3d652c9bc808ee163"
        # Usamos la herramienta oficial para descargar el dataset
        import kaggle
        kaggle.api.dataset_download_files('austinreese/craigslist-carstrucks-data', path='.', unzip=True)
        print("¡Descarga completada con éxito!")
        df = pd.read_csv('vehicles.csv', encoding='latin1', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"Error al descargar de Kaggle: {e}")
        print("Por favor, coloca 'vehicles.csv' en este directorio y vuelve a intentarlo.")
        raise FileNotFoundError("vehicles.csv no encontrado.")

# Limpieza y filtrado según EDA
print("Preprocesando y limpiando el dataset...")
df_clean = df.drop_duplicates()
cols_irrelevantes = ['id', 'url', 'region_url', 'image_url', 'description',
                     'VIN', 'county', 'posting_date', 'lat', 'long']
df_clean = df_clean.drop(columns=[c for c in cols_irrelevantes if c in df_clean.columns])

# Filtrado comercial
df_clean = df_clean[(df_clean['price'] > 500) & (df_clean['price'] <= 150000)]
df_clean = df_clean[df_clean['odometer'].isna() | ((df_clean['odometer'] >= 1) & (df_clean['odometer'] <= 500000))]
df_clean = df_clean[df_clean['year'].isna() | ((df_clean['year'] >= 1980) & (df_clean['year'] <= 2027))]

# ==========================================
# PIPELINE RIDGE REGRESSION
# ==========================================

# 1. Separación inicial
X = df_clean.drop('price', axis=1)
y = df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Identificación automática de tipos de columnas
columnas_numericas = X_train.select_dtypes(include=['int64', 'float64']).columns
columnas_categoricas = X_train.select_dtypes(include=['object', 'category']).columns

# 3. Creación de sub-pipelines para procesar nulos ANTES de transformar
# Para números: Rellenar nulos con la mediana -> Estandarizar
pipeline_numerico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='median')),
    ('escalador', StandardScaler())
])

# Para texto: Rellenar nulos con 'unknown' -> Codificar a 0 y 1
pipeline_categorico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('codificador', OneHotEncoder(handle_unknown='ignore'))
])

# 4. Empaquetado en un ColumnTransformer usando los nuevos sub-pipelines
preprocesador = ColumnTransformer(
    transformers=[
        ('num', pipeline_numerico, columnas_numericas),
        ('cat', pipeline_categorico, columnas_categoricas)
    ])

# 5. Creación del Pipeline completo
pipeline_ridge = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('modelo', Ridge(alpha=1.0))
])

# 6. ¡Entrenamiento!
print("Limpiando nulos, estandarizando, codificando y entrenando Ridge...")
pipeline_ridge.fit(X_train, y_train)
print("¡Entrenamiento completado sin errores!")
