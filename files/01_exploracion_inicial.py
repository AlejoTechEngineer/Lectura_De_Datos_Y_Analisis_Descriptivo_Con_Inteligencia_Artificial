"""
Script 1: Carga y Exploración Inicial de Datos
Bike Sharing Dataset - Análisis Descriptivo
"""

import pandas as pd

# ==========================================
# CARGA DE DATOS
# ==========================================

# Cargar el dataset diario
df = pd.read_csv('../bike+sharing+dataset/day.csv')
df['dteday'] = pd.to_datetime(df['dteday'])

print("="*70)
print("1. INFORMACIÓN GENERAL DEL DATASET")
print("="*70)

# Dimensiones
print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Período: {df['dteday'].min()} a {df['dteday'].max()}")

# Información de tipos de datos
print("\n" + "="*70)
print("2. TIPOS DE DATOS Y ESTRUCTURA")
print("="*70)
print(df.info())

# Primeras filas
print("\n" + "="*70)
print("3. PRIMERAS FILAS DEL DATASET")
print("="*70)
print(df.head(10))

# Últimas filas
print("\n" + "="*70)
print("4. ÚLTIMAS FILAS DEL DATASET")
print("="*70)
print(df.tail(10))

# Nombres de columnas
print("\n" + "="*70)
print("5. VARIABLES DEL DATASET")
print("="*70)
print(f"Columnas: {list(df.columns)}")

# Descripción de variables categóricas
print("\n" + "="*70)
print("6. VALORES ÚNICOS EN VARIABLES CATEGÓRICAS")
print("="*70)
categorical_vars = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
for var in categorical_vars:
    print(f"\n{var}: {sorted(df[var].unique())}")
    print(f"  Cantidad de valores únicos: {df[var].nunique()}")

# ==========================================
# VERIFICACIÓN DE CALIDAD DE DATOS
# ==========================================

print("\n" + "="*70)
print("7. VERIFICACIÓN DE VALORES FALTANTES")
print("="*70)

# Contar valores faltantes
missing_values = df.isnull().sum()
print("\nValores faltantes por columna:")
print(missing_values)

# Porcentaje de valores faltantes
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPorcentaje de valores faltantes:")
print(missing_percentage)

print(f"\n{'✓' if missing_values.sum() == 0 else '✗'} Conclusión: {'NO hay valores faltantes' if missing_values.sum() == 0 else 'SÍ hay valores faltantes'}")

# Verificar duplicados
print("\n" + "="*70)
print("8. VERIFICACIÓN DE REGISTROS DUPLICADOS")
print("="*70)
duplicados = df.duplicated().sum()
print(f"Cantidad de filas duplicadas: {duplicados}")
print(f"{'✓' if duplicados == 0 else '✗'} Conclusión: {'NO hay duplicados' if duplicados == 0 else 'SÍ hay duplicados'}")

# Guardar dataset limpio
df.to_csv('bike_sharing_clean.csv', index=False)
print("\n✓ Dataset limpio guardado en 'bike_sharing_clean.csv'")
