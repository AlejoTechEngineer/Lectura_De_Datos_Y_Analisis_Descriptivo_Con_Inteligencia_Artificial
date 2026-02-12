"""
Script 5: División de Datos - Entrenamiento y Validación
Bike Sharing Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Cargar datos
df = pd.read_csv('../bike+sharing+dataset/day.csv')
df['dteday'] = pd.to_datetime(df['dteday'])

print("="*70)
print("DIVISIÓN DE DATOS: ENTRENAMIENTO Y VALIDACIÓN")
print("="*70)

# ==========================================
# MÉTODO 1: DIVISIÓN TEMPORAL (RECOMENDADO PARA SERIES TEMPORALES)
# ==========================================

print("\n1. DIVISIÓN TEMPORAL 80-20")
print("-" * 70)
print("Este método es el MÁS APROPIADO para series temporales,")
print("ya que respeta el orden cronológico de los datos.\n")

# Calcular punto de corte
train_size = int(0.8 * len(df))

# Dividir datos
df_train_temporal = df.iloc[:train_size].copy()
df_test_temporal = df.iloc[train_size:].copy()

print(f"Total de registros: {len(df)}")
print(f"\nConjunto de ENTRENAMIENTO:")
print(f"  • Registros: {len(df_train_temporal)} ({len(df_train_temporal)/len(df)*100:.1f}%)")
print(f"  • Período:   {df_train_temporal['dteday'].min().strftime('%Y-%m-%d')} a {df_train_temporal['dteday'].max().strftime('%Y-%m-%d')}")
print(f"  • Media cnt: {df_train_temporal['cnt'].mean():.2f}")
print(f"  • Std cnt:   {df_train_temporal['cnt'].std():.2f}")

print(f"\nConjunto de VALIDACIÓN:")
print(f"  • Registros: {len(df_test_temporal)} ({len(df_test_temporal)/len(df)*100:.1f}%)")
print(f"  • Período:   {df_test_temporal['dteday'].min().strftime('%Y-%m-%d')} a {df_test_temporal['dteday'].max().strftime('%Y-%m-%d')}")
print(f"  • Media cnt: {df_test_temporal['cnt'].mean():.2f}")
print(f"  • Std cnt:   {df_test_temporal['cnt'].std():.2f}")

# Guardar archivos
df_train_temporal.to_csv('train_temporal.csv', index=False)
df_test_temporal.to_csv('test_temporal.csv', index=False)

print("\n✓ Archivos guardados:")
print("  • train_temporal.csv")
print("  • test_temporal.csv")

# ==========================================
# MÉTODO 2: DIVISIÓN ALEATORIA (ALTERNATIVO)
# ==========================================

print("\n" + "="*70)
print("2. DIVISIÓN ALEATORIA 80-20 (Alternativa - NO recomendada)")
print("-" * 70)
print("Este método NO respeta el orden temporal y puede causar data leakage.")
print("Solo se muestra con fines comparativos.\n")

# Preparar features y target
features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 
            'weathersit', 'temp', 'hum', 'windspeed']
target = 'cnt'

X = df[features]
y = df[target]

# División aleatoria
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Conjunto de ENTRENAMIENTO:")
print(f"  • Registros: {len(X_train_random)} ({len(X_train_random)/len(df)*100:.1f}%)")
print(f"  • Media cnt: {y_train_random.mean():.2f}")
print(f"  • Std cnt:   {y_train_random.std():.2f}")

print(f"\nConjunto de VALIDACIÓN:")
print(f"  • Registros: {len(X_test_random)} ({len(X_test_random)/len(df)*100:.1f}%)")
print(f"  • Media cnt: {y_test_random.mean():.2f}")
print(f"  • Std cnt:   {y_test_random.std():.2f}")

# ==========================================
# COMPARACIÓN DE DISTRIBUCIONES
# ==========================================

print("\n" + "="*70)
print("3. COMPARACIÓN DE DISTRIBUCIONES")
print("-" * 70)

# Estadísticas comparativas
print("\nEstadísticas de 'cnt' por conjunto:")
print("-" * 70)
print(f"{'Conjunto':<20} {'Media':>10} {'Mediana':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 70)
print(f"{'Dataset completo':<20} {df['cnt'].mean():>10.2f} {df['cnt'].median():>10.2f} {df['cnt'].std():>10.2f} {df['cnt'].min():>10.0f} {df['cnt'].max():>10.0f}")
print(f"{'Train (temporal)':<20} {df_train_temporal['cnt'].mean():>10.2f} {df_train_temporal['cnt'].median():>10.2f} {df_train_temporal['cnt'].std():>10.2f} {df_train_temporal['cnt'].min():>10.0f} {df_train_temporal['cnt'].max():>10.0f}")
print(f"{'Test (temporal)':<20} {df_test_temporal['cnt'].mean():>10.2f} {df_test_temporal['cnt'].median():>10.2f} {df_test_temporal['cnt'].std():>10.2f} {df_test_temporal['cnt'].min():>10.0f} {df_test_temporal['cnt'].max():>10.0f}")
print(f"{'Train (aleatorio)':<20} {y_train_random.mean():>10.2f} {y_train_random.median():>10.2f} {y_train_random.std():>10.2f} {y_train_random.min():>10.0f} {y_train_random.max():>10.0f}")
print(f"{'Test (aleatorio)':<20} {y_test_random.mean():>10.2f} {y_test_random.median():>10.2f} {y_test_random.std():>10.2f} {y_test_random.min():>10.0f} {y_test_random.max():>10.0f}")

# ==========================================
# ANÁLISIS DE REPRESENTATIVIDAD
# ==========================================

print("\n" + "="*70)
print("4. ANÁLISIS DE REPRESENTATIVIDAD")
print("-" * 70)

# Verificar distribución de variables categóricas
print("\nDistribución de ESTACIONES:")
print("-" * 70)
season_map = {1: 'Primavera', 2: 'Verano', 3: 'Otoño', 4: 'Invierno'}

print(f"{'Estación':<15} {'Completo':>10} {'Train(T)':>10} {'Test(T)':>10} {'Train(A)':>10} {'Test(A)':>10}")
print("-" * 70)
for season in [1, 2, 3, 4]:
    pct_full = (df['season'] == season).sum() / len(df) * 100
    pct_train_t = (df_train_temporal['season'] == season).sum() / len(df_train_temporal) * 100
    pct_test_t = (df_test_temporal['season'] == season).sum() / len(df_test_temporal) * 100
    pct_train_a = (X_train_random['season'] == season).sum() / len(X_train_random) * 100
    pct_test_a = (X_test_random['season'] == season).sum() / len(X_test_random) * 100
    
    print(f"{season_map[season]:<15} {pct_full:>9.1f}% {pct_train_t:>9.1f}% {pct_test_t:>9.1f}% {pct_train_a:>9.1f}% {pct_test_a:>9.1f}%")

print("\nDistribución de AÑOS:")
print("-" * 70)
print(f"{'Año':<15} {'Completo':>10} {'Train(T)':>10} {'Test(T)':>10} {'Train(A)':>10} {'Test(A)':>10}")
print("-" * 70)
for year in [0, 1]:
    year_label = '2011' if year == 0 else '2012'
    pct_full = (df['yr'] == year).sum() / len(df) * 100
    pct_train_t = (df_train_temporal['yr'] == year).sum() / len(df_train_temporal) * 100
    pct_test_t = (df_test_temporal['yr'] == year).sum() / len(df_test_temporal) * 100
    pct_train_a = (X_train_random['yr'] == year).sum() / len(X_train_random) * 100
    pct_test_a = (X_test_random['yr'] == year).sum() / len(X_test_random) * 100
    
    print(f"{year_label:<15} {pct_full:>9.1f}% {pct_train_t:>9.1f}% {pct_test_t:>9.1f}% {pct_train_a:>9.1f}% {pct_test_a:>9.1f}%")

# ==========================================
# VISUALIZACIONES
# ==========================================

print("\n5. GENERANDO VISUALIZACIONES...")
print("-" * 70)

fig = plt.figure(figsize=(18, 10))

# 1. Serie temporal con división
ax1 = plt.subplot(2, 3, 1)
plt.plot(df_train_temporal['dteday'], df_train_temporal['cnt'], 
         label='Entrenamiento', color='blue', alpha=0.7, linewidth=1)
plt.plot(df_test_temporal['dteday'], df_test_temporal['cnt'], 
         label='Validación', color='red', alpha=0.7, linewidth=1)
plt.axvline(df_test_temporal['dteday'].min(), color='black', linestyle='--', 
            linewidth=2, label='Punto de corte')
plt.xlabel('Fecha')
plt.ylabel('Número de alquileres')
plt.title('División Temporal 80-20', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 2. Distribuciones comparativas
ax2 = plt.subplot(2, 3, 2)
plt.hist(df_train_temporal['cnt'], bins=30, alpha=0.5, label='Train (temporal)', 
         color='blue', edgecolor='black', density=True)
plt.hist(df_test_temporal['cnt'], bins=30, alpha=0.5, label='Test (temporal)', 
         color='red', edgecolor='black', density=True)
plt.xlabel('Número de alquileres')
plt.ylabel('Densidad')
plt.title('Distribución: Train vs Test (Temporal)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Boxplot comparativo
ax3 = plt.subplot(2, 3, 3)
data_boxplot = [df_train_temporal['cnt'], df_test_temporal['cnt']]
bp = plt.boxplot(
    data_boxplot,
    tick_labels=['Train\n(temporal)', 'Test\n(temporal)'],
    patch_artist=True,
    widths=0.6
)

bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')

plt.ylabel('Número de alquileres')
plt.title('Comparación de Distribuciones', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 4. Distribución de temperaturas
ax4 = plt.subplot(2, 3, 4)
plt.hist(df_train_temporal['temp'], bins=30, alpha=0.5, label='Train', 
         color='blue', edgecolor='black', density=True)
plt.hist(df_test_temporal['temp'], bins=30, alpha=0.5, label='Test', 
         color='red', edgecolor='black', density=True)
plt.xlabel('Temperatura normalizada')
plt.ylabel('Densidad')
plt.title('Distribución de Temperatura', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Estaciones por conjunto
ax5 = plt.subplot(2, 3, 5)
season_dist_train = df_train_temporal['season'].value_counts().sort_index()
season_dist_test = df_test_temporal['season'].value_counts().sort_index()

# Asegurar que ambas tengan las 4 estaciones
season_dist_train = season_dist_train.reindex([1,2,3,4], fill_value=0)
season_dist_test = season_dist_test.reindex([1,2,3,4], fill_value=0)

x = np.arange(4)
width = 0.35
plt.bar(x - width/2, season_dist_train.values, width, label='Train', 
        color='blue', alpha=0.7, edgecolor='black')
plt.bar(x + width/2, season_dist_test.values, width, label='Test', 
        color='red', alpha=0.7, edgecolor='black')
plt.xlabel('Estación')
plt.ylabel('Número de registros')
plt.title('Distribución de Estaciones', fontweight='bold')
plt.xticks(x, ['Primavera', 'Verano', 'Otoño', 'Invierno'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 6. Evolución mensual
ax6 = plt.subplot(2, 3, 6)
monthly_train = df_train_temporal.groupby('mnth')['cnt'].mean()
monthly_test = df_test_temporal.groupby('mnth')['cnt'].mean()
months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
          'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
plt.plot(monthly_train.index, monthly_train.values, marker='o', 
         label='Train', linewidth=2, markersize=8, color='blue')
plt.plot(monthly_test.index, monthly_test.values, marker='s', 
         label='Test', linewidth=2, markersize=8, color='red')
plt.xlabel('Mes')
plt.ylabel('Promedio de alquileres')
plt.title('Promedio Mensual por Conjunto', fontweight='bold')
plt.xticks(range(1, 13), months, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_division_datos.png', dpi=300, bbox_inches='tight')
print("✓ Visualizaciones guardadas en '05_division_datos.png'")

# ==========================================
# GUARDAR INFORMACIÓN DE LA DIVISIÓN
# ==========================================

division_info = {
    'Método': ['Temporal', 'Temporal', 'Aleatorio', 'Aleatorio'],
    'Conjunto': ['Train', 'Test', 'Train', 'Test'],
    'Registros': [len(df_train_temporal), len(df_test_temporal), 
                  len(X_train_random), len(X_test_random)],
    'Media_cnt': [df_train_temporal['cnt'].mean(), df_test_temporal['cnt'].mean(),
                  y_train_random.mean(), y_test_random.mean()],
    'Std_cnt': [df_train_temporal['cnt'].std(), df_test_temporal['cnt'].std(),
                y_train_random.std(), y_test_random.std()]
}

df_division_info = pd.DataFrame(division_info)
df_division_info.to_csv('05_division_info.csv', index=False)
print("✓ Información de división guardada en '05_division_info.csv'")

# ==========================================
# RESUMEN EJECUTIVO
# ==========================================

print("\n" + "="*70)
print("RESUMEN EJECUTIVO - DIVISIÓN DE DATOS")
print("="*70)

print(f"""
MÉTODO RECOMENDADO: División Temporal 80-20

JUSTIFICACIÓN:
• Los datos son una serie temporal (2011-2012)
• La división aleatoria causaría DATA LEAKAGE
• Queremos predecir el FUTURO, no interpolar datos pasados
• Respeta la naturaleza temporal del problema

CARACTERÍSTICAS DE LOS CONJUNTOS:

ENTRENAMIENTO (80%):
• Período: {df_train_temporal['dteday'].min().strftime('%Y-%m-%d')} a {df_train_temporal['dteday'].max().strftime('%Y-%m-%d')}
• {len(df_train_temporal)} registros
• Media: {df_train_temporal['cnt'].mean():.0f} alquileres/día
• Incluye mayormente datos de 2011 y primera mitad de 2012

VALIDACIÓN (20%):
• Período: {df_test_temporal['dteday'].min().strftime('%Y-%m-%d')} a {df_test_temporal['dteday'].max().strftime('%Y-%m-%d')}
• {len(df_test_temporal)} registros
• Media: {df_test_temporal['cnt'].mean():.0f} alquileres/día
• Últimos ~5 meses de 2012

CONSIDERACIONES:
• El conjunto de test tiene media más alta (tendencia creciente)
• Esto es ESPERADO en series temporales con tendencia
• El modelo debe generalizar a valores más altos
• Validación cruzada temporal (rolling window) puede usarse adicionalmente

ARCHIVOS GENERADOS:
✓ train_temporal.csv  - Conjunto de entrenamiento
✓ test_temporal.csv   - Conjunto de validación
✓ 05_division_info.csv - Resumen de la división
""")

print("="*70)
