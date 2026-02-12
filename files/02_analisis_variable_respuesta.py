"""
Script 2: Análisis Descriptivo de la Variable Respuesta 'cnt'
Bike Sharing Dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Cargar datos
df = pd.read_csv('../bike+sharing+dataset/day.csv')

print("="*70)
print("ANÁLISIS DESCRIPTIVO DE LA VARIABLE RESPUESTA: cnt")
print("="*70)

# ==========================================
# ESTADÍSTICAS BÁSICAS
# ==========================================

print("\n1. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS")
print("-" * 70)

# Estadísticas principales
print(f"Media:                  {df['cnt'].mean():.2f} alquileres/día")
print(f"Mediana:                {df['cnt'].median():.2f} alquileres/día")
print(f"Moda:                   {df['cnt'].mode().values[0]} alquileres/día")
print(f"Desviación estándar:    {df['cnt'].std():.2f}")
print(f"Varianza:               {df['cnt'].var():.2f}")
print(f"Mínimo:                 {df['cnt'].min()} alquileres/día")
print(f"Máximo:                 {df['cnt'].max()} alquileres/día")
print(f"Rango:                  {df['cnt'].max() - df['cnt'].min()}")

# Percentiles
print("\nPercentiles:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    print(f"  P{p}:  {df['cnt'].quantile(p/100):.0f}")

# ==========================================
# MEDIDAS DE DISPERSIÓN
# ==========================================

print("\n2. MEDIDAS DE DISPERSIÓN Y FORMA")
print("-" * 70)

# Coeficiente de variación
cv = (df['cnt'].std() / df['cnt'].mean()) * 100
print(f"Coeficiente de variación:  {cv:.2f}%")
print(f"  Interpretación: {'Alta variabilidad' if cv > 30 else 'Baja variabilidad'}")

# Asimetría (Skewness)
skewness = df['cnt'].skew()
print(f"\nAsimetría (Skewness):      {skewness:.3f}")
if skewness > 0.5:
    print(f"  Interpretación: Distribución asimétrica positiva (cola derecha)")
elif skewness < -0.5:
    print(f"  Interpretación: Distribución asimétrica negativa (cola izquierda)")
else:
    print(f"  Interpretación: Distribución aproximadamente simétrica")

# Curtosis
kurtosis = df['cnt'].kurtosis()
print(f"\nCurtosis:                  {kurtosis:.3f}")
if kurtosis > 0:
    print(f"  Interpretación: Distribución leptocúrtica (más puntiaguda)")
elif kurtosis < 0:
    print(f"  Interpretación: Distribución platicúrtica (más aplanada)")
else:
    print(f"  Interpretación: Distribución mesocúrtica (similar a normal)")

# ==========================================
# TEST DE NORMALIDAD
# ==========================================

print("\n3. TESTS DE NORMALIDAD")
print("-" * 70)

# Test de Shapiro-Wilk
stat_shapiro, p_shapiro = stats.shapiro(df['cnt'].sample(min(5000, len(df))))
print(f"Test de Shapiro-Wilk:")
print(f"  Estadístico: {stat_shapiro:.4f}")
print(f"  P-value:     {p_shapiro:.4f}")
print(f"  Conclusión:  {'NO sigue distribución normal (p < 0.05)' if p_shapiro < 0.05 else 'Sigue distribución normal (p >= 0.05)'}")

# Test de Kolmogorov-Smirnov
stat_ks, p_ks = stats.kstest(df['cnt'], 'norm', args=(df['cnt'].mean(), df['cnt'].std()))
print(f"\nTest de Kolmogorov-Smirnov:")
print(f"  Estadístico: {stat_ks:.4f}")
print(f"  P-value:     {p_ks:.4f}")
print(f"  Conclusión:  {'NO sigue distribución normal (p < 0.05)' if p_ks < 0.05 else 'Sigue distribución normal (p >= 0.05)'}")

# ==========================================
# ANÁLISIS DE OUTLIERS
# ==========================================

print("\n4. DETECCIÓN DE OUTLIERS")
print("-" * 70)

# Método IQR
Q1 = df['cnt'].quantile(0.25)
Q3 = df['cnt'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['cnt'] < lower_bound) | (df['cnt'] > upper_bound)]
print(f"Método IQR (Rango Intercuartílico):")
print(f"  Q1:              {Q1:.0f}")
print(f"  Q3:              {Q3:.0f}")
print(f"  IQR:             {IQR:.0f}")
print(f"  Límite inferior: {lower_bound:.0f}")
print(f"  Límite superior: {upper_bound:.0f}")
print(f"  Outliers encontrados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

if len(outliers) > 0:
    print(f"\n  Valores atípicos:")
    print(outliers[['dteday', 'cnt', 'weathersit', 'temp']].head(10))

# ==========================================
# VISUALIZACIONES
# ==========================================

print("\n5. GENERANDO VISUALIZACIONES...")
print("-" * 70)

fig = plt.figure(figsize=(16, 10))

# 1. Histograma con curva normal
ax1 = plt.subplot(2, 3, 1)
plt.hist(df['cnt'], bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
# Superponer curva normal teórica
mu, sigma = df['cnt'].mean(), df['cnt'].std()
x = np.linspace(df['cnt'].min(), df['cnt'].max(), 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal teórica')
plt.axvline(df['cnt'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["cnt"].mean():.0f}')
plt.axvline(df['cnt'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["cnt"].median():.0f}')
plt.xlabel('Número de alquileres (cnt)')
plt.ylabel('Densidad')
plt.title('Distribución de alquileres totales')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Boxplot
ax2 = plt.subplot(2, 3, 2)
bp = plt.boxplot(df['cnt'], vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)
plt.ylabel('Número de alquileres')
plt.title('Boxplot de alquileres')
plt.grid(True, alpha=0.3, axis='y')
# Añadir estadísticas
textstr = f'Media: {df["cnt"].mean():.0f}\nMediana: {df["cnt"].median():.0f}\nQ1: {Q1:.0f}\nQ3: {Q3:.0f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(1.15, 0.5, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='center', bbox=props)

# 3. Q-Q Plot
ax3 = plt.subplot(2, 3, 3)
stats.probplot(df['cnt'], dist="norm", plot=plt)
plt.title('Q-Q Plot (Normalidad)')
plt.grid(True, alpha=0.3)

# 4. Densidad (KDE)
ax4 = plt.subplot(2, 3, 4)
df['cnt'].plot(kind='density', linewidth=2, color='blue')
plt.axvline(df['cnt'].mean(), color='red', linestyle='--', linewidth=2, label='Media')
plt.axvline(df['cnt'].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
plt.xlabel('Número de alquileres')
plt.ylabel('Densidad')
plt.title('Estimación de densidad (KDE)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Distribución acumulada
ax5 = plt.subplot(2, 3, 5)
sorted_data = np.sort(df['cnt'])
cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
plt.plot(sorted_data, cumulative, linewidth=2)
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Mediana')
plt.xlabel('Número de alquileres')
plt.ylabel('Probabilidad acumulada')
plt.title('Función de distribución acumulada (ECDF)')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Violinplot
ax6 = plt.subplot(2, 3, 6)
parts = plt.violinplot([df['cnt']], positions=[1], widths=0.7, 
                       showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightcoral')
    pc.set_alpha(0.7)
plt.ylabel('Número de alquileres')
plt.title('Violin Plot de alquileres')
plt.xticks([1], ['cnt'])
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('02_analisis_variable_respuesta.png', dpi=300, bbox_inches='tight')
print("✓ Visualizaciones guardadas en '02_analisis_variable_respuesta.png'")

# ==========================================
# RESUMEN EJECUTIVO
# ==========================================

print("\n" + "="*70)
print("RESUMEN EJECUTIVO - VARIABLE RESPUESTA 'cnt'")
print("="*70)

print(f"""
CARACTERÍSTICAS PRINCIPALES:
• Promedio:     {df['cnt'].mean():.0f} alquileres/día
• Variabilidad: CV = {cv:.1f}% (alta dispersión)
• Distribución: Aproximadamente simétrica (skew ≈ 0)
• Outliers:     {len(outliers)} valores atípicos ({len(outliers)/len(df)*100:.1f}%)

RECOMENDACIONES PARA MODELIZACIÓN:
1. Considerar transformación logarítmica o Box-Cox por asimetría
2. Evaluar si los outliers corresponden a eventos especiales
3. Validar normalidad de residuos en el modelo final
4. Posible uso de modelos robustos a outliers (Huber, RANSAC)
""")

print("="*70)
