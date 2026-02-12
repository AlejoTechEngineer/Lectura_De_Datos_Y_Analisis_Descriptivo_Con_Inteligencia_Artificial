"""
Script 3: Análisis de Correlaciones
Bike Sharing Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Cargar datos
df = pd.read_csv('../bike+sharing+dataset/day.csv')

print("="*70)
print("ANÁLISIS DE CORRELACIONES CON LA VARIABLE RESPUESTA 'cnt'")
print("="*70)

# ==========================================
# CORRELACIÓN DE PEARSON
# ==========================================

print("\n1. COEFICIENTE DE CORRELACIÓN DE PEARSON")
print("-" * 70)

# Variables numéricas relevantes
numerical_vars = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 
                  'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 
                  'casual', 'registered', 'cnt']

# Calcular correlaciones con cnt
correlations_pearson = df[numerical_vars].corr()['cnt'].sort_values(ascending=False)

print("\nCorrelaciones de Pearson con 'cnt':")
print("-" * 70)
for var, corr in correlations_pearson.items():
    if var != 'cnt':
        # Calcular p-value
        _, p_value = pearsonr(df[var], df['cnt'])
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        # Interpretación
        if abs(corr) >= 0.7:
            strength = "MUY FUERTE"
        elif abs(corr) >= 0.5:
            strength = "FUERTE"
        elif abs(corr) >= 0.3:
            strength = "MODERADA"
        elif abs(corr) >= 0.1:
            strength = "DÉBIL"
        else:
            strength = "MUY DÉBIL"
        
        direction = "positiva" if corr > 0 else "negativa"
        
        print(f"{var:15s}: {corr:7.4f} {significance:3s}  [{strength:12s} {direction}] (p={p_value:.4f})")

print("\nLeyenda de significancia:")
print("  *** p < 0.001  (altamente significativo)")
print("  **  p < 0.01   (muy significativo)")
print("  *   p < 0.05   (significativo)")
print("  ns  p >= 0.05  (no significativo)")

# ==========================================
# CORRELACIÓN DE SPEARMAN
# ==========================================

print("\n2. COEFICIENTE DE CORRELACIÓN DE SPEARMAN (no paramétrico)")
print("-" * 70)

correlations_spearman = {}
print("\nCorrelaciones de Spearman con 'cnt':")
print("-" * 70)

for var in numerical_vars:
    if var != 'cnt':
        corr_s, p_value_s = spearmanr(df[var], df['cnt'])
        correlations_spearman[var] = corr_s
        significance = "***" if p_value_s < 0.001 else "**" if p_value_s < 0.01 else "*" if p_value_s < 0.05 else "ns"
        print(f"{var:15s}: {corr_s:7.4f} {significance:3s} (p={p_value_s:.4f})")

# ==========================================
# CATEGORIZACIÓN DE CORRELACIONES
# ==========================================

print("\n3. CATEGORIZACIÓN DE VARIABLES POR FUERZA DE CORRELACIÓN")
print("-" * 70)

strong_positive = correlations_pearson[(correlations_pearson > 0.5) & (correlations_pearson.index != 'cnt')]
moderate_positive = correlations_pearson[(correlations_pearson >= 0.3) & (correlations_pearson < 0.5)]
weak_positive = correlations_pearson[(correlations_pearson > 0) & (correlations_pearson < 0.3)]
weak_negative = correlations_pearson[(correlations_pearson < 0) & (correlations_pearson > -0.3)]
moderate_negative = correlations_pearson[(correlations_pearson <= -0.3) & (correlations_pearson > -0.5)]
strong_negative = correlations_pearson[correlations_pearson <= -0.5]

print(f"\n✓ Correlación FUERTE POSITIVA (r > 0.5): {len(strong_positive)} variables")
for var, corr in strong_positive.items():
    print(f"    • {var}: {corr:.3f}")

print(f"\n✓ Correlación MODERADA POSITIVA (0.3 ≤ r ≤ 0.5): {len(moderate_positive)} variables")
for var, corr in moderate_positive.items():
    print(f"    • {var}: {corr:.3f}")

print(f"\n✓ Correlación DÉBIL POSITIVA (0 < r < 0.3): {len(weak_positive)} variables")
for var, corr in weak_positive.items():
    print(f"    • {var}: {corr:.3f}")

print(f"\n✓ Correlación DÉBIL NEGATIVA (-0.3 < r < 0): {len(weak_negative)} variables")
for var, corr in weak_negative.items():
    print(f"    • {var}: {corr:.3f}")

print(f"\n✓ Correlación MODERADA NEGATIVA (-0.5 ≤ r ≤ -0.3): {len(moderate_negative)} variables")
for var, corr in moderate_negative.items():
    print(f"    • {var}: {corr:.3f}")

print(f"\n✓ Correlación FUERTE NEGATIVA (r < -0.5): {len(strong_negative)} variables")
for var, corr in strong_negative.items():
    print(f"    • {var}: {corr:.3f}")

# ==========================================
# MULTICOLINEALIDAD
# ==========================================

print("\n4. DETECCIÓN DE MULTICOLINEALIDAD")
print("-" * 70)

# Matriz de correlación entre predictores (sin cnt, casual, registered)
predictors = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 
              'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

corr_matrix = df[predictors].corr()

print("\nPares de variables con alta correlación (|r| > 0.8):")
print("-" * 70)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            high_corr_pairs.append((var1, var2, corr_val))
            print(f"  • {var1} <-> {var2}: {corr_val:.4f}")

if len(high_corr_pairs) == 0:
    print("  No se encontraron pares con |r| > 0.8")
else:
    print(f"\nADVERTENCIA: Se detectaron {len(high_corr_pairs)} pares de variables altamente correlacionadas.")
    print("   Considere eliminar una de cada par para evitar multicolinealidad en el modelo.")

# ==========================================
# VISUALIZACIONES
# ==========================================

print("\n5. GENERANDO VISUALIZACIONES...")
print("-" * 70)

fig = plt.figure(figsize=(18, 12))

# 1. Matriz de correlación completa (heatmap)
ax1 = plt.subplot(2, 3, 1)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, mask=mask,
            vmin=-1, vmax=1)
plt.title('Matriz de Correlación (Predictores)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 2. Barplot de correlaciones con cnt
ax2 = plt.subplot(2, 3, 2)
corr_sorted = correlations_pearson[correlations_pearson.index != 'cnt'].sort_values()
colors = ['red' if x < 0 else 'green' for x in corr_sorted.values]
plt.barh(range(len(corr_sorted)), corr_sorted.values, color=colors, alpha=0.7, edgecolor='black')
plt.yticks(range(len(corr_sorted)), corr_sorted.index)
plt.xlabel('Coeficiente de Correlación de Pearson')
plt.title('Correlaciones con cnt (ordenadas)', fontsize=12, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=0.8, alpha=0.5, label='Fuerte (±0.5)')
plt.axvline(x=-0.5, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
plt.axvline(x=0.3, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label='Moderada (±0.3)')
plt.axvline(x=-0.3, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='x')

# 3. Scatterplot: temp vs cnt
ax3 = plt.subplot(2, 3, 3)
plt.scatter(df['temp'], df['cnt'], alpha=0.5, s=30, color='blue', edgecolors='black', linewidth=0.5)
z = np.polyfit(df['temp'], df['cnt'], 1)
p = np.poly1d(z)
plt.plot(df['temp'], p(df['temp']), "r--", linewidth=2, 
         label=f'r = {correlations_pearson["temp"]:.3f}')
plt.xlabel('Temperatura normalizada')
plt.ylabel('Número de alquileres (cnt)')
plt.title('Relación: Temperatura vs Alquileres', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Scatterplot: yr vs cnt
ax4 = plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='yr', y='cnt', hue='yr',
            palette=['lightcoral', 'lightblue'], legend=False)
plt.xlabel('Año (0=2011, 1=2012)')
plt.ylabel('Número de alquileres (cnt)')
plt.title(f'Relación: Año vs Alquileres (r={correlations_pearson["yr"]:.3f})', 
          fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 5. Scatterplot: hum vs cnt
ax5 = plt.subplot(2, 3, 5)
plt.scatter(df['hum'], df['cnt'], alpha=0.5, s=30, color='green', edgecolors='black', linewidth=0.5)
z = np.polyfit(df['hum'], df['cnt'], 1)
p = np.poly1d(z)
plt.plot(df['hum'], p(df['hum']), "r--", linewidth=2, 
         label=f'r = {correlations_pearson["hum"]:.3f}')
plt.xlabel('Humedad normalizada')
plt.ylabel('Número de alquileres (cnt)')
plt.title('Relación: Humedad vs Alquileres', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Scatterplot: weathersit vs cnt
ax6 = plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='weathersit', y='cnt', hue='weathersit',
            palette='Set2', legend=False)
plt.xlabel('Condición climática (1=Despejado, 2=Nublado, 3=Lluvia ligera)')
plt.ylabel('Número de alquileres (cnt)')
plt.title(f'Relación: Clima vs Alquileres (r={correlations_pearson["weathersit"]:.3f})', 
          fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_analisis_correlaciones.png', dpi=300, bbox_inches='tight')
print("✓ Visualizaciones guardadas en '03_analisis_correlaciones.png'")

# ==========================================
# GUARDAR RESULTADOS
# ==========================================

# Crear DataFrame con resultados
results = pd.DataFrame({
    'Variable': correlations_pearson.index,
    'Pearson_r': correlations_pearson.values,
    'Spearman_r': [correlations_spearman.get(var, np.nan) for var in correlations_pearson.index]
})

results = results[results['Variable'] != 'cnt']
results = results.sort_values('Pearson_r', ascending=False)

results.to_csv('03_correlaciones_resultados.csv', index=False)
print("✓ Resultados guardados en '03_correlaciones_resultados.csv'")

# ==========================================
# RESUMEN EJECUTIVO
# ==========================================

print("\n" + "="*70)
print("RESUMEN EJECUTIVO - ANÁLISIS DE CORRELACIONES")
print("="*70)

print(f"""
VARIABLES MÁS IMPORTANTES PARA PREDECIR 'cnt':

1. TEMPERATURA (temp): r = {correlations_pearson['temp']:.3f}
   → Mayor temperatura → Más alquileres

2. TEMPERATURA APARENTE (atemp): r = {correlations_pearson['atemp']:.3f}
   → ALTA COLINEALIDAD con temp (r = {corr_matrix.loc['temp', 'atemp']:.3f})
   → RECOMENDACIÓN: Usar solo una de las dos

3. AÑO (yr): r = {correlations_pearson['yr']:.3f}
   → Tendencia creciente entre 2011 y 2012

4. ESTACIÓN (season): r = {correlations_pearson['season']:.3f}
   → Efecto estacional moderado

VARIABLES CON CORRELACIÓN NEGATIVA:

• CONDICIÓN CLIMÁTICA (weathersit): r = {correlations_pearson['weathersit']:.3f}
  → Peor clima → Menos alquileres

• HUMEDAD (hum): r = {correlations_pearson['hum']:.3f}
  → Mayor humedad → Ligera disminución de alquileres

CONCLUSIÓN:
Las variables meteorológicas (especialmente temperatura) son los 
predictores más fuertes del número de alquileres. Se recomienda
eliminar 'atemp' del modelo por multicolinealidad con 'temp'.
""")

print("="*70)
