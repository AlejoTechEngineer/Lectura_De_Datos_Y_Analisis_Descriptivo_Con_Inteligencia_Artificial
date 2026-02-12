"""
Script 4: Análisis de Distribuciones y Visualizaciones
Bike Sharing Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Cargar datos
df = pd.read_csv('../bike+sharing+dataset/day.csv')

# Convertir fecha a datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# Mapeos para categorías
season_map = {1: 'Primavera', 2: 'Verano', 3: 'Otoño', 4: 'Invierno'}
weather_map = {1: 'Despejado', 2: 'Nublado/Niebla', 3: 'Lluvia ligera', 4: 'Lluvia fuerte'}
weekday_map = {0: 'Dom', 1: 'Lun', 2: 'Mar', 3: 'Mié', 4: 'Jue', 5: 'Vie', 6: 'Sáb'}

df['season_name'] = df['season'].map(season_map)
df['weather_name'] = df['weathersit'].map(weather_map)
df['weekday_name'] = df['weekday'].map(weekday_map)
df['year'] = df['yr'].map({0: '2011', 1: '2012'})

print("="*70)
print("ANÁLISIS DE DISTRIBUCIONES Y PATRONES")
print("="*70)

# ==========================================
# ANÁLISIS POR ESTACIÓN
# ==========================================

print("\n1. ANÁLISIS POR ESTACIÓN DEL AÑO")
print("-" * 70)

season_stats = df.groupby('season_name')['cnt'].agg([
    ('Media', 'mean'),
    ('Mediana', 'median'),
    ('Desv.Std', 'std'),
    ('Mínimo', 'min'),
    ('Máximo', 'max'),
    ('Registros', 'count')
]).round(2)

# Ordenar por estaciones lógicas
season_order = ['Primavera', 'Verano', 'Otoño', 'Invierno']
season_stats = season_stats.reindex(season_order)

print(season_stats)

print("\nInterpretación:")
best_season = season_stats['Media'].idxmax()
worst_season = season_stats['Media'].idxmin()
print(f"  • Mejor estación:  {best_season} ({season_stats.loc[best_season, 'Media']:.0f} alquileres/día)")
print(f"  • Peor estación:   {worst_season} ({season_stats.loc[worst_season, 'Media']:.0f} alquileres/día)")
print(f"  • Diferencia:      {season_stats.loc[best_season, 'Media'] - season_stats.loc[worst_season, 'Media']:.0f} alquileres/día ({((season_stats.loc[best_season, 'Media'] / season_stats.loc[worst_season, 'Media']) - 1) * 100:.1f}%)")

# ==========================================
# ANÁLISIS POR CONDICIÓN CLIMÁTICA
# ==========================================

print("\n2. ANÁLISIS POR CONDICIÓN CLIMÁTICA")
print("-" * 70)

weather_stats = df.groupby('weather_name')['cnt'].agg([
    ('Media', 'mean'),
    ('Mediana', 'median'),
    ('Desv.Std', 'std'),
    ('Registros', 'count')
]).round(2)

print(weather_stats)

print("\nInterpretación:")
for weather in weather_stats.index:
    pct = (weather_stats.loc[weather, 'Registros'] / len(df)) * 100
    print(f"  • {weather:20s}: {weather_stats.loc[weather, 'Media']:6.0f} alquileres/día ({pct:5.1f}% de los días)")

# ==========================================
# ANÁLISIS POR DÍA DE LA SEMANA
# ==========================================

print("\n3. ANÁLISIS POR DÍA DE LA SEMANA")
print("-" * 70)

weekday_stats = df.groupby('weekday_name')['cnt'].agg([
    ('Media', 'mean'),
    ('Mediana', 'median'),
    ('Registros', 'count')
]).round(2)

# Ordenar por día de la semana
weekday_order = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
weekday_stats = weekday_stats.reindex(weekday_order)

print(weekday_stats)

print("\nInterpretación:")
best_day = weekday_stats['Media'].idxmax()
worst_day = weekday_stats['Media'].idxmin()
print(f"  • Día con más alquileres:  {best_day} ({weekday_stats.loc[best_day, 'Media']:.0f} alquileres)")
print(f"  • Día con menos alquileres: {worst_day} ({weekday_stats.loc[worst_day, 'Media']:.0f} alquileres)")

# ==========================================
# ANÁLISIS POR AÑO
# ==========================================

print("\n4. ANÁLISIS COMPARATIVO POR AÑO")
print("-" * 70)

year_stats = df.groupby('year')['cnt'].agg([
    ('Media', 'mean'),
    ('Mediana', 'median'),
    ('Total', 'sum'),
    ('Registros', 'count')
]).round(2)

print(year_stats)

growth = ((year_stats.loc['2012', 'Media'] - year_stats.loc['2011', 'Media']) / year_stats.loc['2011', 'Media']) * 100
print(f"\nCrecimiento interanual: {growth:.1f}%")
print(f"Incremento promedio diario: {year_stats.loc['2012', 'Media'] - year_stats.loc['2011', 'Media']:.0f} alquileres")

# ==========================================
# ANÁLISIS MENSUAL
# ==========================================

print("\n5. ANÁLISIS MENSUAL")
print("-" * 70)

month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
               'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

monthly_stats = df.groupby('mnth')['cnt'].agg([
    ('Media', 'mean'),
    ('Registros', 'count')
]).round(2)

monthly_stats.index = month_names

print(monthly_stats)

print("\nInterpretación:")
best_month = monthly_stats['Media'].idxmax()
worst_month = monthly_stats['Media'].idxmin()
print(f"  • Mejor mes:  {best_month} ({monthly_stats.loc[best_month, 'Media']:.0f} alquileres/día)")
print(f"  • Peor mes:   {worst_month} ({monthly_stats.loc[worst_month, 'Media']:.0f} alquileres/día)")

# ==========================================
# ANÁLISIS DE TIPOS DE USUARIO
# ==========================================

print("\n6. ANÁLISIS DE TIPOS DE USUARIO")
print("-" * 70)

print(f"Usuarios CASUALES:")
print(f"  • Media diaria:    {df['casual'].mean():.0f} alquileres")
print(f"  • Total 2011-2012: {df['casual'].sum():,} alquileres")
print(f"  • % del total:     {(df['casual'].sum() / df['cnt'].sum()) * 100:.1f}%")

print(f"\nUsuarios REGISTRADOS:")
print(f"  • Media diaria:    {df['registered'].mean():.0f} alquileres")
print(f"  • Total 2011-2012: {df['registered'].sum():,} alquileres")
print(f"  • % del total:     {(df['registered'].sum() / df['cnt'].sum()) * 100:.1f}%")

print(f"\nRatio Registrados/Casuales: {df['registered'].mean() / df['casual'].mean():.2f}:1")

# ==========================================
# VISUALIZACIONES COMPLETAS
# ==========================================

print("\n7. GENERANDO VISUALIZACIONES COMPLETAS...")
print("-" * 70)

# Crear figura grande con múltiples gráficos
fig = plt.figure(figsize=(20, 12))

# 1. Serie temporal completa
ax1 = plt.subplot(3, 4, 1)
plt.plot(df['dteday'], df['cnt'], linewidth=0.8, alpha=0.7, color='steelblue')
plt.fill_between(df['dteday'], df['cnt'], alpha=0.3, color='steelblue')
plt.xlabel('Fecha')
plt.ylabel('Alquileres')
plt.title('Evolución Temporal de Alquileres', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 2. Por estación
ax2 = plt.subplot(3, 4, 2)
season_order = ['Primavera', 'Verano', 'Otoño', 'Invierno']
sns.boxplot(data=df, x='season_name', y='cnt',
            hue='season_name',
            order=season_order,
            palette='Set2',
            legend=False)
plt.xlabel('Estación')
plt.ylabel('Alquileres')
plt.title('Alquileres por Estación', fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# 3. Por condición climática
ax3 = plt.subplot(3, 4, 3)
weather_data = df.groupby('weather_name')['cnt'].mean().sort_values(ascending=False)
colors_weather = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
plt.bar(range(len(weather_data)), weather_data.values, color=colors_weather[:len(weather_data)], 
        edgecolor='black', linewidth=1.5)
plt.xticks(range(len(weather_data)), weather_data.index, rotation=45, ha='right')
plt.ylabel('Promedio de Alquileres')
plt.title('Alquileres por Condición Climática', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 4. Por día de la semana
ax4 = plt.subplot(3, 4, 4)
weekday_data = df.groupby('weekday_name')['cnt'].mean().reindex(weekday_order)
colors_week = ['#e74c3c' if d in ['Sáb', 'Dom'] else '#3498db' for d in weekday_order]
plt.bar(weekday_order, weekday_data.values, color=colors_week, edgecolor='black', linewidth=1.5)
plt.ylabel('Promedio de Alquileres')
plt.xlabel('Día de la Semana')
plt.title('Alquileres por Día de la Semana', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 5. Comparación 2011 vs 2012 mensual
ax5 = plt.subplot(3, 4, 5)
monthly_comparison = df.groupby(['mnth', 'year'])['cnt'].mean().unstack()
monthly_comparison.index = month_names
monthly_comparison.plot(ax=ax5, marker='o', linewidth=2, markersize=8)
plt.xlabel('Mes')
plt.ylabel('Promedio de Alquileres')
plt.title('Comparación Mensual 2011 vs 2012', fontweight='bold')
plt.legend(title='Año')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 6. Temperatura vs Alquileres (con estaciones coloreadas)
ax6 = plt.subplot(3, 4, 6)
season_colors = {'Primavera': '#2ecc71', 'Verano': '#f39c12', 
                 'Otoño': '#e67e22', 'Invierno': '#3498db'}
for season in season_order:
    data_season = df[df['season_name'] == season]
    plt.scatter(data_season['temp'], data_season['cnt'], 
                label=season, alpha=0.6, s=30, c=season_colors[season], edgecolors='black', linewidth=0.5)
plt.xlabel('Temperatura Normalizada')
plt.ylabel('Alquileres')
plt.title('Temperatura vs Alquileres por Estación', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Comparación: Día laborable vs Fin de semana/festivo
ax7 = plt.subplot(3, 4, 7)
workday_comparison = df.groupby('workingday')['cnt'].mean()
labels = ['Fin de semana\n/Festivo', 'Día\nlaborable']
colors_wd = ['#e74c3c', '#3498db']
plt.bar(labels, workday_comparison.values, color=colors_wd, edgecolor='black', linewidth=1.5)
plt.ylabel('Promedio de Alquileres')
plt.title('Día Laborable vs Fin de Semana', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 8. Usuarios casuales vs registrados
ax8 = plt.subplot(3, 4, 8)
user_types = ['Casuales', 'Registrados']
user_means = [df['casual'].mean(), df['registered'].mean()]
colors_users = ['#e74c3c', '#2ecc71']
plt.bar(user_types, user_means, color=colors_users, edgecolor='black', linewidth=1.5)
plt.ylabel('Promedio Diario de Alquileres')
plt.title('Tipos de Usuarios', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
# Añadir valores
for i, v in enumerate(user_means):
    plt.text(i, v + 50, f'{v:.0f}', ha='center', fontweight='bold')

# 9. Heatmap: Día de semana vs Mes
ax9 = plt.subplot(3, 4, 9)
heatmap_data = df.pivot_table(values='cnt', index='weekday', columns='mnth', aggfunc='mean')
heatmap_data.index = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb']
heatmap_data.columns = month_names
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Alquileres'})
plt.title('Heatmap: Día x Mes', fontweight='bold')
plt.xlabel('Mes')
plt.ylabel('Día de la Semana')

# 10. Humedad vs Alquileres
ax10 = plt.subplot(3, 4, 10)
plt.scatter(df['hum'], df['cnt'], alpha=0.4, s=30, color='blue', edgecolors='black', linewidth=0.5)
z = np.polyfit(df['hum'], df['cnt'], 1)
p = np.poly1d(z)
plt.plot(df['hum'], p(df['hum']), "r--", linewidth=2, label='Tendencia')
plt.xlabel('Humedad Normalizada')
plt.ylabel('Alquileres')
plt.title('Humedad vs Alquileres', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 11. Velocidad del viento vs Alquileres
ax11 = plt.subplot(3, 4, 11)
plt.scatter(df['windspeed'], df['cnt'], alpha=0.4, s=30, color='green', edgecolors='black', linewidth=0.5)
z = np.polyfit(df['windspeed'], df['cnt'], 1)
p = np.poly1d(z)
plt.plot(df['windspeed'], p(df['windspeed']), "r--", linewidth=2, label='Tendencia')
plt.xlabel('Velocidad del Viento Normalizada')
plt.ylabel('Alquileres')
plt.title('Viento vs Alquileres', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 12. Distribución de temperaturas por estación
ax12 = plt.subplot(3, 4, 12)
for season in season_order:
    data_season = df[df['season_name'] == season]['temp']
    plt.hist(data_season, bins=20, alpha=0.5, label=season, edgecolor='black', linewidth=0.5)
plt.xlabel('Temperatura Normalizada')
plt.ylabel('Frecuencia')
plt.title('Distribución de Temperaturas por Estación', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_analisis_distribuciones.png', dpi=300, bbox_inches='tight')
print("✓ Visualizaciones guardadas en '04_analisis_distribuciones.png'")

# ==========================================
# RESUMEN EJECUTIVO
# ==========================================

print("\n" + "="*70)
print("RESUMEN EJECUTIVO - ANÁLISIS DE DISTRIBUCIONES")
print("="*70)

print(f"""
PATRONES IDENTIFICADOS:

1. ESTACIONALIDAD MARCADA:
   • Verano y Otoño: Picos de demanda (>5,000 alquileres/día)
   • Primavera: Mínimo anual (2,604 alquileres/día)
   • Diferencia máxima: 117% entre Otoño y Primavera

2. IMPACTO CLIMÁTICO:
   • Días despejados: 4,877 alquileres/día
   • Días con lluvia: 1,803 alquileres/día (-63%)
   • Temperatura: Principal factor meteorológico

3. PATRONES TEMPORALES:
   • Crecimiento 2011→2012: +64.4%
   • Comportamiento similar entre semana y fin de semana
   • Meses pico: Junio-Octubre

4. TIPOLOGÍA DE USUARIOS:
   • Usuarios registrados: {(df['registered'].sum() / df['cnt'].sum()) * 100:.1f}% del total
   • Usuarios casuales: {(df['casual'].sum() / df['cnt'].sum()) * 100:.1f}% del total
   • Ratio: {df['registered'].mean() / df['casual'].mean():.1f} registrados por cada casual

CONCLUSIONES:
El sistema de bike sharing muestra fuerte dependencia de factores
meteorológicos y estacionales. La tendencia creciente indica adopción
sostenida. El predominio de usuarios registrados sugiere uso regular
para transporte, no solo recreativo.
""")

print("="*70)
