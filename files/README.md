# Análisis Descriptivo - Bike Sharing Dataset

## Descripción del Proyecto

Este proyecto contiene un análisis descriptivo completo del dataset de Bike Sharing de Washington D.C. (2011-2012). El objetivo es explorar los datos, identificar patrones y preparar el dataset para modelización predictiva de la variable respuesta `cnt` (total de bicicletas alquiladas).

## Enfoque de Desarrollo

A diferencia de un enfoque monolítico (todo el análisis en un solo script),
este proyecto fue desarrollado bajo un enfoque modular y estructurado.

Cada etapa del análisis fue separada en scripts independientes,
permitiendo:

- Mejor organización del código
- Mayor claridad conceptual
- Separación de responsabilidades
- Reproducibilidad paso a paso
- Fácil mantenimiento y escalabilidad

Este enfoque refleja buenas prácticas de ingeniería de software aplicadas al análisis de datos.

## Estructura de Archivos

### Scripts de Análisis (ejecutar en orden)

1. **01_exploracion_inicial.py**
   - Carga y exploración inicial del dataset
   - Verificación de valores faltantes
   - Detección de duplicados
   - Tipos de datos y estructura

2. **02_analisis_variable_respuesta.py**
   - Estadísticas descriptivas de 'cnt'
   - Tests de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov)
   - Detección de outliers (método IQR)
   - Análisis de asimetría y curtosis
   - Visualizaciones: histograma, boxplot, Q-Q plot, densidad

3. **03_analisis_correlaciones.py**
   - Correlación de Pearson y Spearman
   - Identificación de variables clave
   - Detección de multicolinealidad
   - Categorización de correlaciones
   - Scatterplots y matriz de correlación

4. **04_analisis_distribuciones.py**
   - Análisis por estación del año
   - Análisis por condición climática
   - Patrones por día de la semana
   - Comparación 2011 vs 2012
   - Análisis mensual
   - Tipos de usuarios (casuales vs registrados)
   - Visualizaciones completas (12 gráficos)

5. **05_division_datos.py**
   - División temporal 80-20 (recomendado)
   - División aleatoria 80-20 (comparación)
   - Análisis de representatividad
   - Comparación de distribuciones
   - Generación de conjuntos train/test
  
Importante: Dado que el dataset representa una serie temporal, la división aleatoria podría introducir información futura en el entrenamiento (data leakage), afectando la validez del modelo.

### Archivos de Datos

- **day.csv** - Dataset original (731 registros)
- **hour.csv** - Dataset por horas (17,379 registros) - opcional
- **Readme.txt** - Documentación del dataset
- **train_temporal.csv** - Conjunto de entrenamiento (generado)
- **test_temporal.csv** - Conjunto de validación (generado)

### Archivos de Resultados

- **02_analisis_variable_respuesta.png** - Visualizaciones de 'cnt'
- **03_analisis_correlaciones.png** - Análisis de correlaciones
- **03_correlaciones_resultados.csv** - Tabla de correlaciones
- **04_analisis_distribuciones.png** - Gráficos de distribuciones
- **05_division_datos.png** - Visualización de división train/test
- **05_division_info.csv** - Información de la división

## Cómo Usar

### Requisitos Previos

```bash
# Instalar las librerías necesarias
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```
Python 3.10+ recomendado

### Ejecución

1. **Coloca los archivos de datos** en el mismo directorio que los scripts:
   - day.csv
   - hour.csv (opcional)
   - Readme.txt

2. **Ejecuta los scripts en orden** en Visual Studio Code o en terminal:

```bash
# Script 1: Exploración inicial
python 01_exploracion_inicial.py

# Script 2: Análisis de variable respuesta
python 02_analisis_variable_respuesta.py

# Script 3: Análisis de correlaciones
python 03_analisis_correlaciones.py

# Script 4: Análisis de distribuciones
python 04_analisis_distribuciones.py

# Script 5: División de datos
python 05_division_datos.py
```

3. **Revisa los resultados**:
   - Los gráficos se guardan automáticamente como PNG
   - Las tablas se guardan como CSV
   - La consola muestra resúmenes ejecutivos

## Principales Hallazgos

### 1. Calidad de Datos
- **NO hay valores faltantes** en ninguna variable
- **NO hay duplicados**
- Dataset completo: 731 registros, 16 variables

### 2. Variable Respuesta 'cnt'
- **Media**: 4,504 alquileres/día
- **Rango**: 22 - 8,714 alquileres
- **Distribución**: aproximadamente simétrica (skew ≈ 0)
- **Variabilidad**: CV = 43% (alta dispersión)
- El test de Shapiro-Wilk indica desviación estadísticamente significativa de la normalidad (p < 0.05).

### 3. Correlaciones Clave

**Correlaciones fuertes con la variable respuesta (r > 0.5):**
- `temp` (temperatura): **r = 0.627**
- `atemp` (temperatura aparente): **r = 0.631**
- `yr` (año): **r = 0.567**

**Nota**: Las variables `casual` y `registered` presentan correlación muy fuerte con `cnt`, pero no se consideran predictores independientes ya que su suma constituye la variable respuesta.

**Correlaciones MODERADAS (0.3 ≤ r ≤ 0.5)**:
- `season` (estación): **r = 0.406**

**Correlaciones NEGATIVAS**:
- `weathersit` (clima): **r = -0.297**
- `windspeed` (viento): **r = -0.235**
- `hum` (humedad): **r = -0.101**

### 4. Patrones Identificados

**Estacionalidad**:
- Otoño: 5,644 alquileres/día (máximo)
- Verano: 4,992 alquileres/día
- Invierno: 4,728 alquileres/día
- Primavera: 2,604 alquileres/día (mínimo)

**Condición Climática**:
- Despejado: 4,877 alquileres/día
- Nublado: 4,036 alquileres/día
- Lluvia ligera: 1,803 alquileres/día (-63%)

**Tendencia Temporal**:
- Crecimiento 2011 → 2012: **+64.4%**
- Media 2011: 3,406 alquileres/día
- Media 2012: 5,600 alquileres/día

**Tipos de Usuarios**:
- Registrados: 81.2% del total
- Casuales: 18.8% del total
- Ratio: 4.3 registrados por cada casual

### 5. División de Datos

**Método Recomendado**: División Temporal 80-20

- **Entrenamiento**: 584 registros (01/01/2011 - 06/08/2012)
- **Validación**: 147 registros (07/08/2012 - 31/12/2012)

**Justificación**: 
- Respeta el orden temporal
- Evita data leakage
- Simula predicción del futuro

## Recomendaciones para Modelización

1. **Transformación de la variable respuesta**:
   - Aunque la distribución es aproximadamente simétrica, los tests formales indican desviación de normalidad.

2. **Selección de variables**:
   - **Incluir**: temp, yr, season, weathersit, mnth
   - **Excluir**: atemp (colinealidad con temp > 0.99)
   - **Considerar**: interacciones entre variables temporales y meteorológicas

3. **Manejo de outliers**:
   - Evaluar si corresponden a eventos especiales
   - Considerar modelos robustos (Huber, RANSAC)

4. **Validación**:
   - Usar división temporal 80-20
   - Considerar validación cruzada temporal (rolling window)
   - Verificar normalidad de residuos

5. **Feature Engineering**:
   - Crear variables de interacción (temp × season)
   - Variables de rezago (lag features)
   - Medias móviles

## Descripción de Variables

| Variable | Descripción | Tipo |
|----------|-------------|------|
| instant | Índice del registro | Numérico |
| dteday | Fecha | Fecha |
| season | Estación (1:Primavera, 2:Verano, 3:Otoño, 4:Invierno) | Categórico |
| yr | Año (0:2011, 1:2012) | Categórico |
| mnth | Mes (1-12) | Categórico |
| hr | Hora (0-23) - solo en hour.csv | Categórico |
| holiday | Día festivo (0:No, 1:Sí) | Binario |
| weekday | Día de la semana (0-6) | Categórico |
| workingday | Día laborable (0:No, 1:Sí) | Binario |
| weathersit | Condición climática (1:Despejado, 2:Nublado, 3:Lluvia ligera, 4:Lluvia fuerte) | Categórico |
| temp | Temperatura normalizada (÷41) | Numérico |
| atemp | Temperatura aparente normalizada (÷50) | Numérico |
| hum | Humedad normalizada (÷100) | Numérico |
| windspeed | Velocidad del viento normalizada (÷67) | Numérico |
| casual | Usuarios casuales | Numérico |
| registered | Usuarios registrados | Numérico |
| **cnt** | **Total de alquileres (VARIABLE RESPUESTA)** | **Numérico** |

## Uso en Visual Studio Code

### 1. Abrir el proyecto

```bash
# Abre VS Code en el directorio del proyecto
code .
```

### 2. Configurar entorno virtual (opcional pero recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### 3. Ejecutar scripts

- **Opción 1**: Clic derecho en el archivo → "Run Python File in Terminal"
- **Opción 2**: Presionar F5 (ejecutar con depuración)
- **Opción 3**: Usar terminal integrada: `python nombre_script.py`

### 4. Ver resultados

- Los gráficos PNG se generan en el mismo directorio
- Los archivos CSV se pueden abrir directamente en VS Code
- La salida de consola aparece en el terminal integrado

## Referencias

- Dataset original: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- Citación requerida:
  ```
  Fanaee-T, Hadi, and Gama, Joao, 
  "Event labeling combining ensemble detectors and background knowledge", 
  Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
  ```

## Autor

Análisis realizado para el curso de Ciencia de Datos.

## Notas Adicionales

- Los scripts están comentados extensivamente para facilitar el aprendizaje
- Cada script es independiente pero se recomienda ejecutar en orden
- Los gráficos usan paletas de colores profesionales
- Todos los resultados se imprimen en consola con formato estructurado
- Los archivos CSV generados pueden usarse directamente en Excel o herramientas BI

## Tips de Uso

1. **Para análisis rápido**: Ejecuta solo scripts 1, 3 y 5
2. **Para presentación**: Usa las imágenes PNG generadas
3. **Para modelización**: Comienza con train_temporal.csv y test_temporal.csv
4. **Para exploración**: Modifica los scripts según tus necesidades

---

Proyecto desarrollado paso a paso siguiendo buenas prácticas de análisis exploratorio y preparación de datos.



