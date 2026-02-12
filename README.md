<div align="center">

# 🚴‍♂️ Bike Sharing Analytics Dashboard

### *Análisis Descriptivo Completo del Sistema de Bicicletas Compartidas de Washington D.C.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20ML%20Repo-orange.svg)](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

**[Características](#-características-principales) • [Instalación](#-instalación-rápida) • [Uso](#-guía-de-uso) • [Resultados](#-resultados-principales) • [Estructura](#-estructura-del-proyecto)**

<img src="https://img.shields.io/badge/Período-2011--2012-blue?style=flat-square" alt="Period"/>
<img src="https://img.shields.io/badge/Registros-731%20días-green?style=flat-square" alt="Records"/>
<img src="https://img.shields.io/badge/Variables-16-orange?style=flat-square" alt="Variables"/>
<img src="https://img.shields.io/badge/Calidad-Sin%20Missing-success?style=flat-square" alt="Quality"/>

</div>

---

## 📋 Tabla de Contenidos

- [🎯 Descripción del Proyecto](#-descripción-del-proyecto)
- [✨ Características Principales](#-características-principales)
- [🏗️ Arquitectura Modular](#️-arquitectura-modular)
- [🚀 Instalación Rápida](#-instalación-rápida)
- [📊 Guía de Uso](#-guía-de-uso)
- [🔍 Resultados Principales](#-resultados-principales)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [💡 Recomendaciones](#-recomendaciones-para-modelización)
- [📚 Referencias](#-referencias)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un **análisis exploratorio exhaustivo** del dataset de Bike Sharing de Washington D.C., abarcando el período 2011-2012. El objetivo principal es descubrir patrones de uso, identificar factores clave que influyen en la demanda de bicicletas y preparar los datos para modelización predictiva.

### 🎓 Contexto Académico

> Desarrollado como parte del curso **Aprendizaje Automático y Minería de Datos**  
> **Profesor:** Ing. Rogerio Orlando Beltrán Castro  
> **Institución:** Fundación Universitaria Internacional de La Rioja

### 🌟 ¿Por qué este proyecto es especial?

- 🔧 **Arquitectura Modular**: A diferencia de análisis monolíticos, cada fase está separada
- 📊 **18+ Visualizaciones**: Gráficos profesionales y explicativos
- 🧪 **Rigor Estadístico**: Tests formales, validación y detección de problemas
- 📝 **Documentación Completa**: Código comentado línea por línea
- ✅ **Reproducibilidad Total**: Scripts ejecutables paso a paso

---

## ✨ Características Principales

<table align="center">
<tr>
<td width="50%">

### 🔍 Análisis Exploratorio
- ✅ Verificación de calidad de datos
- ✅ Detección de valores faltantes
- ✅ Identificación de outliers (IQR)
- ✅ Tests de normalidad (Shapiro-Wilk, KS)
- ✅ Análisis de asimetría y curtosis

</td>
<td width="50%">

### 📈 Análisis Estadístico
- 📊 Correlación Pearson y Spearman
- 📊 Detección de multicolinealidad
- 📊 Análisis de distribuciones
- 📊 Patrones temporales y patrones estacionales
- 📊 Segmentación por usuarios

</td>
</tr>
</table>

### 🎯 Hallazgos Clave

<div align="center">

| 🌡️ **Temperatura** | 📅 **Año** | 🍂 **Estación** | ☁️ **Clima** |
|:------------------:|:---------:|:---------------:|:------------:|
| r = **0.627** | r = **0.567** | r = **0.406** | r = **-0.297** |
| Mayor temp → Más uso | 64.4% crecimiento | Otoño = pico | Lluvia = -63% |

</div>

---

## 🏗️ Arquitectura Modular

### 📦 Módulos del Proyecto

| # | Script | Propósito | Output |
|:-:|--------|-----------|--------|
| **1** | `01_exploracion_inicial.py` | 🔍 Carga, validación y limpieza | `bike_sharing_clean.csv` |
| **2** | `02_analisis_variable_respuesta.py` | 📊 Estadísticas descriptivas, normalidad | `02_analisis_variable_respuesta.png` |
| **3** | `03_analisis_correlaciones.py` | 🔗 Correlaciones, multicolinealidad | `03_analisis_correlaciones.png/csv` |
| **4** | `04_analisis_distribuciones.py` | 📈 Patrones temporales y categorías | `04_analisis_distribuciones.png` |
| **5** | `05_division_datos.py` | ✂️ Train/test split temporal | `train_temporal.csv` + `test_temporal.csv` |

> 💡 **Ventaja clave**: Cada módulo es **independiente**, **testeable** y **reutilizable**

---

## 🚀 Instalación Rápida

### Prerrequisitos

- Python 3.10 o superior
- pip (gestor de paquetes)

### 1️⃣ Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/bike-sharing-analysis.git
cd bike-sharing-analysis
```

### 2️⃣ Crear Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Instalar Dependencias

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### 4️⃣ Verificar Instalación

```bash
python -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn; print('✅ Todas las librerías instaladas correctamente')"
```

---

## 📊 Guía de Uso

### 🎬 Ejecución Paso a Paso

```bash
# Paso 1: Exploración inicial
python 01_exploracion_inicial.py
# ✅ Output: bike_sharing_clean.csv

# Paso 2: Análisis de variable respuesta
python 02_analisis_variable_respuesta.py
# ✅ Output: 02_analisis_variable_respuesta.png + estadísticas en consola

# Paso 3: Análisis de correlaciones
python 03_analisis_correlaciones.py
# ✅ Output: 03_analisis_correlaciones.png + 03_correlaciones_resultados.csv

# Paso 4: Análisis de distribuciones
python 04_analisis_distribuciones.py
# ✅ Output: 04_analisis_distribuciones.png (12 gráficos)

# Paso 5: División de datos
python 05_division_datos.py
# ✅ Output: train_temporal.csv + test_temporal.csv + 05_division_datos.png
```

### ⚡ Ejecución Rápida (Solo hallazgos clave)

```bash
python 01_exploracion_inicial.py && python 03_analisis_correlaciones.py && python 05_division_datos.py
```

---

## 🔍 Resultados Principales

### 📊 Calidad de Datos

<div align="center">

| Métrica | Resultado |
|:-------:|:---------:|
| **Valores Faltantes** | ✅ 0 (0%) |
| **Duplicados** | ✅ 0 |
| **Registros** | 731 días |
| **Variables** | 16 |
| **Período** | 2011-01-01 a 2012-12-31 |

</div>

### 🎯 Variable Respuesta: `cnt` (Alquileres Totales)

<h3>📊 Estadísticas Descriptivas</h3>

<table align="center">
  <tr>
    <th>Métrica</th>
    <th>Valor</th>
  </tr>
  <tr>
    <td>Media</td>
    <td>4,504 alquileres/día</td>
  </tr>
  <tr>
    <td>Mediana</td>
    <td>4,548 alquileres/día</td>
  </tr>
  <tr>
    <td>Desv. Estándar</td>
    <td>1,937</td>
  </tr>
  <tr>
    <td>Mínimo</td>
    <td>22</td>
  </tr>
  <tr>
    <td>Máximo</td>
    <td>8,714</td>
  </tr>
  <tr>
    <td>Coef. Variación</td>
    <td>43% (alta dispersión)</td>
  </tr>
  <tr>
    <td>Asimetría</td>
    <td>≈ 0 (simétrica)</td>
  </tr>
  <tr>
    <td>Test Normalidad</td>
    <td>❌ No normal (p &lt; 0.05)</td>
  </tr>
</table>

### 🔗 Top Correlaciones con `cnt`

<table align="center">
<tr>
<th>🔥 Positivas</th>
<th>❄️ Negativas</th>
</tr>
<tr>
<td>

| Variable | r | Interpretación |
|----------|---|----------------|
| `temp` | **0.627** | 🌡️ Mayor temp → Más uso |
| `atemp` | **0.631** | ⚠️ Colineal con temp |
| `yr` | **0.567** | 📈 Tendencia creciente |
| `season` | **0.406** | 🍂 Efecto estacional |

</td>
<td>

| Variable | r | Interpretación |
|----------|---|----------------|
| `weathersit` | **-0.297** | ☁️ Mal clima → Menos uso |
| `windspeed` | **-0.235** | 💨 Viento → Menos uso |
| `hum` | **-0.101** | 💧 Humedad (débil) |

</td>
</tr>
</table>

### 📅 Patrones Temporales

<details>
<summary><b>🍂 Por Estación del Año</b></summary>

```
Otoño      ████████████████████████████ 5,644 alquileres/día
Verano     ███████████████████████ 4,992 alquileres/día
Invierno   █████████████████████ 4,728 alquileres/día
Primavera  ████████████ 2,604 alquileres/día
```

**📊 Diferencia**: 117% entre Otoño y Primavera

</details>

<details>
<summary><b>☁️ Por Condición Climática</b></summary>

```
☀️ Despejado       ████████████████████████ 4,877 alquileres/día
⛅ Nublado/Niebla  ████████████████████ 4,036 alquileres/día
🌧️ Lluvia ligera   ████████ 1,803 alquileres/día (-63%)
⛈️ Lluvia fuerte   [Muy pocos casos]
```

</details>

<details>
<summary><b>📈 Crecimiento Interanual</b></summary>

| Año | Promedio Diario | Total Anual | Crecimiento |
|:---:|:---------------:|:-----------:|:-----------:|
| 2011 | 3,406 | 1,243,103 | - |
| 2012 | 5,600 | 2,049,576 | **+64.4%** 🚀 |

</details>

<details>
<summary><b>👥 Tipos de Usuarios</b></summary>

<div align="center">

```
Usuarios Registrados ████████████████████████ 81.2%
Usuarios Casuales    ████ 18.8%

Ratio: 4.3 registrados por cada casual
```

💡 **Interpretación**: Predominio de uso regular (transporte) sobre recreativo

</div>

</details>

### ✂️ División de Datos

<div align="center">

| Conjunto | Registros | Período | Media cnt |
|:--------:|:---------:|:-------:|:---------:|
| **🎓 Train** | 584 (80%) | 2011-01-01 a 2012-08-06 | 4,153 |
| **🎯 Test** | 147 (20%) | 2012-08-07 a 2012-12-31 | 5,897 |

</div>

> ⚠️ **Nota Importante**: Se usó **división temporal** (no aleatoria) para evitar **data leakage** en series temporales

---

## 📁 Estructura del Proyecto

```
bike-sharing-analysis/
│
├── 📂 bike+sharing+dataset/     # Datos originales
│   ├── day.csv                  # Dataset diario (731 registros)
│   ├── hour.csv                 # Dataset horario (17,379 registros)
│   └── Readme.txt               # Documentación oficial UCI
│
├── 📂 files/                    # Scripts de análisis
│   ├── 01_exploracion_inicial.py
│   ├── 02_analisis_variable_respuesta.py
│   ├── 03_analisis_correlaciones.py
│   ├── 04_analisis_distribuciones.py
│   ├── 05_division_datos.py
│   └── README.md                # Este archivo
│
├── 📂 outputs/                  # Resultados generados
│   ├── 📊 02_analisis_variable_respuesta.png
│   ├── 📊 03_analisis_correlaciones.png
│   ├── 📊 04_analisis_distribuciones.png
│   ├── 📊 05_division_datos.png
│   ├── 📄 03_correlaciones_resultados.csv
│   ├── 📄 05_division_info.csv
│   ├── 📄 bike_sharing_clean.csv
│   ├── 📄 train_temporal.csv
│   └── 📄 test_temporal.csv
│
└── 📄 README.md                 # Documentación principal
```

---

## 💡 Recomendaciones para Modelización

### 🔧 Preprocesamiento

<table align="center">
<tr>
<td>

#### ✅ Variables a Incluir
- `temp` (r=0.627)
- `yr` (r=0.567)
- `season` (r=0.406)
- `weathersit` (r=-0.297)
- `mnth` (temporal)
- `weekday` (patrón semanal)

</td>
<td>

#### ❌ Variables a Excluir
- `atemp` (colinealidad con temp)
- `casual` (parte de cnt)
- `registered` (parte de cnt)
- `instant` (índice sin valor)

</td>
</tr>
</table>

### 🤖 Modelos Recomendados

| Modelo | Ventaja | Cuándo Usarlo |
|--------|---------|---------------|
| **📏 Regresión Lineal** | Interpretabilidad | Baseline, relaciones lineales |
| **🌲 Random Forest** | Captura no-linealidades | Interacciones complejas |
| **🚀 XGBoost** | Mejor rendimiento | Competencia, producción |
| **📊 Ridge/Lasso** | Regularización | Multicolinealidad, selección |
| **🧠 LSTM** | Secuencias temporales | Si se usan datos horarios |

### 🔍 Feature Engineering

<details>
<summary><b>Ideas de Nuevas Features</b></summary>

```python
# 1. Interacciones
df['temp_x_season'] = df['temp'] * df['season']

# 2. Variables de rezago (lag)
df['cnt_lag1'] = df['cnt'].shift(1)
df['cnt_lag7'] = df['cnt'].shift(7)  # Misma día semana anterior

# 3. Medias móviles
df['cnt_ma7'] = df['cnt'].rolling(window=7).mean()

# 4. Features cíclicas (para capturar estacionalidad)
df['month_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)

# 5. Indicadores booleanos
df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)
df['is_summer'] = (df['season'] == 2).astype(int)
```

</details>

---

## 📚 Referencias

### 📖 Dataset Original

> **Fanaee-T, H., & Gama, J.** (2013).  
> *Event labeling combining ensemble detectors and background knowledge.*  
> Progress in Artificial Intelligence, 2(2-3), 113-127.  
> Springer Berlin Heidelberg.  
> DOI: [10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3)

**🔗 Enlaces:**
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- [Capital Bikeshare (Washington D.C.)](https://www.capitalbikeshare.com/)

### 🛠️ Herramientas Utilizadas

| Librería | Versión | Propósito |
|----------|---------|-----------|
| pandas | 2.0+ | Manipulación de datos |
| numpy | 1.24+ | Operaciones numéricas |
| matplotlib | 3.7+ | Visualizaciones base |
| seaborn | 0.12+ | Visualizaciones estadísticas |
| scipy | 1.10+ | Tests estadísticos |
| scikit-learn | 1.3+ | División train/test |

---

## 👨‍💻 Autor

<div align="center">

**Alejandro De Mendoza**

</div>

### 🎓 Contexto Académico

- **Curso:** Aprendizaje Automático y Minería de Datos
- **Institución:** Fundación Universitaria Internacional de La Rioja
- **Profesor:** Ing. Rogerio Orlando Beltrán Castro
- **Fecha:** Febrero 2026

---

## 🙏 Agradecimientos

> Especial agradecimiento al **Ing. Rogerio Orlando Beltrán Castro** por su guía, conocimientos compartidos y apoyo durante el desarrollo de este proyecto. Sus enseñanzas en análisis exploratorio, preprocesamiento de datos y buenas prácticas en ciencia de datos fueron fundamentales para lograr este resultado.

---

## 📈 Estado del Proyecto

<div align="center">

![Progress](https://img.shields.io/badge/Progress-100%25-success?style=for-the-badge)

| Fase | Estado |
|------|--------|
| ✅ Exploración Inicial | Completado |
| ✅ Análisis Descriptivo | Completado |
| ✅ Análisis de Correlaciones | Completado |
| ✅ Análisis de Distribuciones | Completado |
| ✅ División Train/Test | Completado |
| ⏳ Modelización | Pendiente |
| ⏳ Deploy | Pendiente |

</div>

---

## 💻 Stack Tecnológico

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)

</div>

---

<div align="center">

### 🚴‍♂️ *Hecho con* ❤️ *y mucho* ☕ *en Bogotá, Colombia*

**[⬆ Volver arriba](#-bike-sharing-analytics-dashboard)**

---

*Última actualización: Febrero 2026*

</div>
