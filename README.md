# Detección de fraude eléctrico con enfoques tabulares y secuenciales con *deep learning*

El objetivo del proyecto es establecer una metodología transparente, estricta y de código abierto para la detección de fraude eléctrico basada en el *dataset SGCC*. Este flujo de trabajo documenta y justifica cada decisión para ofrecer una solución robusta y operativamente viable, priorizando la validez en escenarios reales —eficiencia, xAI e impacto de los tipos de error— frente a la optimización artificial de métricas.

Este repositorio muestra el código que se utiliza para crear un informe o memoria con el contexto del problema, análisis de literatura, etc. Por el momento, este informe no se adjunta.

## 📂 Estructura del Repositorio

El proyecto está centrado en los *notebooks*, que constituyen el eje principal del análisis y la experimentación. En ellos se desarrolla la narrativa completa del trabajo, desde la exploración hasta los resultados.

Los scripts (`.py`) se utilizan como apoyo para mantener los *notebooks* limpios y legibles (encapsulando lógica compleja), y también sirven como punto de partida para reutilizar código o hacer experimentos adicionales.

```text
.
|-- 01_EDA.ipynb
|-- 02_feature_engineering.ipynb
|-- 03_ML_modeling.ipynb
|-- 04_DL_modeling.ipynb
|-- Plots.ipynb                 # Gráficos adicionales para la memoria
|-- config.py                   # Constantes globales (semilla, umbrales)
|-- utils.py                    # Utilidades de limpieza, particionado y optimización del umbral
|-- feature_extraction.py       # Lógica de extracción de variables (propias y TSFEL)
|-- feature_selection.py        # Pipeline de BorutaShap y filtros
|-- modeling.py                 # Optimización (Optuna) y evaluación
|-- requirements.txt
|-- models/                     # Modelos finales y metadatos (ML/ y DL/)
|-- data/                       # Dataset original, extracciones, variables filtradas, hiperparámetros
   --> data NO SE PUBLICA POR PESO (1.67Gb)
```

## Flujo del proyecto

| Notebook | Contenido |
|---|---|
| `01_EDA.ipynb` | Análisis exploratorio: estructura temporal, desbalance, significado de ceros y nulos, outliers, estacionalidad y separabilidad |
| `02_feature_engineering.ipynb` | Explica la lógica y el flujo del preprocesamiento, extracción de variables y selección con *BorutaShap* |
| `03_ML_modeling.ipynb` | Conjunto óptimo, *Resampling*, Comparativa LightGBM / XGBoost / CatBoost / ensamblados, optimización con Optuna (TPE), explicabilidad SHAP |
| `04_DL_modeling.ipynb` | CNN 1D multiescala con atención temporal sobre series crudas, xAI por pesos de atención y oclusión |

## 📊 Resultados Principales

| Métrica | XGBoost tabular | CNN 1D multi-escala + atención* |
|---|---:|---:|
| PR-AUC | 0.545 ± 0.006 | **0.684 ± 0.014** |
| ROC-AUC | 0.865 ± 0.002 | **0.907 ± 0.009** |
| MCC | 0.464 ± 0.003 | **0.595 ± 0.012** |
| Precision_Fraud | 0.575 ± 0.046 | **0.767 ± 0.051** |
| Recall_Fraud | 0.446 ± 0.044 | **0.509 ± 0.057** |
| Precision_NoFraud | 0.948 ± 0.003 | **0.954 ± 0.005** |
| Recall_NoFraud | 0.967 ± 0.009 | **0.984 ± 0.007** |

>*\*CNN 1D multi-escala con ramas convolucionales paralelas, pooling global mixto (avg/max) y mecanismo de atención temporal (soft attention)*

El modelo profundo se sitúa en la frontera del estado del arte, superando al tabular en todas las métricas, con una mejora especialmente notable en la precisión. Su notable mejora en Precision_Fraud (reduciendo falsas alarmas) lo convierte en el modelo más eficiente para el despliegue de inspecciones.

## Cómo reproducir

```bash
# 1. Crear entorno con Python 3.12
# 2. Instalar dependencias
pip install -r requirements.txt
# 3. Ejecutar los notebooks en orden
```

## Referencias

- Keany, E. (2020). *Boruta-Shap* (v1.1). GitHub: https://github.com/Ekeany/Boruta-Shap
- Zheng, Z. et al. (2018). *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids*. IEEE TII, 14(4). GitHub: https://github.com/henryRDlab/ElectricityTheftDetection
