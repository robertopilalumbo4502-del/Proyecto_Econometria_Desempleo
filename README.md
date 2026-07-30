# Determinantes del Desempleo Urbano en Ecuador (Logit vs. Probit)

Este repositorio contiene el código y los artefactos del proyecto final de Econometría, enfocado en estimar y comparar modelos de respuesta binaria (**Logit** y **Probit**) para determinar la probabilidad de desempleo en Ecuador utilizando datos de la ENEMDU.

## 📁 Estructura del Proyecto

- `data/`: Dataset procesado y limpio (`enemdu_desempleo_clean.csv`).
- `src/`: Scripts modulares de Python.
  - `01_descarga_datos.py`: Procesamiento y limpieza de la base de datos.
  - `02_modelo_econometrico.py`: Estimación econométrica y efectos marginales.
  - `03_visualizacion.py`: Generación de curvas ROC y gráficos de probabilidad.
- `outputs/`: Resultados econométricos en JSON (`resultados_modelos.json`) y gráficos exportados.
- `prompts/`: Bitácora y registro de uso de Inteligencia Artificial (`registro_uso_ia.md`).

## 🛠️ Tecnologías Utilizadas
- **Lenguaje:** Python 3.x
- **Librerías:** `pandas`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`
- **Control de Versiones:** Git & GitHub

## 👤 Autor
- **Roberto Pilalumbo** - Universidad Técnica de Cotopaxi (UTC)