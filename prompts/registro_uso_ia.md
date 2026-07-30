# Registro de Uso de Inteligencia Artificial

**Proyecto:** Determinantes del Desempleo Urbano en Ecuador  
**Modalidad:** Modalidad A (Modelos de Respuesta Binaria: Logit vs. Probit)  
**Herramienta de IA utilizada:** Gemini (Google)  

---

## Declaración de Autoría y Transparencia
El presente proyecto econométrico utilizó herramientas de inteligencia artificial generativa como asistencia técnica para:
1. La estruturación de scripts de Python modularizados (`pandas`, `statsmodels`, `scikit-learn`).
2. La automatización del procesamiento de datos y exportación de métricas a formato `.json`.
3. El diseño de gráficos econométricos (curvas ROC y probabilidades predichas).
4. La revisión conceptual de la teoría detrás de los modelos Logit y Probit.

Todos los resultados, códigos y redactados fueron revisados, ejecutados y validados manualmente en el entorno de desarrollo local.

---

## Registro de Prompts Principales

### Prompt 1: Generación y preparación de datos
* **Consulta:** "Necesito un script en Python para estructurar la base de datos ENEMDU del INEC enfocada en desempleo, incluyendo variables explicativas de escolaridad, edad, sexo y área."
* **Uso:** Código utilizado en `src/01_descarga_datos.py`.

### Prompt 2: Estimación econométrica
* **Consulta:** "Genera un código en Python usando statsmodels para estimar e interpretar un modelo Logit y un Probit con la misma ecuación, obteniendo AIC, BIC, Pseudo R2 y efectos marginales promedio."
* **Uso:** Código utilizado en `src/02_modelo_econometrico.py`.

### Prompt 3: Evaluación predictiva
* **Consulta:** "Escribe el código para graficar la curva ROC comparativa de Logit vs Probit y la probabilidad predicha según escolaridad."
* **Uso:** Código utilizado en `src/03_visualizacion.py`.