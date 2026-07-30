import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
import json
import os

print("--- INICIANDO ESTIMACIÓN ECONOMÉTRICA (LOGIT VS PROBIT) ---")

# 1. Cargar datos limpios
df = pd.read_csv('data/enemdu_desempleo_clean.csv')

# 2. Definir la fórmula econométrica
formula = 'desempleado ~ educacion + edad + sexo + area'

# 3. Estimación del Modelo Logit
logit_mod = smf.logit(formula, data=df).fit()
logit_mfx = logit_mod.get_margeff(at='overall') # Corregido: get_margeff

# 4. Estimación del Modelo Probit
probit_mod = smf.probit(formula, data=df).fit()
probit_mfx = probit_mod.get_margeff(at='overall') # Corregido: get_margeff

# 5. Evaluación Predictiva (AUC-ROC)
y_true = df['desempleado']
logit_auc = roc_auc_score(y_true, logit_mod.predict(df))
probit_auc = roc_auc_score(y_true, probit_mod.predict(df))

print("\n================ RESUMEN MODELO LOGIT ================")
print(logit_mod.summary())
print("\n--- Efectos Marginales Logit ---")
print(logit_mfx.summary())

print("\n================ RESUMEN MODELO PROBIT ================")
print(probit_mod.summary())
print("\n--- Efectos Marginales Probit ---")
print(probit_mfx.summary())

# 6. Guardar métricas comparativas en JSON para Vercel/Dashboard
resultados = {
    "proyecto": "Determinantes del Desempleo Urbano en Ecuador",
    "modalidad": "Modalidad A - Respuesta Binaria",
    "observaciones": int(df.shape[0]),
    "modelos": {
        "logit": {
            "aic": float(logit_mod.aic),
            "bic": float(logit_mod.bic),
            "pseudo_r2": float(logit_mod.prsquared),
            "auc": float(logit_auc),
            "log_likelihood": float(logit_mod.llf)
        },
        "probit": {
            "aic": float(probit_mod.aic),
            "bic": float(probit_mod.bic),
            "pseudo_r2": float(probit_mod.prsquared),
            "auc": float(probit_auc),
            "log_likelihood": float(probit_mod.llf)
        }
    }
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/resultados_modelos.json', 'w') as f:
    json.dump(resultados, f, indent=4)

print("\n✅ Métricas comparativas guardadas exitosamente en 'outputs/resultados_modelos.json'")