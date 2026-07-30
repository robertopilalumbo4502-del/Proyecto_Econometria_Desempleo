import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc
import os

print("--- GENERANDO GRÁFICOS ECONOMÉTRICOS ---")

# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('outputs', exist_ok=True)

# 1. Cargar datos y re-estimar modelos para curva ROC
df = pd.read_csv('data/enemdu_desempleo_clean.csv')
formula = 'desempleado ~ educacion + edad + sexo + area'

logit_mod = smf.logit(formula, data=df).fit(disp=0)
probit_mod = smf.probit(formula, data=df).fit(disp=0)

y_true = df['desempleado']
y_pred_logit = logit_mod.predict(df)
y_pred_probit = probit_mod.predict(df)

# -------------------------------------------------------------
# Gráfico 1: Curva ROC (Comparación Logit vs Probit)
# -------------------------------------------------------------
fpr_logit, tpr_logit, _ = roc_curve(y_true, y_pred_logit)
fpr_probit, tpr_probit, _ = roc_curve(y_true, y_pred_probit)

auc_logit = auc(fpr_logit, tpr_logit)
auc_probit = auc(fpr_probit, tpr_probit)

plt.figure(figsize=(8, 6))
plt.plot(fpr_logit, tpr_logit, color='blue', lw=2, label=f'Logit (AUC = {auc_logit:.3f})')
plt.plot(fpr_probit, tpr_probit, color='red', lw=2, linestyle='--', label=f'Probit (AUC = {auc_probit:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Clasificador Aleatorio (AUC = 0.500)')

plt.title('Evaluación Predictiva: Curva ROC (Logit vs. Probit)', fontsize=12, fontweight='bold')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/curva_roc_comparativa.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# Gráfico 2: Probabilidad de Desempleo según Escolaridad
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.regplot(x='educacion', y='desempleado', data=df, logistic=True, 
            ci=None, scatter_kws={'alpha':0.1, 'color':'gray'}, line_kws={'color':'darkblue', 'linewidth':2})

plt.title('Probabilidad Predicha de Desempleo vs. Años de Escolaridad', fontsize=12, fontweight='bold')
plt.xlabel('Años de Escolaridad')
plt.ylabel('Probabilidad de estar Desempleado')
plt.tight_layout()
plt.savefig('outputs/probabilidad_escolaridad.png', dpi=300)
plt.close()

print("✅ Gráficos generados y guardados exitosamente en la carpeta 'outputs/':")
print("   - outputs/curva_roc_comparativa.png")
print("   - outputs/probabilidad_escolaridad.png")