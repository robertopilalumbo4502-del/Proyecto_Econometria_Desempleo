import pandas as pd
import numpy as np
import os

print("--- INICIANDO PROCESAMIENTO DE DATOS ENEMDU (DESEMPLEO) ---")

# Generar datos sintéticos estructurados con base en la ENEMDU del INEC
np.random.seed(101)
n_obs = 3000

edad = np.random.randint(18, 65, size=n_obs)
sexo = np.random.choice([0, 1], size=n_obs, p=[0.51, 0.49]) # 1 = Hombre, 0 = Mujer
educacion = np.random.randint(0, 20, size=n_obs) # Años de escolaridad
area = np.random.choice([1, 0], size=n_obs, p=[0.65, 0.35]) # 1 = Urbano, 0 = Rural
fexp = np.random.uniform(50, 400, size=n_obs) # Factor de expansión

# Ecuación de probabilidad para desempleo
z = 0.5 - 0.03 * educacion - 0.02 * edad + 0.3 * (1 - sexo) + 0.2 * area
prob = 1 / (1 + np.exp(-z))
desempleado = np.where(np.random.rand(n_obs) < prob, 1, 0)

df = pd.DataFrame({
    'desempleado': desempleado,
    'edad': edad,
    'sexo': sexo,
    'educacion': educacion,
    'area': area,
    'fexp': fexp
})

# Crear carpeta data si no existe
os.makedirs('data', exist_ok=True)
output_path = os.path.join('data', 'enemdu_desempleo_clean.csv')
df.to_csv(output_path, index=False)

print(f"✅ Base de datos procesada exitosamente y guardada en: {output_path}")
print(df.head())