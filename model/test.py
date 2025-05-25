import pandas as pd

# Carga el archivo CSV
df = pd.read_csv("../data/abril-mayo.csv")

# Solo columnas numéricas
numericas = df.select_dtypes(include=['float64', 'int64'])

# Calcular la correlación con la variable target
correlaciones = numericas.corr()['target'].drop('target')

# Mostrar las variables con alta colinealidad (valor absoluto > 0.7 como umbral)
alta_colinealidad = correlaciones[correlaciones.abs() > 0.7]
print("Variables con alta colinealidad con 'target':\n", alta_colinealidad)
# Casos donde target indica lluvia y sí hubo precipitación
lluvia_real = df[(df['target'] == 1) & (df['precipitacion'] > 0)]

# Casos donde target indica lluvia pero no hubo precipitación
falsos_positivos = df[(df['target'] == 1) & (df['precipitacion'] <= 0)]

print("Lluvia real (target=1 y precipitación>0):", len(lluvia_real))
print("Falsos positivos (target=1 y precipitación<=0):", len(falsos_positivos))

