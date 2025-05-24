import pandas as pd
from sklearn.model_selection import train_test_split


# Carga los dos datasets
df = pd.read_csv("marzo-abril.csv")
df_2 = pd.read_csv("abril-mayo.csv")

# Filtra el primer dataset para obtener solo los registros donde target == 1
df_filtrado = df[df['target'] == 1]

# Une (concatena) el dataset filtrado con el segundo dataset
df_unido = pd.concat([df_filtrado, df_2], ignore_index=True)

# Mostrar cuántos valores diferentes existen en la columna 'target' del dataset unido
valores_unicos_target = df_unido['target'].unique()
print("Valores únicos en target:", valores_unicos_target)

# Mostrar conteo de cada valor de target
conteo_target = df_unido['target'].value_counts()
print(conteo_target)


import pandas as pd

# Supongamos que df_unido ya está definido

# Filtrar todos los registros con target == 1
df_target_1 = df_unido[df_unido['target'] == 1]

# Filtrar registros con target == 0
df_target_0 = df_unido[df_unido['target'] == 0]

# Muestrear 135 registros aleatorios de target == 0
df_target_0_sample = df_target_0.sample(n=135, random_state=42)

# Concatenar los dos DataFrames
df_final = pd.concat([df_target_1, df_target_0_sample], ignore_index=True)


df_balanceado = pd.concat([df_target_1, df_target_0_sample], ignore_index=True)

# Guardar el dataset balanceado
df_balanceado.to_csv("balanceado.csv", index=False)


# Mostrar conteo de valores únicos en target
print(df_final['target'].value_counts())
