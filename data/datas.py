import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar dataset balanceado
df_balanceado = pd.read_csv("balanceado.csv")


# Dividir en 80% train y 20% temp (validación + prueba), con estratificación por target
df_train, df_temp = train_test_split(df_balanceado, test_size=0.2, random_state=42, stratify=df_balanceado['target'])

# Dividir temp en 50% validación y 50% prueba (cada uno 10% total)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['target'])

# Guardar datasets en CSV
df_train.to_csv("balanceado_train.csv", index=False)
df_val.to_csv("balanceado_validation.csv", index=False)
df_test.to_csv("balanceado_test.csv", index=False)

# Mostrar tamaños y conteo por target para verificar balance
print("Tamaño entrenamiento:", len(df_train))
print("Tamaño validación:", len(df_val))
print("Tamaño prueba:", len(df_test))

print("\nConteo target en entrenamiento:\n", df_train['target'].value_counts())
print("\nConteo target en validación:\n", df_val['target'].value_counts())
print("\nConteo target en prueba:\n", df_test['target'].value_counts())
