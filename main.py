import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# === 1. Cargar datos ===
df = pd.read_csv("clima_guatemala.csv")

# === 2. Definir las variables (features) ===
features = ['year', 'mes', 'dia', 'franja_horaria', 'temperatura', 'humedad',
            'viento_vel_m_s', 'viento_dir', 'presion', 'precipitacion', 'nubosidad']

df = df[features]

# === 3. Eliminar filas con valores nulos ===
df = df.dropna()

# === 4. Codificar 'franja_horaria' (variable categórica) ===
df = pd.get_dummies(df, columns=['franja_horaria'], drop_first=True)

# === 5. Separar en 80% train, 10% validación, 10% test ===

# Primero 80% train y 20% restante
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)

# Luego dividir 20% restante en 10% validación y 10% test
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# === 6. Mostrar tamaños y una muestra de cada conjunto ===
print("Tamaño del set de entrenamiento:", len(df_train))
print("Tamaño del set de validación:", len(df_val))
print("Tamaño del set de prueba:", len(df_test))