import pandas as pd

# Ruta del archivo CSV
file_path = "./dataset_phishing.csv"

# Cargar el archivo CSV en un DataFrame
dataset = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset
print(dataset.head())
