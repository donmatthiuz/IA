import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

#Cargar el archivo CSV
csv_file_path = 'bank_transactions.csv'
df = pd.read_csv(csv_file_path)


print("📌 Primeras filas del dataset:")
print(df.head())


print("\n📌 Información general del dataset:")

print(df.info())


print("\n📌 Estadísticas descriptivas:")
print(df.describe())


print("\n📌 Valores nulos en cada columna:")
print(df.isnull().sum())

# Mostrar el número de filas y columnas
print(f"Número de filas: {df.shape[0]}")


# #Vamos a eliminar los que son nulos de 

if 'CustGender' in df.columns and 'CustLocation' in df.columns:
    df.dropna(subset=['CustGender', 'CustLocation', 'CustomerDOB'], inplace=True)


def fix_birth_year(date_str):
    if pd.isnull(date_str):
        return np.nan 
    
    # Dividir el string por las barras
    parts = date_str.split('/')
    
    # Asegurarse de que tiene tres partes
    if len(parts) == 3:
        day, month, year = parts
        
        # Corregir año si tiene 2 dígitos
        if len(year) == 2:
            corrected_year = f'20{year}' if int(year) < 25 else f'19{year}'
            return f'{day}/{month}/{corrected_year}'
    
    return date_str  # Si no se puede corregir, devolver la fecha original




if 'CustomerDOB' in df.columns:
    df['CustomerDOB'] = df['CustomerDOB'].astype(str).apply(fix_birth_year)  # Corregir años

    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], format='%d/%m/%Y', errors='coerce')

   
    # Calculamos la edad
    df['Age'] = df['CustomerDOB'].apply(lambda x: (pd.Timestamp.now().year - x.year) if pd.notnull(x) else np.nan)

    print(df[['CustomerID', 'CustomerDOB', 'Age']])
    df.drop(columns=['CustomerDOB'], inplace=True)
    print(df[['CustomerID', 'Age']])


# Verificar los valores nulos en la columna CustomerDOB



# Rellenar CustAccountBalance con la mediana
if 'CustAccountBalance' in df.columns:
    df['CustAccountBalance'].fillna(df['CustAccountBalance'].median(), inplace=True)




# # ECODING


# Convertimos genero a numeros
df['CustGender'], unique_values = pd.factorize(df['CustGender'])
print("Valores únicos y su asignación numérica:")
print(dict(enumerate(unique_values)))


# Convertimos localizacion a numeros

df['CustLocation'], location_unique_values = pd.factorize(df['CustLocation'])
print("\nValores únicos de 'CustLocation' y su asignación numérica:")
# print(dict(enumerate(location_unique_values)))


# Convertir 'TransactionDate' a datetime (si no está ya en formato datetime)
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')

# Ahora con los años
df['TransactionYear'] = df['TransactionDate'].dt.year
df['TransactionMonth'] = df['TransactionDate'].dt.month
df['TransactionDay'] = df['TransactionDate'].dt.day
df['TransactionWeekday'] = df['TransactionDate'].dt.weekday  # Día de la semana (0=lunes, 6=domingo)
print(df[['TransactionDate', 'TransactionYear', 'TransactionMonth', 'TransactionDay', 'TransactionWeekday']].head())


df.to_csv("bank_transactions_cleaned.csv", index=False)

csv_file_path = 'bank_transactions_cleaned.csv'
df = pd.read_csv(csv_file_path)

print("\n📌 Valores nulos en cada columna:")
print(df.isnull().sum())