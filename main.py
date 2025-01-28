import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from Regresion_logistica import Logistica
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#=================PARTE 1 ANALISIS Y LIMPIEZA DATASET========================
file_path = "./dataset_phishing.csv"
dataset = pd.read_csv(file_path)



print("==================ANALISIS EXPLORATORIO===============")
length_url = dataset['length_url']
random_domain_unique = dataset['random_domain'].unique()
port_unique = dataset['port'].unique()
dns_record_unique = dataset['dns_record'].unique()
login_form_unique = dataset['login_form'].unique()
google_index_unique = dataset['google_index'].unique()


dataset_count = len(dataset)


mean_values = dataset[['length_url','random_domain', 'port', 'dns_record', 'login_form', 'google_index']].mean(numeric_only=True)
median_values = dataset[['length_url','random_domain', 'port', 'dns_record', 'login_form', 'google_index']].median(numeric_only=True)
mode_values = dataset[['length_url','random_domain', 'port', 'dns_record', 'login_form', 'google_index']].mode().iloc[0]

print("\nRango para 'length_url':")
print(f"{length_url.min()} - {length_url.max()}"  )

print("\nValores únicos para 'random_domain':")

print(random_domain_unique)
print("\nValores únicos para 'port':")

print(port_unique)

print("\nValores únicos para 'dns_record':")
print(dns_record_unique)

print("\nValores únicos para 'login_form':")
print(login_form_unique)

print("\nValores únicos para 'google_index':")
print(google_index_unique)

print(f"\nConteo total de filas en el dataset: {dataset_count}")

print("\nMedia de las columnas numéricas:")
print(mean_values)

print("\nMediana de las columnas numéricas:")
print(median_values)

print("\nModa de las columnas:")
print(mode_values)


# No tienen datos nulos no es necesario hacer una limpieza de datos. 

print("==================BALANCEO===============")



# Ya estan balanceados tenemos los mismos numeros en casos phising y legitimos. 
legitimos = dataset.loc[dataset['status'] == 'legitimate']
phising = dataset.loc[dataset['status'] == 'phishing']
print("Legitimos",legitimos['status'].count())
print("Phising", phising['status'].count())

# Vamos a normalizar el length url
min_length = dataset['length_url'].min()
max_length = dataset['length_url'].max()

dataset['length_url'] = (dataset['length_url'] - min_length)/(max_length - min_length)


#Ahora vamos a convertir nuestro status a un valor numerico
# PHISING 1 y legitimate 0
dataset['status_numeric'] = dataset['status'].map({'phishing': 1, 'legitimate': 0})





#=====================PARTE 2 IMPLEMENTACIION DE ALGORITMO REGRESION A MANO==============

#Vamos a dividir en 2 test y entrenamiento

X = dataset[['length_url','google_index']].values
Y = dataset['status_numeric'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

unique, counts = np.unique(y_train, return_counts=True)
for value, count in zip(unique, counts):
    print(f"Valor único: {value}, Cantidad: {count}")
    
# No es necesario hacer ajustes


#Variables a Definir
w = np.zeros(X_train.shape[1])  
b = 0.4
learning_rate = 0.5
epocas = 7000

regresion = Logistica(learning_rate=learning_rate, epocas=epocas)

W, B = regresion.algoritmo_sin_librerias(X=X_train, Y=y_train, w=w, b=b)
print(W ,B)

#Ahora vamos a predecir

y_prediccion = 1 / (1 + np.exp(-(np.dot(X_test, W) + B)))
y_predichas = [1 if val >= 0.5 else 0 for val in y_prediccion]

accuracy = accuracy_score(y_test, y_predichas)
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(y_test, y_predichas))