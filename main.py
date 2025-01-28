import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
# PHISING 0 y legitimate 1
dataset['status_numeric'] = dataset['status'].map({'phishing': 0, 'legitimate': 1})





#=====================PARTE 2 IMPLEMENTACIION DE ALGORITMO REGRESION A MANO==============

#Ahora vamos a graficar la funcion. 



# x = np.linspace(0,dataset['length_url'].max(),100)
# y = 1/(1+np.exp(-(w*x+b)))

# grafica de la recta
# dataset.plot.scatter(x='length_url',y='status_numeric')
# plt.plot(x, y, 'r')
# plt.ylim(0,dataset['status_numeric'].max()*1.1)
# # plt.grid()
# plt.show()

# Ahora vamos a obtener el trainloss y el promedio de perdida

# # calculo de las predicciones
# dataset['sigmoid'] = 1/(1+np.exp(-(dataset['length_url']*w+b)))

# # calculo de la funcion de error
# dataset['loss_xi'] = -dataset['status_numeric']*np.log(dataset['sigmoid'])-(1-dataset['status_numeric'])*np.log(1-dataset['sigmoid'])
# cost_j = dataset['loss_xi'].mean()
# print(cost_j)


X_train = dataset[['length_url','dns_record']].values
Y_train = dataset['status_numeric'].values


#Variables a Definir
w = np.zeros(X_train.shape[1])  
b = 0
learning_rate = 0.01
epocas = 1000