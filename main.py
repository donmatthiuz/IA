import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from regresion_logistica import Logistica
from validacion import f1_score
from matplotlib.colors import ListedColormap as Colors


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

y_predichas_mimodelo = regresion.algoritmo_sin_librerias(X=X_train, Y=y_train, w=w, b=b, X_test=X_test)
y_predichas_modelo_sklearn= regresion.algoritmo_con_librerias(X_train=X_train, Y_train=y_train, X_test=X_test)
#Ahora vamos a predecir

print("F1 SECORE METRICAS")
print("Mi modelo:",f1_score(ytest=y_test, ypredichas=y_predichas_mimodelo))
print("Modelo de SKLEARN:",f1_score(ytest=y_test, ypredichas=y_predichas_modelo_sklearn))



# Ahora vamos a hacer el diagrama
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))


plt.contourf(X1, X2, np.array(regresion.algoritmo_sin_librerias(X_test=np.array([X1.ravel(), X2.ravel()]).T, X=X_train, Y=y_train, w=w, b=b)).reshape(X1.shape),
             alpha = 0.5, cmap = Colors(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = Colors(('red', 'green'))(i), label = j, s=10)
plt.title('Regresion Logistica ')
plt.xlabel('Largo de la URL')
plt.ylabel('Indexado en Google')
plt.show()