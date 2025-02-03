import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from SVM import SVM, SVM2


# Replace the path with your actual file path
csv_file_path = 'task3\\high_diamond_ranked_10min.csv'



# ================== PARTE 3.1  ANALISIS EXPLORATORIO ===================

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file_path)


## Primero vamos a verificar que este balanceado
ceros = df.loc[df['blueWins'] == 0]
unos = df.loc[df['blueWins'] == 1]

#Se puede ver que estan balanceados pues blueWins tiene 4949 y 4930 de ceros y unos
print(ceros["blueWins"].count())
print(unos["blueWins"].count())



# Eleccion de variables 

# Como variables vamos a ver cuales hay. 

# Eligiremos el blueAvgLevel
# Eligiremos el blueKills

# Vamos a usar estas 2 debido a que se intuye que tener un nivel 
# en el promedio de los azules y las muertes provocadas que puede 


#Mostramos el rango de los valores
print(f"Promedio de nivel de los Azules {df["blueAvgLevel"].min()} - {df["blueAvgLevel"].max()}")
print(f"Muertes causadas por los Azules {df["blueKills"].min()} - {df["blueKills"].max()}")


# No es necesario escalarlas ni mapearlas , tampoco normalizarlas. Por lo que se usara como estan

#=====================PARTE 3.2 Support Vector Machines  Clasificación de Partidas de League of Legends==============

#Vamos a dividir en 2 test y entrenamiento

X = df[['blueAvgLevel','blueKills']].values
Y = df['blueWins'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# ENTRENAMIENTO 80%

#Variables a Definir
learning_rate = 0.001
epocas = 1000
lambda_par =0.01

svm = SVM(learning_rate=learning_rate, epocas=epocas, lambda_par=lambda_par)

svm.algoritmo_sin_librerias_fit(X=X_train, Y=y_train)

y_manual_predichas = svm.algoritmo_sin_librerias_prediccion(X_test=X_test)

#y_librerias_predichas = svm.algoritmo_con_librerias(X_train=X_train, Y_train=y_train, X_test= X_test)
#print(y_manual_predichas)

# svm2 = SVM2()

# svm2.fit(X_train, y_train)
# ys = svm2.predict(X_test)
print(y_manual_predichas)






