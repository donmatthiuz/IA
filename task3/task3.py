import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support,  f1_score
from Tunning import Tunning
from SVM import SVM_manual
from grafica import graph_smv

# Reemplazamos por el path actual
csv_file_path = 'task3\\high_diamond_ranked_10min.csv'
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
  



# ================== PARTE 3.1  ANALISIS EXPLORATORIO ===================

# Cargamos el dataframe
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

# Vamos a usar estas 2 debido a que se intuye que tener un nivel bueno entre los jugadores
# Y tambien al tener muchas muertes causadas pueden provocar que gananen la partida


#Mostramos el rango de los valores
print(f"Promedio de nivel de los Azules {df["blueAvgLevel"].min()} - {df["blueAvgLevel"].max()}")
print(f"Muertes causadas por los Azules {df["blueKills"].min()} - {df["blueKills"].max()}")


# No es necesario escalarlas ni mapearlas , tampoco normalizarlas. Por lo que se usara como estan

#=====================PARTE 3.2 Support Vector Machines  Clasificación de Partidas de League of Legends==============

#Vamos a dividir en 3 test , entrenamiento y 

X = df[['blueAvgLevel','blueKills']].values
Y = df['blueWins'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# =================== TUNING 10% ==============================
# PRIMERO CON LOS 10% de los datos vamos a hacer Tunnig en ambos casos

# Con el modelo manual usaremos una clase llamada tunnig
#   tunn = Tunning(X_val=X_val, X_Train=X_train, Y_train=y_train, y_val=y_val)
#   tunn.encontrar_variables()

# Estan desabilitadas ya que se tardan un monton en encontrar pero se encontro que
#   lr: 0.001, ep: 2000, lambda: 0.01 -> fue el mejor al tener un acurrancy de 0.6775

# Con el modelo de la libreria usaremos la libreria grid
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
y_val_Trained = np.where(y_val <= 0, -1, 1)
# fitting the model for grid search 
grid.fit(X_val, y_val) 

#iprimos los hiperparametros.
print(grid.best_params_) 


#======================= ENTRENAMIENTO 80% ===========================

# Ahora lo entrenaremos usando los mejores parametros encontrados antes

# Manual
learning_rate = 0.001
epocas = 1000
lambda_par =0.01

svm_manual = SVM_manual(learning_rate=learning_rate, epocas=epocas, lambda_par=lambda_par)
svm_manual.algoritmo_sin_librerias_fit(X=X_train, Y=y_train)

# Con librerias
y_trained_parametrized = np.where(y_train <= 0, -1, 1 )
clf = svm.SVC(C=1000, gamma=0.01, kernel='rbf')
clf.fit(X_train, y_trained_parametrized)


# ================ 10% TESTING =============================
#ahora hacemos el testing y usaremos f1score para comparar despues entre las 2. 
y_test_parametrized = np.where(y_test <= 0, -1, 1)

# Manual
y_predic_manual = svm_manual.algoritmo_sin_librerias_prediccion(X_test=X_test)
f1_score_manual = f1_score(y_test_parametrized, y_predic_manual, average='macro')

# Por codigo
y_predic_lib = clf.predict(X_test)
f1_score_lib = f1_score(y_test_parametrized, y_predic_lib, average='macro')


# Como resultado nos da
print("F1 SCORE MANUAL: ",f1_score_manual)
print("F1 SCORE LIBRERIA: ",f1_score_lib)


# ============  GRAFICAR ===================
graph_smv(data_X=X_train, labels_y=y_trained_parametrized, model=svm_manual)



