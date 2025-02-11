from matplotlib.pylab import set_printoptions
import pandas as pd
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

# De la anterior sabemos que estan balanceados. 
#Obtenemos path
csv_file_path = 'task2.2\\high_diamond_ranked_10min.csv'
df = pd.read_csv(csv_file_path)

# ============== PRIMERA TECNICA PCA ==============


#No obtenemos el primero ya que es el gameID , ni el segundo ya que es nuestra variable objetivo
X = df.iloc[:, 2:21] 
y = df.iloc[:, 1]


kmo,kmo_modelo = calculate_kmo(X)
print(kmo_modelo)

# El kmodelo es mayor a 0.68 esto se usa para determinar la correlacion
# de variables por lo que podemos usar PCA

# =========================== PCA ===========================
pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=7))
pca_pipe.fit(X)

modelo_pca = pca_pipe.named_steps['pca']

pca = pd.DataFrame(
    data=modelo_pca.components_,
    columns=X.columns,
    index=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
)

print(pca.head())


# Dado esto vamos a obtener la varianza para ver cuantos datos son explicados

print(modelo_pca.explained_variance_ratio_)


# Como podemos ver con los primeros 4 PCA se explican el 66% de los datos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

scaler = StandardScaler()
pca = PCA(n_components=7)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Hacemos tunning
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_val, y_val)
#Imprimos los hiperparametros.
print(grid.best_params_) 

# ENTRENAMOS CON EL PCA
clf = svm.SVC(C=1, gamma=0.01, kernel='rbf')
clf.fit(X_train, y_train)


# TESTEAMOS CON EL PCA
y_predic_lib = clf.predict(X_test)
f1_score_pca = f1_score(y_test, y_predic_lib, average='macro')
print(f1_score_pca)
# Nos dio 0.7190


# ============== SEGUNDA TECNICA EXTRA TREE ==============



# Entrenamos el modelo
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, y)

# Obtenemos las importancias
importances = model.feature_importances_

# Creamos un DataFrame de importancias
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Ordenamos de mayor a menor
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Esta sera la importancia acumulada
importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()

# Imprimimos las características ordenadas con su importancia y acumulada
print(importance_df)

# Ahora ordenamos las columnas de X según la importancia
X_sorted = X[importance_df['Feature']]

X_selected = X_sorted.iloc[:, :1]


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_selected, y, test_size=0.2, random_state=42)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train2, y_train2, test_size=0.1, random_state=42)



# Hacemos tunning
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_val2, y_val2)
#Imprimos los hiperparametros.
print(grid.best_params_) 


# Entrenamos de nuevo
# ENTRENAMOS CON IMPORTANCIA
clf = svm.SVC(C=1, gamma=0.0001, kernel='rbf')
clf.fit(X_train2, y_train2)


# TESTEAMOS CON IMPORTANCIA
y_predic_lib2 = clf.predict(X_test2)
f1_score_pca = f1_score(y_test2, y_predic_lib2, average='macro')
print(f1_score_pca)
# Nos dio 0.71649434351695


#====================== ANOVA ALGORITMO =============================

test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, y)
set_printoptions(precision=3)
print(fit.scores_)

scores_df = pd.DataFrame({
    'Feature': X.columns,
    'Scores': fit.scores_
})

scores_df = scores_df.sort_values(by='Scores', ascending=False)

print(scores_df)

X_sorted = X[scores_df['Feature']]

print(X_sorted.head())


X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2, random_state=42)
X_train3, X_val3, y_train3, y_val3 = train_test_split(X_train3, y_train3, test_size=0.1, random_state=42)

selector = SelectKBest(f_classif, k=2) 
selector.fit(X_train3, y_train3)

X_train_selected = selector.transform(X_train3) 
X_test_selected = selector.transform(X_test3)
X_validator_selected = selector.transform(X_val3)

param_grid = {'C': [0.1, 1, 10, 100, 1000, 2000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 
              'kernel': ['rbf']}  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_validator_selected, y_val3)


print(grid.best_params_) 


# ENTRENAMOS CON ANOVA
clf = svm.SVC(C=1, gamma=0.00001, kernel='rbf')
clf.fit(X_train_selected, y_train3)


# TESTEAMOS CON ANOVA
y_predic_lib3 = clf.predict(X_test_selected)
f1_score_pca = f1_score(y_test3, y_predic_lib3, average='macro')
print(f1_score_pca)
# Nos dio 0.71649434351695