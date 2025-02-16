import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# Cargar los datos
csv_file_path = 'bank_transactions_cleaned.csv'
df = pd.read_csv(csv_file_path)
print("📌 Datos cargados correctamente.")


#=========================== SIN PCA =========================
# Seleccionar las variables relevantes
df_subset = df[['CustGender', 
                'CustLocation', 
                'CustAccountBalance',
                'TransactionTime',
                'TransactionAmount (INR)',
                'Age',
                'TransactionYear',
                'TransactionMonth',
                'TransactionDay',
                'TransactionWeekday']]


# Dividir en entrenamiento, validación y prueba
X_temp, X_test = train_test_split(df_subset, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.1, random_state=42)



# Validación del número óptimo de clusters
n_components = np.arange(1, 10)



models = []
bic_scores = []
aic_scores = []

for n in n_components:
    print(f"Entrenando modelo con {n} componentes...")
    model = GaussianMixture(n_components=n, random_state=42).fit(X_val)
    models.append(model)
    bic_scores.append(model.bic(X_val))
    aic_scores.append(model.aic(X_val))
    print(f"Modelo con {n} componentes - BIC: {bic_scores[-1]}, AIC: {aic_scores[-1]}")

# Graficar los resultados
plt.plot(n_components, bic_scores, label='BIC', marker='o')
plt.plot(n_components, aic_scores, label='AIC', marker='s')
plt.legend()
plt.xlabel("Número de componentes")
plt.ylabel("Puntaje")
plt.title("Selección del número óptimo de componentes")
plt.show()


#Ahora sabemos que 5 viendo el grafico es el numero de clusters 
#necesarios asi que vamos a utilizar 5 clusters

# #===================== ENTRENAMIENTO ==================

model = GaussianMixture(n_components=5,
                        random_state=42).fit(X_train)



# #======================= TESTEO =========================
cluster = pd.Series(model.predict(X_test))


sil_score = silhouette_score(X_test, cluster)
print(f"Silhouette Score: {sil_score}")


# ====================== CON PCA =============================

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_subset)


pca = PCA()
pca.fit(df_scaled)



explained_variance = np.cumsum(pca.explained_variance_ratio_)


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', color='b')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada por los Componentes Principales')
plt.grid(True)
plt.show()


optimal_components = np.argmax(explained_variance >= 0.90) + 1
print(f"El número óptimo de componentes para explicar el 90% de la varianza es: {optimal_components}")

# Ahora usaremos 8 PCA
pca = PCA(n_components=8)
X_pca = pca.fit_transform(df_scaled)

X_temp, X_test = train_test_split(X_pca, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.1, random_state=42)

n_components = np.arange(1, 10)



models = []
bic_scores = []
aic_scores = []

for n in n_components:
    print(f"Entrenando modelo con {n} componentes...")
    model = GaussianMixture(n_components=n, random_state=42).fit(X_val)
    models.append(model)
    bic_scores.append(model.bic(X_val))
    aic_scores.append(model.aic(X_val))
    print(f"Modelo con {n} componentes - BIC: {bic_scores[-1]}, AIC: {aic_scores[-1]}")

# Graficar los resultados
plt.plot(n_components, bic_scores, label='BIC', marker='o')
plt.plot(n_components, aic_scores, label='AIC', marker='s')
plt.legend()
plt.xlabel("Número de componentes")
plt.ylabel("Puntaje")
plt.title("Selección del número óptimo de componentes")
plt.show()


# Podemos ver que el numero de clusters es de 6 para nuestros PCA


model = GaussianMixture(n_components=6, random_state=42).fit(X_train)

cluster = pd.Series(model.predict(X_test))

# Evaluar el rendimiento del clustering con Silhouette Score
sil_score = silhouette_score(X_test, cluster)
print(f"Silhouette Score (con PCA): {sil_score}")

# Ahora los graficamos coon los PCA 1 y 2
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=cluster, cmap='viridis', s=100, alpha=0.7)
plt.title('Clusters obtenidos con PCA y GMM')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()