import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Ruta del archivo CSV
csv_file_path = 'bank_transactions_cleaned.csv'

# Definir las columnas
column_names = [
    'TransactionID', 'CustomerID', 'CustGender', 'CustLocation', 'CustAccountBalance',
    'TransactionDate', 'TransactionTime', 'TransactionAmount', 'Age', 'TransactionYear',
    'TransactionMonth', 'TransactionDay', 'TransactionWeekday'
]

df = pd.read_csv(csv_file_path, header=None, names=column_names)

print("El archivo se leyo correctamente")

# ========================== PREPROCESAMIENTO ==========================
df_subset = df_subset.dropna()
scaler = MinMaxScaler()
X = scaler.fit_transform(df_subset)

# ========================== K-Means ==================================
def initialize_centroids(X, k):
    """Inicializa aleatoriamente k centroides"""
    return X[np.random.choice(X.shape[0], k, replace=False)]

def compute_distances(X, centroids):
    """Calcula la distancia euclidiana de cada punto a cada centroide"""
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances

def assign_clusters(distances):
    """Asigna cada punto al cluster más cercano"""
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """Recalcula los centroides como el promedio de los puntos asignados"""
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids

def k_means(X, k, max_iters=100, tol=1e-4):
    """Implementación del algoritmo K-Means desde cero"""
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(X, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids

# ========================== Método del Codo ==========================
def optimal_k(X, max_k=10):
    inertia = []
    n_samples = X.shape[0]

    if max_k > n_samples:
        print(f"Advertencia: el número máximo de clusters no puede ser mayor que el número de muestras ({n_samples}). Ajustando 'max_k' a {n_samples}.")
        max_k = n_samples

    for k in range(1, max_k + 1):
        labels, centroids = k_means(X, k)
        inertia.append(np.sum(np.min(compute_distances(X, centroids), axis=1)))

    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para Selección de k')
    plt.show()

optimal_k(X)

# ========================== PCA y Clustering ==========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

k = 5
labels, centroids = k_means(X, k)

# Visualizar los clusters obtenidos
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
plt.title(f'Clusters obtenidos con PCA y K-Means (k={k})')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()

