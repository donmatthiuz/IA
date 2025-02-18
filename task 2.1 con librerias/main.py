import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

csv_file_path = 'bank_transactions_cleaned.csv'

df_subset = df.drop(['TransactionID', 'CustomerID', 'CustGender', 'CustLocation', 'TransactionDate', 'TransactionTime'], axis=1)
df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
df_subset = df_subset.dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_subset)

# Split
X_temp, X_test = train_test_split(df_scaled, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.1, random_state=42)

# Determinamos el número de clusters usando el método del codo y el método de la silueta
wcss = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_val)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_val, kmeans.labels_))

plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.title('Método del codo')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(k_range, sil_scores, marker='s', linestyle='--', color='red')
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de la Silueta')
plt.title('Coeficiente de la Silueta para el número de clusters')
plt.show()

k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
kmeans.fit(X_train)
clusters = kmeans.predict(X_test)

final_silhouette = silhouette_score(X_test, clusters)
print(f"Coeficiente de la Silueta Final: {final_silhouette}")

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.7)
plt.title('Segmentación de Cliente usando K-Means and PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar(label='Cluster')
plt.show()
