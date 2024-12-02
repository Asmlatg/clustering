import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

red_data_path = "red_quality_normalized.csv"
white_data_path = "white_quality_normalized.csv"

data_red = pd.read_csv(red_data_path, sep=',')
data_white = pd.read_csv(white_data_path, sep=',')

scaler = StandardScaler()
data_red_normalized = scaler.fit_transform(data_red)
data_white_normalized = scaler.fit_transform(data_white)


def kmeans_analysis(data, max_k=10):
    inertia = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        inertia.append(kmeans.inertia_)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)

    return k_range, inertia, silhouette_scores, cluster_labels


print("\nAnalyse des clusters pour le vin rouge :")
k_range_red, inertia_red, silhouette_red, red_clusters = kmeans_analysis(
    data_red_normalized)

print("\nAnalyse des clusters pour le vin blanc :")
k_range_white, inertia_white, silhouette_white, white_clusters = kmeans_analysis(
    data_white_normalized)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(k_range_red, inertia_red, marker='o', label='Vin rouge')
plt.plot(k_range_white, inertia_white, marker='o', label='Vin blanc')
plt.title('Méthode du coude')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_range_red, silhouette_red, marker='o', label='Vin rouge')
plt.plot(k_range_white, silhouette_white, marker='o', label='Vin blanc')
plt.title('Score de silhouette')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Score de silhouette')
plt.legend()

plt.tight_layout()
plt.show()

data_red['Cluster'] = red_clusters
data_white['Cluster'] = white_clusters

print("\nDescription des clusters pour le vin rouge :")
print(data_red.groupby('Cluster').mean())

print("\nDescription des clusters pour le vin blanc :")
print(data_white.groupby('Cluster').mean())
