import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
""" Pré-traitement des jeu de données"""
"""
red_data_path = "wine+quality/winequality-red.csv"
white_data_path = "wine+quality/winequality-white.csv"

data_red = pd.read_csv(red_data_path, sep=';')
data_white = pd.read_csv(white_data_path, sep=';')

print("Aperçu des données - Vin rouge :")
print(data_red.head())
print("\nAperçu des données - Vin blanc :")
print(data_white.head())

print("\nValeurs manquantes - Vin rouge :")
print(data_red.isnull().sum())
print("\nValeurs manquantes - Vin blanc :")
print(data_white.isnull().sum())

features_red = data_red.columns[:-1]  # Exclure la colonne 'quality'
features_white = data_white.columns[:-1]

data_red_features = data_red[features_red]
data_white_features = data_white[features_white]

scaler = StandardScaler()
data_red_normalized = scaler.fit_transform(data_red_features)
data_white_normalized = scaler.fit_transform(data_white_features)

normalized_df_red = pd.DataFrame(data_red_normalized, columns=features_red)
normalized_df_white = pd.DataFrame(
    data_white_normalized, columns=features_white)

print("\nDonnées normalisées - Vin rouge :")
print(normalized_df_red.head())
print("\nDonnées normalisées - Vin blanc :")
print(normalized_df_white.head())

normalized_df_red.to_csv("red_quality_normalized.csv", index=False)
normalized_df_white.to_csv("white_quality_normalized.csv", index=False)
"""
red_data_path = "red_quality_normalized.csv"
white_data_path = "white_quality_normalized.csv"

data_red = pd.read_csv(red_data_path, sep=';')
data_white = pd.read_csv(white_data_path, sep=';')


def kmeans_analysis(data, max_k=10):
    inertia = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        inertia.append(kmeans.inertia_)

        # Score de silhouette
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)

    return k_range, inertia, silhouette_scores


print("\nAnalyse des clusters pour le vin rouge :")
k_range_red, inertia_red, silhouette_red = kmeans_analysis(data_red)

print("\nAnalyse des clusters pour le vin blanc :")
k_range_white, inertia_white, silhouette_white = kmeans_analysis(
    data_white)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(k_range_red, inertia_red, marker='o', label='Vin rouge')
plt.plot(k_range_white, inertia_white, marker='o', label='Vin blanc')
plt.title('Méthode du coude')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.legend()

# Score de silhouette
plt.subplot(1, 2, 2)
plt.plot(k_range_red, silhouette_red, marker='o', label='Vin rouge')
plt.plot(k_range_white, silhouette_white, marker='o', label='Vin blanc')
plt.title('Score de silhouette')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Score de silhouette')
plt.legend()

plt.tight_layout()
plt.show()
