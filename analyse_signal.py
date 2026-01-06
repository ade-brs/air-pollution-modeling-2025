import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from scipy.linalg import eigh

# Chargement du dataset
df_pollution = pd.read_csv("C:/Users/ade/Desktop/Projet Ing1/stations_metro2.csv", sep=';')

# Nettoyage de la colonne 'niveau_pollution' : mise en minuscules + suppression des espaces
df_pollution['niveau_pollution'] = df_pollution['niveau_pollution'].str.lower().str.strip()

# Correspondance des niveaux de pollution en scores numériques
conversion = {
    'pollution faible': 1,
    'pollution moyenne': 2,
    'pollution élevée': 3  
}

# Appliquer la conversion
df_pollution['pollution_score'] = df_pollution['niveau_pollution'].map(conversion)

# Suppression des lignes sans données
df_valid = df_pollution.dropna(subset=['pollution_score'])

# Création du dictionnaire 
pollution_dict = dict(zip(df_valid['Nom de la Station'], df_valid['pollution_score']))


# Charger les connexions entre stations
graph_df = pd.read_csv("graph_stations.csv", sep=';')

# Extraire toutes les stations uniques
stations = pd.unique(graph_df[['Départ', 'Arrivée']].values.ravel())
station_index = {station: i for i, station in enumerate(stations)}
n = len(stations)

# Matrice d'adjacence A
A = np.zeros((n, n), dtype=int)
for _, row in graph_df[graph_df["Bool"] == True].iterrows():
    i = station_index[row["Départ"]]
    j = station_index[row["Arrivée"]]
    A[i, j] = 1
    A[j, i] = 1  # graphe non orienté

# Matrice de degré D
D = np.diag(np.sum(A, axis=1))

# Matrice laplacienne L
L = D - A

# Vecteur pollution x0 
x0 = np.zeros(n)
for station, idx in station_index.items():
    x0[idx] = pollution_dict.get(station, 0)  # 0 si la station n'a pas de donnée

import matplotlib.pyplot as plt

# Paramètres de simulation
tau = 0.0001  # pas de temps
iterations = 50  # nombre d'étapes
x_t = x0.copy()

# Sauvegarde de l'évolution
evolution = [x_t.copy()]

for t in range(iterations):
    x_t = x_t - tau * (L @ x_t)
    evolution.append(x_t.copy())

# Visualisation : évolution moyenne de la pollution
avg_pollution = [np.mean(x) for x in evolution]

plt.figure(figsize=(8, 4))
plt.plot(range(iterations + 1), avg_pollution, marker='o')
plt.title("Évolution de la pollution moyenne dans le réseau")
plt.xlabel("Nombre d'itérations")
plt.ylabel("Pollution moyenne")
plt.grid(True)
plt.show()

# Calcul de la variance de pollution à chaque itération
variances = [np.var(x) for x in evolution]

plt.figure(figsize=(6, 4))
plt.plot(range(len(variances)), variances, marker='o')
plt.title("Évolution de la variance du niveau de pollution")
plt.xlabel("Nombre d'itérations")
plt.ylabel("Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

# Tri décroissant des valeurs de pollution finale
x_final = evolution[-1]

top_indices = np.argsort(x_final)[-10:][::-1]

print("Top 10 des stations les plus polluées après diffusion :")
for idx in top_indices:
    station = list(station_index.keys())[list(station_index.values()).index(idx)]
    print(f"{station}: {x_final[idx]:.2f}")

# Graphe simulant les niveaux de pollution des stations après diffusion de la pollution

# Construction du graphe
G = nx.Graph()
for _, row in graph_df[graph_df["Bool"] == True].iterrows():
    G.add_edge(row["Départ"], row["Arrivée"])

# Pollution finale
x_final = evolution[-1]

for station, idx in station_index.items():
    G.nodes[station]['pollution'] = x_final[idx]

# Couleurs
node_colors = [G.nodes[station]['pollution'] for station in G.nodes()]
vmin = min(min(x0), min(x_final))
vmax = max(max(x0), max(x_final))

# Création du graphe
fig, ax = plt.subplots(figsize=(12, 10))
pos = nx.kamada_kawai_layout(G)
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,vmin=vmin, vmax=vmax, cmap=plt.cm.Reds, node_size=300, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.03, width=0.2, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

# Barre de couleur
cbar = plt.colorbar(nodes, ax=ax)
cbar.set_label("Niveau de pollution (après diffusion)")

plt.title("Propagation spatiale de la pollution dans le réseau")
plt.tight_layout()
plt.show()



# Graphe simulant les niveaux de pollution pour chaque station

pos = nx.kamada_kawai_layout(G)
# Couleurs selon x0
node_colors = [x0[station_index[station]] for station in G.nodes()]
vmin = min(min(x0), min(x_final))
vmax = max(max(x0), max(x_final))

# Création du graphe
fig, ax = plt.subplots(figsize=(12, 10))
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,vmin=vmin, vmax=vmax, cmap=plt.cm.Reds, node_size=300, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.3, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

# Barre de couleur
cbar = plt.colorbar(nodes, ax=ax)
cbar.set_label("Niveau de pollution (avant diffusion)")
plt.title("Pollution initiale dans le réseau")
plt.tight_layout()
plt.show()

print("Moyenne x0 :", np.mean(x0), " | écart-type :", np.std(x0))
print("Moyenne x_final :", np.mean(x_final), " | écart-type :", np.std(x_final))


# Evolution du niveau de pollution pour les 5 stations les plus polluées et les 5 moins polluées

# Sélection des indices
top5 = x0.argsort()[-5:][::-1]  # les plus polluées au départ
low5 = x0.argsort()[:5]         # les moins polluées au départ
mixed_indices = np.concatenate((top5, low5))
index_to_station = {v: k for k, v in station_index.items()}

# Création de la figure
plt.figure(figsize=(12, 6))
for idx in top5:
    station_name = index_to_station[idx]
    signal = [x[idx] for x in evolution]
    plt.plot(range(len(evolution)), signal, label=f"{station_name} (pollution forte)", color='red')

for idx in low5:
    station_name = index_to_station[idx]
    signal = [x[idx] for x in evolution]
    plt.plot(range(len(evolution)), signal, label=f"{station_name} (pollution faible)", linestyle='--', color='blue')

plt.title("Évolution de la pollution pour 5 stations très polluées et 5 peu polluées")
plt.xlabel("Nombre d'itérations")
plt.ylabel("Niveau de pollution")
plt.legend(title="Stations observées")
plt.grid(True)
plt.tight_layout()
plt.show()

# Graphe simulant les valeurs du vecteur de Fiedler

# Calcul des k plus petites valeurs propres 
eigvals, eigvecs = eigh(L)

# Affichage des premières valeurs propres
for i in range(10):
    print(f"Valeur propre {i}: {eigvals[i]:.6f}")

# Second vecteur propre 
fiedler_vector = eigvecs[:, 1]

# Création du graphe
pos = nx.kamada_kawai_layout(G)
fig, ax = plt.subplots(figsize=(12, 10))
nodes = nx.draw_networkx_nodes(G, pos, node_color=fiedler_vector, cmap=plt.cm.coolwarm, node_size=300, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.3, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

# Barre de couleur
cbar = plt.colorbar(nodes, ax=ax)
cbar.set_label("Valeurs du vecteur de Fiedler (λ₂)")

plt.title("Visualisation spectrale du réseau selon le vecteur de Fiedler")
plt.tight_layout()
plt.show()

# Graphe simulant les stations ayant un niveau de pollution dépassant le niveau seuil

seuil = 1.7
zones_critiques = [station for station, idx in station_index.items() if x_final[idx] >= seuil]
print(f"{len(zones_critiques)} stations dépassent le seuil de {seuil}")

G_critique = G.subgraph(zones_critiques)
clusters = list(nx.connected_components(G_critique))
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1} ({len(cluster)} stations) : {', '.join(cluster)}")

# stations critiques en rouge sinon en grise
color_map = ['red' if node in zones_critiques else 'lightgray' for node in G.nodes()]

# Création du graphe
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 10))
nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=300, edge_color='lightgray', alpha=0.3)
plt.title("Zones critiques : stations où la pollution finale dépasse le seuil de 1.7")

legend_elements = [
    mpatches.Patch(color='red', label='Stations critiques (pollution > 1.7)'),
    mpatches.Patch(color='lightgray', label='Autres stations')
]
plt.legend(handles=legend_elements, loc='lower left', title='Légende')
plt.tight_layout()
plt.show()


#Algorithme des K means pour 30 stations

def kmeans_1d_with_levels(data, k=3, max_iter=100):
    centroids = np.random.choice(data, k, replace=False)
    for _ in range(max_iter):
        distances = np.abs(data.reshape(-1, 1) - centroids.reshape(1, -1))
        clusters = np.argmin(distances, axis=1)
        new_centroids = np.array([
            data[clusters == i].mean() if len(data[clusters == i]) > 0 else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Ordre des centroïdes croissant  pollution faible à forte
    sorted_idx = np.argsort(new_centroids)
    cluster_to_level = {original: level for level, original in enumerate(sorted_idx, start=1)}
    levels = np.array([cluster_to_level[c] for c in clusters])
    return levels, new_centroids


levels, centroids = kmeans_1d_with_levels(x_final, k=3)


df_kmeans = pd.DataFrame({
    'station': list(station_index.keys()),
    'pollution': x_final,
    'pollution_level': levels
})
station_clusters = {row['station']: row['pollution_level'] for _, row in df_kmeans.iterrows()}

# Sélection des stations
sample_nodes = []
for level in [1, 2, 3]:
    cluster_nodes = [station for station, lvl in station_clusters.items() if lvl == level]
    sample_nodes += cluster_nodes[:10]

# Création du graphe
G_sample = G.subgraph(sample_nodes)
pos_sample = nx.kamada_kawai_layout(G_sample)
level_colors = {1: '#66c2a5', 2: '#fc8d62', 3: '#e78ac3'}
node_colors = [level_colors[station_clusters[station]] for station in G_sample.nodes()]

fig, ax = plt.subplots(figsize=(12, 8))
nx.draw_networkx_nodes(G_sample, pos_sample, node_color=node_colors, node_size=500, ax=ax)
nx.draw_networkx_edges(G_sample, pos_sample, alpha=0.3, width=1.0, edge_color='gray', ax=ax)
nx.draw_networkx_labels(G_sample, pos_sample, font_size=8, ax=ax)

legend_labels = [
    'Niveau 1 : Pollution faible',
    'Niveau 2 : Pollution moyenne',
    'Niveau 3 : Pollution forte'
]
legend_elements = [
    mpatches.Patch(color=level_colors[i + 1], label=legend_labels[i]) for i in range(3)
]
ax.legend(handles=legend_elements, loc='lower left', title="Clusters de pollution")

plt.title("Algorithme des KMeans sur 30 stations")
plt.tight_layout()
plt.show()

stations_ordonnées = sorted(station_clusters.items(), key=lambda x: x_final[station_index[x[0]]])

print("\nStations faiblement polluées :")
for s, lvl in stations_ordonnées[:5]:
    print(f"{s} - pollution {x_final[station_index[s]]:.2f} - niveau {lvl}")

print("\nStations fortement polluées :")
for s, lvl in stations_ordonnées[-5:]:
    print(f"{s} - pollution {x_final[station_index[s]]:.2f} - niveau {lvl}")

#Boxplot permettant de comparer les valeurs du vecteur de Fiedler au niveau de pollution

station_pollution_level = pollution_dict  

# Valeur du vecteur de Fiedler
index_to_station = {v: k for k, v in station_index.items()}
fiedler_values = {index_to_station[i]: fiedler_vector[i] for i in range(len(fiedler_vector))}

group_1, group_2, group_3 = [], [], []

for station, level in station_pollution_level.items():
    if station in fiedler_values:
        val = fiedler_values[station]
        if level == 1:
            group_1.append(val)
        elif level == 2:
            group_2.append(val)
        elif level == 3:
            group_3.append(val)

# Création de la figure
data = [group_1, group_2, group_3]
labels = ['Pollution faible', 'Pollution moyenne', 'Pollution élevée']

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels, patch_artist=True,
            boxprops=dict(facecolor='#a6bddb'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='gray'),
            flierprops=dict(markerfacecolor='red', marker='o', markersize=5, linestyle='none'))



plt.title("Distribution du vecteur de Fiedler selon le niveau de pollution initial")
plt.xlabel("Niveau de pollution (1 = faible, 2 = moyen, 3 = élevé)")
plt.ylabel("Valeur spectrale (vecteur de Fiedler)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
