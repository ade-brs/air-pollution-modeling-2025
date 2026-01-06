## ANALYSE K-MEANS DES STATIONS DE MÉTRO (70% des stations) ##
# Seules variables numériques dispo du csv : Longitude et Latitude => Questionnement sur utilité de méthode des kmeans


# Chargement du fichier
df <- read.csv("train.csv", sep = ";", stringsAsFactors = FALSE)

# Sélection et centrage des variables numériqueshttp://127.0.0.1:27131/graphics/3d66b0b4-5653-4b7b-bb65-edd11a854411.png
X <- scale(df[, c("stop_lat", "stop_lon")])  # centrer-réduire

# ----- Règle du coude -----
R <- numeric()

for (k in 1:10) {
  kmeans_result <- kmeans(X, centers = k, nstart = 10)
  R[k] <- kmeans_result$tot.withinss   #stockage inertie inter-classe totale 
}

# Affichage du graphe du coude
plot(1:10, R, type = "b", pch = 19, frame = FALSE,
     xlab = "Nombre de clusters k", ylab = "Inertie intra-classe",
     main = "régle du coude pour déterminer k")
axis(1, at = 1:10)

# ----- CLUSTERING FINAL -----
# Interprétation graphique => grand saut d'information à k = 3
set.seed(123)
cl <- kmeans(X, centers = 3)

# Ajout des clusters au dataset
df$cluster <- as.factor(cl$cluster)

# ----- VISUALISATION  -----

# S'assurer que pollution_unifiee est bien facteur et ordonnée
df$pollution_unifiee <- factor(df$pollution_unifiee,
                               levels = c("pollution faible", "pollution moyenne", "pollution élevée"))

# Charger ggplot2
library(ggplot2)

# Tracer les clusters avec forme selon pollution
ggplot(df, aes(x = stop_lon, y = stop_lat, color = cluster, shape = pollution_unifiee)) +
  geom_point(size = 3) +
  labs(
    title = "Regroupement spatial des stations de métro avec différenciation par pollution",
    x = "Longitude", y = "Latitude",
    color = "Cluster", shape = "Niveau de pollution"
  ) +
  scale_color_manual(values = c("1" = "red", "2" = "green", "3" = "cyan")) +
  theme(legend.position = "right")



## ANALYS K PLUS PROCHE VOISIN ##




