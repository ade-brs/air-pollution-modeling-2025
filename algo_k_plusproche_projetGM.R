#-------chargement & préparation des données d'entraînement---------

#lecture du fichier des données d'entraînement "train.csv" en utilisant le point-virgule comme séparateur
train <- read.csv("train.csv", sep=";", stringsAsFactors = FALSE)

#suppression des lignes où la variable cible "pollution_unifiée" est vide
train <- train[!is.na(train$pollution_unifiee) & train$pollution_unifiee != "", ]

#affichage des modalités de la variable cible "pollution unifiée" pour vérifier la supression ci-dessus
unique(train$pollution_unifiee)

#affichage de la structure des données
str(train)

all_levels <- union(unique(train$Nom.de.la.ligne), unique(test$Nom.de.la.ligne))

#-------prétraitement de la variable cible et des variables explicatives-------

#conversion de la variable cible "pollution_unifiée" catégorielle en facteur (classification)
train$pollution_unifiee <- as.factor(train$pollution_unifiee)

#transformation de la variable explicative "nom de la ligne" en valeur numérique grâce à un endocage factoriel qui encode les différentes valeurs possibles de la variable "nom de la ligne"
train$Nom.de.la.ligne <- as.numeric(factor(train$Nom.de.la.ligne, levels=all_levels))

#encodage binaire de la variable explicative "Recommandation de surveillance" -> 1 si "oui" & 0 si "non"
train$Recommandation.de.surveillance <- ifelse(tolower(train$Recommandation.de.surveillance)== "oui", 1, 0)

#--------Choix et normalisation des variables explicatives--------

#contient les variables explicatives "latitude", "longitude", "nom de la ligne" & "recommandation de surveillance"
VE1_train <- train[, c("stop_lat", "stop_lon","Nom.de.la.ligne","Recommandation.de.surveillance")]

#contient la variable cible "pollution_unifiée"
VC_train <- train$pollution_unifiee

#normalisation (centrage & réduction) des variables explicatives -> utile pour les algos basés sur des distances comme le KNN
VE1_train <- scale(VE1_train)

#affichage des indicateurs concernant la distribution des variables explicatives afin de vérifier la normalisation de ces dernières
summary(VE1_train)

#affichage du nombre des observations de chaque modalité de la variable cible "pollution_unifiee" dans les données d'entraînement préparées
table(VC_train)


#-------chargement & préparation des données de test---------

#lecture du fichier des données de test "test.csv" en utilisant le point-virgule comme séparateur
test <- read.csv("test.csv", sep=";", stringsAsFactors = FALSE)

#suppression des lignes où la variable cible "pollution_unifiée" est vide
test <- test[!is.na(test$pollution_unifiee) & test$pollution_unifiee != "", ]

#affichage des modalités de la variable cible "pollution unifiée" pour vérifier la supression ci-dessus
unique(test$pollution_unifiee)

#affichage de la structure des données
str(test)


#-------prétraitement de la variable cible et des variables explicatives-------

#conversion de la variable cible "pollution_unifiée" catégorielle en facteur (classification)
test$pollution_unifiee <- as.factor(test$pollution_unifiee)

#transformation de la variable explicative "nom de la ligne" en valeur numérique grâce à un endocage factoriel qui encode les différentes valeurs possibles de la variable "nom de la ligne"
test$Nom.de.la.ligne <- as.numeric(factor(test$Nom.de.la.ligne, levels=all_levels))

#encodage binaire de la variable explicative "Recommandation de surveillance" -> 1 si "oui" & 0 si "non"
test$Recommandation.de.surveillance <- ifelse(tolower(test$Recommandation.de.surveillance)== "oui", 1, 0)

#--------Choix et normalisation des variables explicatives--------

#contient les variables explicatives "latitude", "longitude", "nom de la ligne" & "recommandation de surveillance"
VE1_test=test[, c("stop_lat", "stop_lon","Nom.de.la.ligne","Recommandation.de.surveillance")]

#contient la variable cible "pollution_unifiée"
VC_test=test$pollution_unifiee

#normalisation du "test" avec les mêmes paramètres que pour le "train" -> important pour éviter les biais de centrage
VE1_test <- scale(VE1_test,
                  center = attr(VE1_train, "scaled:center"),
                  scale = attr(VE1_train, "scaled:scale"))

#affichage des indicateurs concernant la distribution des variables explicatives afin de vérifier la normalisation de ces dernières
summary(VE1_test)

#affichage du nombre des observations de chaque modalité de la variable cible "pollution_unifiee" dans les données de test préparées
table(VC_test)

#--------classification avec l'algorithme k-NN------

#chargement de la librairie "class" pour appliquer l'algorithme k-NN
library(class)

#--------recherche du meilleur k pour la obtenir la meilleure précision----------

k_test<-1:20

#initialisation du vecteur vide qui va stocker toutes les précisions des applications suivantes de l'algo du k-NN
precisions <- numeric(length(k_test))

#application de l'algorithme k-NN avec les valeurs de k comprises entre 1 et 20 et calcul des précisions en % et arrondie à 2 décimales associées aux valeurs testées de k 
for (i in k_test) {
  VC_pred <- knn(train = VE1_train, test = VE1_test, cl = VC_train, k = i)
  precisions[i] <- mean(VC_pred == VC_test)
}
#création et affichage d'un tableau 2D afin d'afficher la précision en % et arrondie à 2 décimales, associée à chaque valeur testée de k
print(data.frame(k = k_test, precision = round(precisions * 100, 2)))

#sélection du meilleur k selon la comparaison des précisions associées
k_optimal <- k_test[which.max(precisions)]

#affichage du meilleur k et sa précision associée
cat("meilleur k :",k_optimal, "avec une précision de",round(max(precisions) * 100, 2),"%\n")

#visualisation graphique de la performance de l'algorithme du k-NN selon les valeurs testées de k
plot(k_test, precisions * 100, type = "b", col = "blue", pch = 19,
     xlab = "k", ylab = "Précision (%)", main = "Choix du meilleur k")
abline(v = k_optimal, col = "red", lty = 2)

#--------Matrice de confusion-----------

#chargement des librairies "carret" et "ggplot2" afin de créer la visualisation graphique de la amtrice de confusion
library(caret)
library(ggplot2)

#application de l'algorithme k-NN avec le meilleur k afin de prédire les données de test
VC_prediction <- knn(train = VE1_train, test = VE1_test, cl=VC_train, k=k_optimal)

#calcul de la matrice de confusion
matrice_confusion <- confusionMatrix(VC_prediction, VC_test)

#visualisation graphique finale de la matrice de confusion
df_conf <- as.data.frame(matrice_confusion$table)
ggplot(df_conf, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "                Matrice de confusion") +
  theme_minimal()







