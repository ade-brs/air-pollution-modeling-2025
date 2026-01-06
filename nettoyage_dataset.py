import pandas as pd

pd.options.display.max_rows = 9999

df = pd.read_csv('DATA\data.csv', sep=';')

#1ER nettoyage------------------

#drop colonnes sans infos
df.drop(columns=['Lien vers les mesures en direct', 'Mesures d’amélioration mises en place ou prévues','air','actions'],inplace=True)
df.drop_duplicates(inplace=True)
df.dropna(how='all',inplace=True)
print(df.info())


# Création d'une colonne unifiée par rapport au Niveau de pollution donné
df['pollution_unifiee'] = df['niveau_pollution'].fillna(df['niveau']) \
    .fillna(df['pollution_air']) \
    .fillna(df['Niveau de pollution aux particules']) \
    .fillna(df['Niveau de pollution'])

# Mise en minuscule + extraction du niveau clair (faible/moyenne/élevée)
df['pollution_unifiee'] = df['pollution_unifiee'].str.lower().str.extract(r'(pollution (faible|moyenne|élevée))', expand=False)[0]

# Supp des colonnes répétant mêmes info sur niveau pollution 
colonnes_pollution = [
    'niveau_pollution',
    'niveau',
    'pollution_air',
    'Niveau de pollution aux particules',
    'Niveau de pollution'
]
df.drop(columns=colonnes_pollution, inplace=True)

#Question 3 – Filtrage métro
Metro = df[df["Nom de la ligne"].str.contains("Métro", case=False, na=False)]

# Filtrage des lignes sans info pollution
Metro = Metro[~Metro['pollution_unifiee'].isin(['pas de données', 'station aérienne'])]

Train=Metro.head(int(0.7*len(Metro)+1)) #70 % de l'info de data.csv
Test=Metro.tail(int(0.3*len(Metro)))    #30%restant
print(Metro.to_string())
print(Train.info())
print(Test.info())

Metro.to_csv("stations_metro2.csv", index=False, encoding='utf-8-sig',sep=';')
Train.to_csv("train.csv", index=False, encoding='utf-8-sig',sep=';')
Test.to_csv("test.csv", index=False, encoding='utf-8-sig',sep=';')

#Question 2 Pour verif si ACP utile 
numeric_df = df.select_dtypes(include=['number'])
print(numeric_df.columns)
print(numeric_df.describe())

correlations = numeric_df.corr()
print(correlations)
# résultat Non  !!Ajoute justification Pk!!
