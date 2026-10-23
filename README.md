# Projet fil rouge : température à Lille-Lesquin en 2023

## Objectif

Ce projet analyse les données météo de la station de Lille-Lesquin (station 07015, année 2023, une mesure toutes les 3 heures). Il prédit ensuite la température avec une régression linéaire de scikit-learn.

## Données

Les données proviennent de Météo-France (SYNOP) et sont dans le fichier `Fil rouge donnees.csv.csv`. Le séparateur est `;` et les valeurs manquantes sont notées `mq`. Le fichier compte 2864 lignes et 60 colonnes. Une année complète à raison de 8 mesures par jour en demanderait 2920 : 56 relevés sont absents, répartis en 8 interruptions, ce qui peut sous-estimer légèrement les cumuls de pluie mensuels.

Le script utilise six colonnes :

- `date` : la date et l'heure du relevé.
- `t` : la température, en kelvins.
- `pres` : la pression, en Pa.
- `u` : l'humidité, en %.
- `ff` : la vitesse du vent, en m/s.
- `rr3` : les précipitations sur 3 heures, en mm.

## Lancer le projet

1. Installez les dépendances avec `pip install pandas matplotlib scikit-learn`.
2. Placez le fichier de données dans le même dossier que `projet_meteo.py`.
3. Lancez `python projet_meteo.py`, ou exécutez le fichier dans Spyder.

Le fichier `projet_meteo_avant.py` est ma première version du script (fils rouges 2 et 3, sans fonctions). Je l'ai conservé pour montrer l'évolution du projet. La version finale est `projet_meteo.py`.

Le script affiche le résumé mensuel, les corrélations et les résultats de la régression. Il enregistre aussi 6 graphiques dans le dossier `figures/` :

- Quatre graphiques d'exploration : les températures, l'histogramme des pluies, les pluies mensuelles et la température selon l'humidité.
- Deux graphiques d'évaluation : les prédictions et les erreurs par mois, sans puis avec la variable `mois`.

## Choix de nettoyage

Le script applique les opérations suivantes :

- Il convertit la température de kelvins en degrés Celsius (`t_celsius`).
- Il supprime les 5 lignes sans valeur de `rr3` (0,17 % des 2864 relevés), car ce nombre est trop faible pour changer les résultats.
- Il remplace la valeur -0,1 de `rr3` par 0, car elle signifie « précipitations non mesurables » (moins de 0,1 mm) et fausserait les cumuls.
- Il extrait les colonnes `mois` et `heure` de la date.

## Exploration

La température suit la saison : sa moyenne est de 5,7 °C en janvier et de 20,4 °C en juin, et elle varie de -5,0 °C à 33,9 °C sur l'année, soit une amplitude d'environ 39 °C. Environ 72 % des relevés de pluie valent 0.

Les corrélations des variables avec la température sont les suivantes :

- l'humidité `u` : -0,61.
- la pression `pres` : -0,06.
- le vent `ff` : 0,02.
- la pluie `rr3` : -0,03.

## Modèle

Le projet utilise une régression linéaire de scikit-learn.

### Découpage des données

Le découpage est aléatoire avec une seed fixe (42), donc le résultat est identique à chaque exécution. 80 % des relevés servent à l'entraînement (2287) et 20 % servent au test (572).

### Choix des variables

- `u` est conservée, car c'est la variable la plus corrélée à la température.
- `pres` et `ff` sont conservées, car le modèle permet de confirmer leur faible utilité.
- `td` (le point de rosée) est écartée, car elle est calculée à partir de la température elle-même et donnerait la réponse au modèle (fuite de données).
- `mois` est testée dans une variante.
- `heure` est testée mais non retenue, car son coefficient est quasi nul (-0,016) et la MAE reste à 4,15 °C avec ou sans elle dans le modèle avec le mois.

### Résultats

Le tableau suivant compare les trois modèles sur le jeu de test.

| Modèle | MAE test | RMSE test | R² test | R² train |
|---|---|---|---|---|
| Naïf (température moyenne) | 5,81 °C | - | - | - |
| `pres`, `u`, `ff` | 4,36 °C | 5,40 °C | 0,42 | 0,41 |
| `pres`, `u`, `ff`, `mois` | 4,15 °C | 5,10 °C | 0,48 | 0,47 |

### Coefficients

Le modèle avec le mois donne les coefficients suivants :

- `pres` : -0,001 par pascal.
- `u` : -0,27 par point d'humidité.
- `ff` : -0,34 par m/s.
- `mois` : +0,49 par mois.

Chaque coefficient est l'effet d'une unité de sa variable. Comme les unités sont différentes (pascals, %, m/s, numéro de mois), les coefficients ne sont pas comparables entre eux. Le signe négatif de `u` est cohérent avec le nuage de points : plus l'air est humide, plus il fait froid. Le coefficient de `ff` est négatif alors que sa corrélation avec la température est presque nulle (0,02) : une fois l'humidité prise en compte, son effet est faible et il ne faut pas l'interpréter trop vite. Enfin, le coefficient positif de `mois` traduit la hausse des températures de janvier à l'été, mais il continue d'augmenter en fin d'année, alors que la température redescend.

## Conclusion et limites

Le modèle est utile mais limité. Sa MAE est de 4,4 °C, contre 5,8 °C pour un modèle naïf qui prédirait toujours la température moyenne de l'année : il fait donc mieux que cette référence. Cette erreur représente environ un neuvième de l'amplitude annuelle (39 °C), mais elle reste trop grande pour une prédiction précise, puisque le modèle n'explique que 42 % des variations de température (R² = 0,42). Les R² sont presque identiques à l'entraînement (0,41) et au test (0,42) : le modèle n'a donc pas appris les données par cœur, il est simplement trop simple avec ces trois variables.

Les erreurs sont systématiques et non aléatoires. Le modèle surestime la température en hiver (environ +4 °C de janvier à avril) et la sous-estime en été (environ -4 à -5 °C de juin à septembre), car il prédit des valeurs trop proches de la moyenne de l'année.

Ajouter le mois réduit l'erreur de janvier à avril (environ +2 à +3 °C) et ne change presque rien en été (toujours environ -4 à -5 °C). Il aggrave même l'erreur en novembre et décembre (environ +4 °C), car un coefficient unique sur le numéro du mois ne peut pas représenter une saison qui monte puis redescend.