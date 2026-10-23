"""
Projet fil rouge : analyse des données météo de Lille-Lesquin en 2023 et prédiction de la
température par régression linéaire.

Choix des variables : l'humidité u est la plus corrélée à la température (-0,61). La pression
pres et le vent ff le sont très peu (-0,06 et 0,02) et sont conservés pour que le modèle
confirme leur faible utilité. Le point de rosée td est écarté, car il se calcule à partir de
la température (fuite de données). Le mois est testé dans une variante pour donner la saison
au modèle.

Découpage : les relevés sont séparés au hasard en 80 % d'entraînement et 20 % de test, avec une
seed fixe pour que les résultats soient reproductibles.

Valeurs manquantes : les 5 lignes sans précipitations (0,17 %) sont supprimées, et la valeur
-0,1 (précipitations non mesurables) est remplacée par 0.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = "Fil rouge donnees.csv.csv"
FIGURES_DIR = "figures"
USEFUL_COLUMNS = ["date", "t", "pres", "u", "ff", "rr3"]
TEST_SIZE = 0.2
SEED = 42
MIN_ROWS = 100

FEATURES = ["pres", "u", "ff"]
FEATURES_WITH_MONTH = FEATURES + ["mois"]
TARGET = "t_celsius"


def load_data(path):
    """Charge le CSV Météo-France et renvoie un DataFrame.

    Lève une FileNotFoundError si le fichier est introuvable, une ValueError s'il est vide.
    """
    try:
        df = pd.read_csv(path, sep=";", na_values=["mq"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Vérifiez qu'il est dans le même dossier que le script."
        ) from None
    except pd.errors.EmptyDataError:
        raise ValueError(f"Le fichier {path} est vide.") from None
    return df


def check_columns(df, required):
    """Lève une ValueError si une colonne attendue est absente de df."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes du fichier : {missing}")


def clean_data(df):
    """Renvoie une copie nettoyée de df : colonnes utiles, température en °C, mois, heure.

    Supprime les lignes sans précipitations (rr3) et remplace -0,1 (précipitations non
    mesurables) par 0.
    """
    check_columns(df, USEFUL_COLUMNS)
    df = df[USEFUL_COLUMNS].copy()
    df["t_celsius"] = df["t"] - 273.15
    df = df.dropna(subset=["rr3"]).copy()
    df.loc[df["rr3"] < 0, "rr3"] = 0
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S")
    df["mois"] = df["date"].dt.month
    df["heure"] = df["date"].dt.hour
    return df


def monthly_summary(df):
    """Renvoie la température moyenne, minimale et maximale de chaque mois."""
    return df.groupby("mois")["t_celsius"].agg(["mean", "min", "max"])


def save_figure(name):
    """Enregistre la figure courante dans FIGURES_DIR."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, name), dpi=150, bbox_inches="tight")


def plot_exploration(df):
    """Trace et enregistre les figures d'exploration : températures, pluie, humidité.

    L'histogramme exclut les relevés sans pluie, trop nombreux, et la pluie mensuelle
    est un cumul.
    """
    ax = df.plot(x="date", y="t_celsius", figsize=(10, 4), legend=False)
    ax.set_title("Température à Lille-Lesquin en 2023")
    ax.set_ylabel("Température (°C)")
    save_figure("temperatures.png")
    plt.show()

    rain = df.loc[df["rr3"] > 0, "rr3"]
    ax = rain.plot(kind="hist", bins=20, figsize=(8, 4))
    ax.set_title("Précipitations sur 3 h (relevés avec pluie)")
    ax.set_xlabel("Précipitations (mm)")
    ax.set_ylabel("Nombre de relevés")
    save_figure("precipitations_hist.png")
    plt.show()

    ax = df.groupby("mois")["rr3"].sum().plot(kind="bar", figsize=(8, 4))
    ax.set_title("Précipitations totales par mois en 2023")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Précipitations (mm)")
    save_figure("precipitations_mensuelles.png")
    plt.show()

    print(df[["t_celsius", "pres", "u", "ff", "rr3"]].corr()["t_celsius"].round(2))
    ax = df.plot.scatter(x="u", y="t_celsius", figsize=(8, 4), alpha=0.3)
    ax.set_title("Température selon l'humidité")
    ax.set_xlabel("Humidité (%)")
    ax.set_ylabel("Température (°C)")
    save_figure("temperatures_vs_humidite.png")
    plt.show()


def prepare_xy(df, features, target):
    """Sépare df en X (variables explicatives) et y (température à prédire).

    Lève une ValueError si une colonne manque, s'il y a moins de MIN_ROWS lignes
    ou s'il reste des valeurs manquantes.
    """
    check_columns(df, features + [target])
    if len(df) < MIN_ROWS:
        raise ValueError(f"Pas assez de lignes : {len(df)} (minimum {MIN_ROWS}).")
    X, y = df[features], df[target]
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Valeurs manquantes dans X ou y : revenez au nettoyage.")
    return X, y


def split_data(X, y):
    """Découpe en train (80 %) et test (20 %), au hasard avec une seed fixe."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


def evaluate(y_true, y_pred):
    """Renvoie la MAE (°C), la RMSE (°C) et le R² des prédictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def plot_predictions(df, y_test, y_pred, name):
    """Trace les prédictions contre les valeurs réelles et l'erreur moyenne par mois.

    L'erreur est positive quand le modèle surestime la température.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_test, y_pred, alpha=0.3)
    low, high = y_test.min(), y_test.max()
    ax1.plot([low, high], [low, high], color="red")
    ax1.set_title("Prédictions contre valeurs réelles")
    ax1.set_xlabel("Température réelle (°C)")
    ax1.set_ylabel("Température prédite (°C)")

    errors = y_pred - y_test
    months = df.loc[y_test.index, "mois"]
    errors.groupby(months).mean().plot(kind="bar", ax=ax2)
    ax2.set_title("Erreur moyenne par mois")
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("Erreur moyenne (°C)")

    save_figure(name)
    plt.show()


def run_regression(df, features, figure_name):
    """Entraîne une régression linéaire, l'évalue et la compare à un modèle naïf.

    Le modèle naïf prédit toujours la température moyenne. Le point de rosée (td) n'est
    pas utilisé : il se calcule à partir de la température et fausserait la prédiction.
    """
    X, y = prepare_xy(df, features, TARGET)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train : {len(X_train)} relevés, test : {len(X_test)} relevés")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae, rmse, r2 = evaluate(y_test, y_pred)
    r2_train = r2_score(y_train, model.predict(X_train))

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    mae_naive = mean_absolute_error(y_test, baseline.predict(X_test))

    print(f"Variables : {features}")
    print(f"Coefficients : {model.coef_.round(3).tolist()} (constante : {model.intercept_:.1f})")
    print(f"Amplitude des températures : {y.max() - y.min():.1f} °C")
    print(f"MAE test : {mae:.2f} °C (modèle naïf : {mae_naive:.2f} °C)")
    print(f"RMSE test : {rmse:.2f} °C")
    print(f"R² test : {r2:.2f} | R² train : {r2_train:.2f}")
    plot_predictions(df, y_test, y_pred, figure_name)
    return model


def main():
    """Enchaîne les étapes du projet."""
    df = clean_data(load_data(DATA_FILE))
    print(monthly_summary(df).round(1))
    plot_exploration(df)
    print("\n--- Sans le mois ---")
    run_regression(df, FEATURES, "erreurs_sans_mois.png")
    print("\n--- Avec le mois ---")
    run_regression(df, FEATURES_WITH_MONTH, "erreurs_avec_mois.png")


if __name__ == "__main__":
    main()