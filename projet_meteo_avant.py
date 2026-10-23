"""
Première version du projet fil rouge (séances 2 et 3) : chargement, nettoyage et
exploration des données météo de Lille-Lesquin en 2023.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

# Fil rouge 2

df = pd.read_csv("Fil rouge donnees.csv.csv", sep=";", na_values=["mq"])
print(df.shape)

print("Lignes attendues : ", 365 * 8)
print(df.dtypes)

df = df[["date", "t", "pres", "u", "ff", "rr3"]].copy()
print(df.dtypes)

df["t_celsius"] = df["t"] - 273.15
print(df["t_celsius"].min(), df["t_celsius"].max())

print(df.isna().sum())

print(df["rr3"].isna().mean() * 100)
df = df.dropna(subset=["rr3"])
print(df.isna().sum().sum())

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S")
df["mois"] = df["date"].dt.month
print(df[["date", "mois"]].head())

mensuel = df.groupby("mois")["t_celsius"].agg(["mean", "min", "max"])
print(mensuel.round(1))

# Fil rouge 3

os.makedirs("figures", exist_ok=True)

ax = df.plot(x="date", y="t_celsius", figsize=(10, 4), legend=False)
ax.set_title("Température à Lille-Lesquin en 2023")
ax.set_ylabel("Température (°C)")
plt.savefig("figures/temperatures.png", dpi=150, bbox_inches="tight")
plt.show()

print(df["rr3"].describe())
print("Valeurs négatives :", (df["rr3"] < 0).sum())
print("Proportion de zéros :", (df["rr3"] == 0).mean())

df.loc[df["rr3"] < 0, "rr3"] = 0

pluie = df.loc[df["rr3"] > 0, "rr3"]
ax = pluie.plot(kind="hist", bins=20, figsize=(8, 4))
ax.set_title("Précipitations sur 3h (relevés avec pluie)")
ax.set_xlabel("Précipitations (en mm)")
ax.set_ylabel("Nombre de relevés")
plt.savefig("figures/precipitations_hist.png", dpi=150, bbox_inches="tight")
plt.show()

corr = df[["t_celsius", "pres", "u", "ff", "rr3"]].corr()
print(corr["t_celsius"].round(2))

ax = df.plot.scatter(x="u", y="t_celsius", figsize=(8, 4), alpha=0.3)
ax.set_title("Température selon l'humidité")
ax.set_xlabel("Humidité (%)")
ax.set_ylabel("Température (°C)")
plt.savefig("figures/temperatures_vs_humidite.png", dpi=150, bbox_inches="tight")
plt.show()