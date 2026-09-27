
# EDA - Variables catégorielles
# Dataset : hypertension_dataset.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import chi2_contingency
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hypertension_dataset.csv"

# 1) CHARGEMENT DU DATASET
df = pd.read_csv(DATA_PATH)
target = "Has_Hypertension"

# 2) DÉTECTION DES VARIABLES
explanatory_cols = [col for col in df.columns if col != target]

numeric_cols = df[explanatory_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in explanatory_cols if col not in numeric_cols]

# 3) NETTOYAGE
# Cible
df[target] = df[target].astype("string").fillna("Missing")

# Variables catégorielles
for col in categorical_cols:
    df[col] = df[col].astype("string").fillna("Missing")

# Variables numériques : forcer en numérique si jamais il y a des valeurs bizarres
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4) AFFICHAGE DES INFOS GÉNÉRALES
print("=" * 60)
print("INFO GÉNÉRALE DU DATASET")
print("=" * 60)

print("Shape :", df.shape)
print("\nVariables numériques :", numeric_cols)
print("Variables catégorielles :", categorical_cols)

print("\nValeurs manquantes :")
print(df.isna().sum())

print("\nRépartition de la cible :")
print(df[target].value_counts())

print("\nPourcentage de la cible :")
print((df[target].value_counts(normalize=True) * 100).round(2))

# ------------------------------------------------------------
# TABLEAU DE DESCRIPTION DE TOUTES LES VARIABLES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("DESCRIPTION COMPLÈTE DE TOUTES LES VARIABLES")
print("=" * 60)
print(df.describe(include="all").T)

# 5) STYLE GRAPHIQUE
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# 6) FONCTIONS UTILES

def cramers_v(x, y):
    """Mesure d'association entre deux variables catégorielles."""
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return np.nan
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    if n == 0:
        return np.nan
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1)))) if min(r - 1, k - 1) > 0 else 0


def eta_squared(cat, num):
    """Mesure d'association entre une variable catégorielle et une variable numérique."""
    df_tmp = pd.DataFrame({"cat": cat, "num": num}).dropna()
    if df_tmp.empty:
        return np.nan

    grand_mean = df_tmp["num"].mean()
    ss_total = ((df_tmp["num"] - grand_mean) ** 2).sum()

    ss_between = 0
    for level in df_tmp["cat"].unique():
        group = df_tmp[df_tmp["cat"] == level]["num"]
        ss_between += len(group) * (group.mean() - grand_mean) ** 2

    return ss_between / ss_total if ss_total != 0 else 0


def association_ratio(col1, col2, df_data, numeric_cols, categorical_cols):
    """Retourne une mesure d'association adaptée au type des variables."""
    if col1 in numeric_cols and col2 in numeric_cols:
        return df_data[col1].corr(df_data[col2])
    elif col1 in categorical_cols and col2 in categorical_cols:
        return cramers_v(df_data[col1], df_data[col2])
    else:
        if col1 in categorical_cols:
            return eta_squared(df_data[col1], df_data[col2])
        else:
            return eta_squared(df_data[col2], df_data[col1])

# 7) HISTOGRAMMES DES VARIABLES NUMÉRIQUES
if len(numeric_cols) > 0:
    n = len(numeric_cols)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=15, kde=True, ax=axes[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Fréquence")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# 8) HEATMAP DE CORRÉLATION ENTRE VARIABLES NUMÉRIQUES
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 7))
    corr_num = df[numeric_cols].corr()
    sns.heatmap(corr_num, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Corrélation entre variables numériques")
    plt.tight_layout()
    plt.show()

# 7bis) DISTRIBUTION DES VARIABLES CATÉGORIELLES (EN GRILLE)
if len(categorical_cols) > 0:
    n = len(categorical_cols)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(categorical_cols):

        # sécurisation (évite ton ancienne erreur)
        temp_col = df[col].astype(str).fillna("Missing")

        order = temp_col.value_counts().index

        sns.countplot(x=temp_col, order=order, ax=axes[i])
        axes[i].set_title(f"Distribution de {col}")
        axes[i].set_ylabel("Effectif")
        axes[i].tick_params(axis="x", rotation=45)

    # supprimer les graphes vides
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



# 12) MATRICE D'ASSOCIATION MIXTE
#     - Pearson : numérique vs numérique
#     - Cramér's V : catégorielle vs catégorielle
#     - Eta² : catégorielle vs numérique
#     - la variable cible est incluse

all_cols = numeric_cols + categorical_cols + [target]
mixed_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)

def mixed_association(col1, col2):
    """
    Mesure d'association adaptée selon le type des variables.
    La cible est traitée comme une variable catégorielle.
    """
    if col1 == col2:
        return 1.0

    # numérique vs numérique
    if col1 in numeric_cols and col2 in numeric_cols:
        return df[col1].corr(df[col2])

    # catégorielle vs catégorielle
    if (col1 in categorical_cols or col1 == target) and (col2 in categorical_cols or col2 == target):
        return cramers_v(df[col1], df[col2])

    # numérique vs catégorielle
    if col1 in numeric_cols and (col2 in categorical_cols or col2 == target):
        return eta_squared(df[col2], df[col1])

    if col2 in numeric_cols and (col1 in categorical_cols or col1 == target):
        return eta_squared(df[col1], df[col2])

    return np.nan

for col1 in all_cols:
    for col2 in all_cols:
        mixed_matrix.loc[col1, col2] = mixed_association(col1, col2)

plt.figure(figsize=(14, 11))
sns.heatmap(mixed_matrix.astype(float), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matrice d'association mixte avec la variable cible")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()