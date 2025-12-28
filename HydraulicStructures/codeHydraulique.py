# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 22:22:55 2025

@author: achra
"""

# -*- coding: utf-8 -*-
"""
AFC sur le tableau 8 canaux × 8 critères hydrauliques
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Chargement du fichier avec header à la ligne 4 ---
file_path = r'C:\Users\achra\Desktop\OuvragesHydrauliques\Ouvrages_hydrauliques.xlsx'
data_hydraulique = pd.read_excel(file_path, header=0)

# --- Suppression colonne inutile si présente ---
"""if 'Unnamed: 0' in data_bim.columns:
    data_bim = data_bim.drop(columns=['Unnamed: 0'])"""

# --- Renommage des colonnes pour cohérence ---
data_hydraulique.columns = ['Channel', 'Flow rate (m³/s)', 'Water velocity (m/s)', 'Width (m)', 'Depth (m)',
                    'Roughness', 'Slope (%)', 'Water temperature (°C)', 'Siltation (%)']
							
										
# Nettoyage des colonnes numériques
numeric_cols = ['Flow rate (m³/s)', 'Water velocity (m/s)', 'Width (m)', 'Depth (m)',
                    'Roughness', 'Slope (%)', 'Water temperature (°C)', 'Siltation (%)']

for col in numeric_cols:
    data_hydraulique[col] = data_hydraulique[col].astype(str).str.strip().str.replace(' ', '').replace('', np.nan)
    data_hydraulique[col] = pd.to_numeric(data_hydraulique[col], errors='coerce')

data_hydraulique_clean = data_hydraulique.dropna()

X = data_hydraulique_clean[numeric_cols]

# ----- Partie 1 -----
print("=== Partie 1 : Statistiques descriptives et matrices ===\n")

# 1 - Moyenne, écart-type et interprétation
moyennes = X.mean()
ecarts_type = X.std()
print("1 - Moyenne et Écart-Type par critère :")
for c in numeric_cols:
    print(f"{c} : Moyenne = {moyennes[c]:.2f}, Écart-type = {ecarts_type[c]:.2f} | "
          f"Interprétation : La moyenne exprime la tendance centrale, l'écart-type la dispersion.")

# 2 - Matrice centrée réduite
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
print("\n2 - Matrice centrée réduite (extrait) :")
print(X_scaled_df.head())
X_scaled_df.to_csv(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\matrice_centree_reducee.csv", index=False)
print("-> Matrice centrée réduite exportée dans matrice_centree_reducee.csv")

plt.figure(figsize=(14, 20))
sns.heatmap(
    X_scaled_df,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.8,
    linecolor='gray',
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 6}
)
plt.title("Standardized Data Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Division en 4 parties de 25 lignes (canaux)
n_canaux_par_figure = 25
parts = [
    X_scaled_df.iloc[0:25, :],      # Canaux 1-25
    X_scaled_df.iloc[25:50, :],     # Canaux 26-50
    X_scaled_df.iloc[50:75, :],     # Canaux 51-75
    X_scaled_df.iloc[75:100, :]     # Canaux 76-100
]

# Paramètres communs
kwargs_heatmap = {
    'annot': True,
    'fmt': '.2f',
    'cmap': 'RdBu_r',
    'center': 0,
    'linewidths': 0.8,
    'linecolor': 'gray',
    'cbar_kws': {'shrink': 0.8},
    'annot_kws': {'size': 6}
}

# Création de 4 figures séparées
for i, part in enumerate(parts):
    # Nouvelle figure pour chaque partie
    plt.figure(figsize=(14, 7))  # Taille optimisée pour 25 lignes
    
    sns.heatmap(part, **kwargs_heatmap)
    
    # Titre spécifique
    debut_canal = i * 25 + 1
    fin_canal = (i + 1) * 25
    plt.title(f"Standardized Data Matrix - Channels {debut_canal}-{fin_canal}", 
              fontsize=14, pad=20)
    
    # Rotation des labels
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()  # Affiche chaque figure séparément

# 3 - Matrice de corrélation et interprétation
corr_df = pd.DataFrame(np.corrcoef(X_scaled.T), index=numeric_cols, columns=numeric_cols)
print("\n3 - Matrice de corrélation :")
print(corr_df.round(2))
plt.figure(figsize=(10, 7))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
print("\nInterprétation complète des corrélations :")
for i, ci in enumerate(numeric_cols):
    for j, cj in enumerate(numeric_cols):
        if i < j:
            val = corr_df.iloc[i, j]
            desc = "aucune corrélation claire"
            if val > 0.7: desc = "corrélation positive forte"
            elif 0.3 < val <= 0.7: desc = "corrélation positive modérée"
            elif 0 < val <= 0.3: desc = "corrélation positive faible"
            elif -0.3 < val < 0: desc = "corrélation négative faible"
            elif -0.7 <= val <= -0.3: desc = "corrélation négative modérée"
            elif val < -0.7: desc = "corrélation négative forte"
            print(f"- Entre {ci} et {cj} : {desc} (r = {val:.2f})")
corr_df.to_csv(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\matrice_correlation.csv")
print("-> Matrice de corrélation exportée dans matrice_correlation.csv")

# 4 - Valeurs propres et inertie
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

valeurs_propres = pca.explained_variance_
inertie_expliquee = pca.explained_variance_ratio_
inertie_cumulee = np.cumsum(inertie_expliquee)

print("\n4 - Valeurs propres et inertie expliquée :")
for i, val in enumerate(valeurs_propres, 1):
    print(f"Component {i} : {val:.4f} | Inertie expliquée : {inertie_expliquee[i-1]:.4f} ({inertie_expliquee[i-1]*100:.2f}%) - Cumulée : {inertie_cumulee[i-1]*100:.2f}%")

plt.figure(figsize=(8,5))
plt.plot(range(1, len(inertie_expliquee)+1), inertie_expliquee*100, marker='o', label='Inertia % per component')
plt.plot(range(1, len(inertie_cumulee)+1), inertie_cumulee*100, marker='s', label='Cumulative inertia %')
plt.title("Inertia explained by principal components")
plt.xlabel("Main components")
plt.ylabel("Percentage of inertia")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----- Partie 2 -----
print("\n=== Partie 2 : Analyse en Composantes Principales ===\n")

# 1 - Composantes principales individus et variables
individus_pca = pd.DataFrame(X_pca, columns=[f"Component_{i+1}" for i in range(X_pca.shape[1])], index=data_hydraulique_clean['Channel'])
variables_pca = pd.DataFrame(pca.components_, columns=numeric_cols, index=[f"Component_{i+1}" for i in range(len(numeric_cols))])

plt.figure(figsize=(14, 20))
sns.heatmap(
    individus_pca,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.8,
    linecolor='gray',
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 6}
)
plt.title("Principal components of the individuals")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
    

# Division en 4 parties de 25 lignes (individus)
n_individus_par_figure = 25
parts = [
    individus_pca.iloc[0:25, :],      # Individus 1-25
    individus_pca.iloc[25:50, :],     # Individus 26-50
    individus_pca.iloc[50:75, :],     # Individus 51-75
    individus_pca.iloc[75:100, :]     # Individus 76-100
]

# Paramètres communs (adaptés à ton code)
kwargs_heatmap = {
    'annot': True,
    'fmt': '.2f',
    'cmap': 'coolwarm',
    'linewidths': 0.8,
    'linecolor': 'gray',
    'cbar_kws': {'shrink': 0.8},
    'annot_kws': {'size': 6}
}

# Création de 4 figures séparées
for i, part in enumerate(parts):
    # Nouvelle figure pour chaque partie
    plt.figure(figsize=(14, 7))  # Taille optimisée pour 25 lignes
    
    sns.heatmap(part, **kwargs_heatmap)
    
    # Titre spécifique
    debut_individu = i * 25 + 1
    fin_individu = (i + 1) * 25
    plt.title(f"Principal components - Individuals {debut_individu}-{fin_individu}", 
              fontsize=14, pad=20)
    
    # Rotation des labels (comme ton code original)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()  # Affiche chaque figure séparément



plt.figure(figsize=(10, 7))
sns.heatmap(
    variables_pca,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",   # Palette avec divergence (positif/négatif)
    center=0,
    linewidths=0.8,
    linecolor='gray',
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8}
)
plt.title("Principal components of the variables")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("1 - Composantes principales des individus (extrait) :")
print(individus_pca.head())

print("\n1 - Composantes principales des variables :")
print(variables_pca)

individus_pca.to_csv(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\composantes_individus.csv")
variables_pca.to_csv(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\composantes_variables.csv")
print("\n-> Tables des composantes principales exportées (individus et variables)")

# 2 - Plan factoriel des individus
plt.figure(figsize=(8,6))
plt.scatter(individus_pca['Component_1'], individus_pca['Component_2'])
for i, txt in enumerate(individus_pca.index):
    plt.annotate(txt, (individus_pca['Component_1'][i], individus_pca['Component_2'][i]), fontsize=8)
plt.title("Factorial plane of individuals (Components 1 et 2)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nInterprétation : proximities entre individus indiquent la formation éventuelle de clusters / groupes.")

# 3 - Plan factoriel des variables
plt.figure(figsize=(8,6))
for i in range(len(numeric_cols)):
    plt.arrow(0, 0, variables_pca.loc["Component_1", numeric_cols[i]], variables_pca.loc["Component_2", numeric_cols[i]], 
              head_width=0.03, head_length=0.03, color='red')
    plt.text(variables_pca.loc["Component_1", numeric_cols[i]]*1.15, variables_pca.loc["Component_2", numeric_cols[i]]*1.15, numeric_cols[i], 
             color='blue', fontsize=9)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axhline(0, color='grey', linewidth=1)
plt.axvline(0, color='grey', linewidth=1)
plt.title("Factor plane of variables (Components 1 et 2)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nInterprétation : proximité des variables indique corrélation positive, opposition corrélation négative.")

# 4 - Qualité de représentation des individus
coord_carres = individus_pca**2
qualite_rep = pd.DataFrame()
qualite_rep['Quality Axis 1'] = coord_carres['Component_1'] / np.sum(coord_carres['Component_1'])
qualite_rep['Quality Axis 2'] = coord_carres['Component_2'] / np.sum(coord_carres['Component_2'])
qualite_rep['Factor Plane Quality'] = qualite_rep['Quality Axis 1'] + qualite_rep['Quality Axis 2']

print("\n4 - Qualité de représentation des individus (extrait) :")
print(qualite_rep.head().round(3))
qualite_rep.to_csv(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\qualite_representation.csv")
print("\n-> Qualité de représentation exportée dans qualite_representation.csv")

plt.figure(figsize=(15, 17))
sns.heatmap(
    qualite_rep,
    annot=True,
    fmt=".3f",
    cmap="crest",
    linewidths=0.8,
    linecolor='gray',
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 6}
)
plt.title("Quality of representation for individuals")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Division en 4 parties de 25 lignes (individus)
n_individus_par_figure = 25
parts = [
    qualite_rep.iloc[0:25, :],      # Individus 1-25
    qualite_rep.iloc[25:50, :],     # Individus 26-50
    qualite_rep.iloc[50:75, :],     # Individus 51-75
    qualite_rep.iloc[75:100, :]     # Individus 76-100
]

# Paramètres communs (adaptés à ton code)
kwargs_heatmap = {
    'annot': True,
    'fmt': '.3f',  # 3 décimales comme ton original
    'cmap': 'crest',
    'linewidths': 0.8,
    'linecolor': 'gray',
    'cbar_kws': {'shrink': 0.8},
    'annot_kws': {'size': 6}
}

# Création de 4 figures séparées
for i, part in enumerate(parts):
    # Nouvelle figure pour chaque partie
    plt.figure(figsize=(14, 7))  # Taille optimisée pour 25 lignes
    
    sns.heatmap(part, **kwargs_heatmap)
    
    # Titre spécifique
    debut_individu = i * 25 + 1
    fin_individu = (i + 1) * 25
    plt.title(f"Quality of representation - Individuals {debut_individu}-{fin_individu}", 
              fontsize=14, pad=20)
    
    # Rotation des labels (comme ton code original)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()  # Affiche chaque figure séparément


# 5 - Contribution des individus et des variables sur les axes 1 et 2

# Contribution des individus (lignes) sur les deux premiers axes
coord_carres_12 = individus_pca[['Component_1', 'Component_2']]**2
somme_coord = coord_carres_12.sum(axis=0)           # somme par axe
contrib_ind = coord_carres_12.divide(somme_coord, axis=1) * 100

print("\n5 - Contribution des individus (axes 1 et 2) – en % (extrait) :")
print(contrib_ind.head().round(2))

contrib_ind.to_csv(
    r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\contribution_individus.csv"
)

plt.figure(figsize=(15, 17))
sns.heatmap(
    contrib_ind,
    annot=True,
    fmt=".1f",
    cmap="OrRd",
    linewidths=0.8,
    linecolor="gray",
    cbar_kws={"shrink": 0.8, "label": "Contribution (%)"},
    annot_kws={"size": 6}
)
plt.title("Contribution of individuals (%)")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Division en 4 parties de 25 lignes (individus)
n_individus_par_figure = 25
parts = [
    contrib_ind.iloc[0:25, :],      # Individus 1-25
    contrib_ind.iloc[25:50, :],     # Individus 26-50
    contrib_ind.iloc[50:75, :],     # Individus 51-75
    contrib_ind.iloc[75:100, :]     # Individus 76-100
]

# Paramètres communs (adaptés à ton code)
kwargs_heatmap = {
    'annot': True,
    'fmt': '.1f',  # 1 décimale comme ton original
    'cmap': 'OrRd',
    'linewidths': 0.8,
    'linecolor': 'gray',
    'cbar_kws': {'shrink': 0.8, 'label': 'Contribution (%)'},
    'annot_kws': {'size': 6}
}

# Création de 4 figures séparées
for i, part in enumerate(parts):
    # Nouvelle figure pour chaque partie
    plt.figure(figsize=(8, 7))  # Taille adaptée (souvent 2 colonnes pour contrib)
    
    sns.heatmap(part, **kwargs_heatmap)
    
    # Titre spécifique
    debut_individu = i * 25 + 1
    fin_individu = (i + 1) * 25
    plt.title(f"Contribution of individuals - {debut_individu}-{fin_individu} (%)", 
              fontsize=14, pad=20)
    
    # Rotation des labels (comme ton code original - les deux axes à 0°)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()  # Affiche chaque figure séparément

# Contribution des variables (colonnes) sur les axes 1 et 2
# On part des coordonnées des variables dans variables_pca (lignes = composantes, colonnes = variables)
comp_carres_12 = variables_pca.loc[["Component_1", "Component_2"], numeric_cols]**2
somme_comp = comp_carres_12.sum(axis=1)             # somme par axe
contrib_var = comp_carres_12.divide(somme_comp, axis=0) * 100

print("\n5 - Contribution des variables sur les axes 1 et 2 – en % :")
print(contrib_var.round(2))

contrib_var.to_csv(
    r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\contribution_variables.csv"
)

plt.figure(figsize=(8, 4))
sns.heatmap(
    contrib_var,
    annot=True,
    fmt=".1f",
    cmap="OrRd",
    linewidths=0.8,
    linecolor="gray",
    cbar_kws={"shrink": 0.8, "label": "Contribution (%)"},
    annot_kws={"size": 8}
)
plt.title("Contribution of variables  (%)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans

# Utilisation des deux premières composantes principales
X_pca_plot = individus_pca.iloc[:, :2] # colonnes 'Composante_1' et 'Composante_2'

# K-Means avec K=3
kmeans_pca3 = KMeans(n_clusters=3, random_state=42)
individus_pca['Cluster_K3'] = kmeans_pca3.fit_predict(X_pca_plot)

print("\n== K-Means sur ACP (K=3) ==")
print(individus_pca[['Component_1', 'Component_2', 'Cluster_K3']].head())

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=individus_pca['Component_1'],
    y=individus_pca['Component_2'],
    hue=individus_pca['Cluster_K3'],
    palette="tab10"
)
for i, txt in enumerate(individus_pca.index):
    plt.annotate(txt, (individus_pca['Component_1'][i], individus_pca['Component_2'][i]), fontsize=7)
plt.title("Clusters K-Means (ACP, K=3)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# K-Means avec K=4
kmeans_pca4 = KMeans(n_clusters=4, random_state=42)
individus_pca['Cluster_K4'] = kmeans_pca4.fit_predict(X_pca_plot)

print("\n== K-Means sur ACP (K=4) ==")
print(individus_pca[['Component_1', 'Component_2', 'Cluster_K4']].head())

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=individus_pca['Component_1'],
    y=individus_pca['Component_2'],
    hue=individus_pca['Cluster_K4'],
    palette="Set2"
)
for i, txt in enumerate(individus_pca.index):
    plt.annotate(txt, (individus_pca['Component_1'][i], individus_pca['Component_2'][i]), fontsize=7)
plt.title("Clusters K-Means (ACP, K=4)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()


print("\nRépartition des effectifs (K=3) :")
print(individus_pca['Cluster_K3'].value_counts().sort_index())

print("\nRépartition des effectifs (K=4) :")
print(individus_pca['Cluster_K4'].value_counts().sort_index())

import seaborn as sns
import matplotlib.pyplot as plt

# Remettre à zéro l'index pour caler les deux DataFrames
X_reset = X.reset_index(drop=True)
individus_pca_reset = individus_pca.reset_index(drop=True)

# Ajout des colonnes cluster à X avec index cohérent
X_with_clusters_K3 = X_reset.copy()
X_with_clusters_K3['Cluster_K3'] = individus_pca_reset['Cluster_K3']

cluster_profiles_K3 = X_with_clusters_K3.groupby('Cluster_K3').mean()

print(cluster_profiles_K3)  # Vérifie le contenu et la forme

# Affichage Heatmap pour K=3
if cluster_profiles_K3.size > 0:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    sns.heatmap(
        cluster_profiles_K3,
        annot=True,
        fmt=".2f",
        cmap="Reds",          # palette plus cohérente avec des profils moyens
        linewidths=0.5,
        linecolor="white",
        cbar_kws={'shrink': 0.7, 'label': 'Average value'},
        annot_kws={"size": 9, "color": "black"},
        square=True,
        ax=ax
    )

    ax.set_title(
        "Clusters Interpretation (K=3) – average profiles",
        fontsize=14,
        pad=12
    )
    ax.set_xlabel("Hydraulic variables", fontsize=11)
    ax.set_ylabel("Clusters", fontsize=11)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    fig.tight_layout()
    plt.show()
else:
    print("Aucune donnée à afficher pour K=3.")


# Pour K=4
X_with_clusters_K4 = X_reset.copy()
X_with_clusters_K4['Cluster_K4'] = individus_pca_reset['Cluster_K4']

cluster_profiles_K4 = X_with_clusters_K4.groupby('Cluster_K4').mean()
print(cluster_profiles_K4)  # Vérifie le contenu et la forme

if cluster_profiles_K4.size > 0:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    sns.heatmap(
        cluster_profiles_K4,
        annot=True,
        fmt=".2f",
        cmap="Reds",          # ou autre palette séquentielle
        linewidths=0.5,
        linecolor="white",
        cbar_kws={'shrink': 0.7, 'label': 'Average value'},
        annot_kws={"size": 9, "color": "black"},
        square=True,          # cellules carrées
        ax=ax
    )

    ax.set_title(
        "Clusters Interpretation (K=4) – average profiles",
        fontsize=14,
        pad=12
    )
    ax.set_xlabel("Hydraulic variables", fontsize=11)
    ax.set_ylabel("Clusters", fontsize=11)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    fig.tight_layout()
    plt.show()
else:
    print("Aucune donnée à afficher pour K=4.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Exemple pour K=3
rep_k3 = individus_pca_reset['Cluster_K3'].value_counts().sort_index()
df_rep_k3 = pd.DataFrame({'Effective': rep_k3.values}, index=[f'Cluster {i}' for i in rep_k3.index])

plt.figure(figsize=(5,3))
sns.heatmap(
    df_rep_k3,
    annot=True,
    fmt=".0f",
    cmap="Greens",
    cbar=False,
    linewidths=0.6,
    linecolor='gray',
    annot_kws={'size': 13}
)
plt.title("Distribution of effectives by cluster (K=3)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Pour K=4, même logique
rep_k4 = individus_pca_reset['Cluster_K4'].value_counts().sort_index()
df_rep_k4 = pd.DataFrame({'Effective': rep_k4.values}, index=[f'Cluster {i}' for i in rep_k4.index])
#import cmocean 
plt.figure(figsize=(5,3))
sns.heatmap(
    df_rep_k4,
    annot=True,
    fmt=".0f",
    cmap="Greens",
    cbar=False,
    linewidths=0.6,
    linecolor='gray',
    annot_kws={'size': 13}
)
plt.title("Distribution of effectives by cluster (K=4)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ===== GRAPHIQUES POURCENTAGES INDIVIDUS PAR CLUSTER =====
print("\n=== Pourcentages des individus par cluster (K=3) ===")

# Calcul des pourcentages pour chaque cluster
cluster_counts = individus_pca['Cluster_K3'].value_counts().sort_index()
cluster_percentages = (cluster_counts / len(individus_pca) * 100).round(2)

print("Répartition en pourcentage :")
for cluster, pct in cluster_percentages.items():
    print(f"Cluster {cluster}: {pct:.2f}% ({cluster_counts[cluster]} individus)")

# DataFrame pour les graphiques
df_cluster_pct = pd.DataFrame({
    'Cluster': [f'Cluster {i}' for i in cluster_percentages.index],
    'Percentage': cluster_percentages.values,
    'Effective': cluster_counts.values
})

# 1. GRAPHIQUE EN BARRES (VERTICAL)
plt.figure(figsize=(8, 6))
bars = plt.bar(df_cluster_pct['Cluster'], df_cluster_pct['Percentage'], 
               color=['#FF6B9D', '#C44569', '#F4A261', '#E76F51'] , alpha=0.8, edgecolor='black', linewidth=1.2)
plt.title("Distribution of effectives by cluster (K=3) - Percentage", fontsize=14, fontweight='bold')
plt.ylabel("Percentage (%)")
plt.xlabel("Clusters")
plt.ylim(0, max(df_cluster_pct['Percentage']) * 1.1)

# Ajout des valeurs sur les barres
for bar, pct, effectif in zip(bars, df_cluster_pct['Percentage'], df_cluster_pct['Effective']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{pct}%\n({effectif})', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 2. GRAPHIQUE EN CAMEMBERT (PIE CHART)
plt.figure(figsize=(8, 6))
colors = ['#FF6B9D', '#C44569', '#F4A261', '#E76F51'] 
wedges, texts, autotexts = plt.pie(df_cluster_pct['Percentage'], 
                                   labels=df_cluster_pct['Cluster'], 
                                   autopct='%1.1f%%', 
                                   colors=colors,
                                   startangle=90,
                                   explode=(0.05, 0.05, 0.05),  # Légère séparation
                                   textprops={'fontsize': 11, 'fontweight': 'bold'})

plt.title("Proportional distribution of individuals by cluster (K=3)", fontsize=14, fontweight='bold')
plt.axis('equal')  # Cercle parfait
plt.tight_layout()
plt.show()

# Export des résultats
df_cluster_pct.to_csv(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\pourcentages_clusters_k3.csv", index=False)
print("-> Pourcentages exportés dans 'pourcentages_clusters_k3.csv'")

# ===== GRAPHIQUES POURCENTAGES INDIVIDUS PAR CLUSTER =====
print("\n=== Pourcentages des individus par cluster (K=4) ===")

# Calcul des pourcentages pour chaque cluster
cluster_counts = individus_pca['Cluster_K4'].value_counts().sort_index()
cluster_percentages = (cluster_counts / len(individus_pca) * 100).round(2)

print("Répartition en pourcentage :")
for cluster, pct in cluster_percentages.items():
    print(f"Cluster {cluster}: {pct:.2f}% ({cluster_counts[cluster]} individus)")

# DataFrame pour les graphiques
df_cluster_pct = pd.DataFrame({
    'Cluster': [f'Cluster {i}' for i in cluster_percentages.index],
    'Percentage': cluster_percentages.values,
    'Effective': cluster_counts.values
})

# 1. GRAPHIQUE EN BARRES (VERTICAL)
plt.figure(figsize=(8, 6))
bars = plt.bar(df_cluster_pct['Cluster'], df_cluster_pct['Percentage'], 
               color=['#FF6B9D', '#C44569', '#F4A261', '#E76F51'] , alpha=0.8, edgecolor='black', linewidth=1.2)
plt.title("Distribution of effectives by cluster (K=4) - Percentage", fontsize=14, fontweight='bold')
plt.ylabel("Percentage (%)")
plt.xlabel("Clusters")
plt.ylim(0, max(df_cluster_pct['Percentage']) * 1.1)

# Ajout des valeurs sur les barres
for bar, pct, effectif in zip(bars, df_cluster_pct['Percentage'], df_cluster_pct['Effective']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{pct}%\n({effectif})', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 2. GRAPHIQUE EN CAMEMBERT (PIE CHART)
plt.figure(figsize=(8, 6))
colors = ['#FF6B9D', '#C44569', '#F4A261', '#E76F51'] 
wedges, texts, autotexts = plt.pie(df_cluster_pct['Percentage'], 
                                   labels=df_cluster_pct['Cluster'], 
                                   autopct='%1.1f%%', 
                                   colors=colors,
                                   startangle=90,
                                   explode=(0.05, 0.05, 0.05, 0.05),  # Légère séparation
                                   textprops={'fontsize': 11, 'fontweight': 'bold'})

plt.title("Proportional distribution of individuals by cluster (K=4)", fontsize=14, fontweight='bold')
plt.axis('equal')  # Cercle parfait
plt.tight_layout()
plt.show()

# Export des résultats
df_cluster_pct.to_csv(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\pourcentages_clusters_k4.csv", index=False)
print("-> Pourcentages exportés dans 'pourcentages_clusters_k4.csv'")


# ===================== K-MEANS AVEC K = 5 =====================

# K-Means avec K=5
kmeans_pca5 = KMeans(n_clusters=5, random_state=42)
individus_pca['Cluster_K5'] = kmeans_pca5.fit_predict(X_pca_plot)

print("\n== K-Means sur ACP (K=5) ==")
print(individus_pca[['Component_1', 'Component_2', 'Cluster_K5']].head())

# Nuage sur le plan factoriel (ACP, K=5)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=individus_pca['Component_1'],
    y=individus_pca['Component_2'],
    hue=individus_pca['Cluster_K5'],
    palette="tab10"
)
for i, txt in enumerate(individus_pca.index):
    plt.annotate(
        txt,
        (individus_pca.iloc[i, 0], individus_pca.iloc[i, 1]),
        fontsize=7
    )

plt.title("Clusters K-Means (ACP, K=5)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Profils moyens K=5 =====
# on travaille directement avec X et individus_pca (pas besoin de reset)
X_with_clusters_K5 = X.copy()
X_with_clusters_K5['Cluster_K5'] = individus_pca['Cluster_K5']

cluster_profiles_K5 = X_with_clusters_K5.groupby('Cluster_K5').mean()
print(cluster_profiles_K5)

if cluster_profiles_K5.size > 0:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    sns.heatmap(
        cluster_profiles_K5,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={'shrink': 0.7, 'label': 'Average value'},
        annot_kws={"size": 9, "color": "black"},
        square=True,
        ax=ax
    )

    ax.set_title(
        "Clusters Interpretation (K=5) – average profiles",
        fontsize=14,
        pad=12
    )
    ax.set_xlabel("Hydraulic variables", fontsize=11)
    ax.set_ylabel("Clusters", fontsize=11)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    fig.tight_layout()
    plt.show()
else:
    print("Aucune donnée à afficher pour K=5.")

# ===== Heatmap des effectifs K=5 =====
rep_k5 = individus_pca['Cluster_K5'].value_counts().sort_index()
df_rep_k5 = pd.DataFrame(
    {'Effective': rep_k5.values},
    index=[f'Cluster {i}' for i in rep_k5.index]
)

plt.figure(figsize=(5, 3))
sns.heatmap(
    df_rep_k5,
    annot=True,
    fmt=".0f",
    cmap="Greens",
    cbar=False,
    linewidths=0.6,
    linecolor='gray',
    annot_kws={'size': 13}
)
plt.title("Distribution of effectives by cluster (K=5)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ===== POURCENTAGES INDIVIDUS PAR CLUSTER (K=5) =====
print("\n=== Pourcentages des individus par cluster (K=5) ===")

cluster_counts = individus_pca['Cluster_K5'].value_counts().sort_index()
cluster_percentages = (cluster_counts / len(individus_pca) * 100).round(2)

print("Répartition en pourcentage :")
for cluster, pct in cluster_percentages.items():
    print(f"Cluster {cluster}: {pct:.2f}% ({cluster_counts[cluster]} individus)")

df_cluster_pct = pd.DataFrame({
    'Cluster': [f'Cluster {i}' for i in cluster_percentages.index],
    'Percentage': cluster_percentages.values,
    'Effective': cluster_counts.values
})

# 1. GRAPHIQUE EN BARRES (VERTICAL)
plt.figure(figsize=(8, 6))
colors = ['#FF6B9D', '#C44569', '#F4A261', '#E76F51', '#2A9D8F']
bars = plt.bar(
    df_cluster_pct['Cluster'],
    df_cluster_pct['Percentage'],
    color=colors,
    alpha=0.8,
    edgecolor='black',
    linewidth=1.2
)
plt.title(
    "Distribution of effectives by cluster (K=5) - Percentage",
    fontsize=14,
    fontweight='bold'
)
plt.ylabel("Percentage (%)")
plt.xlabel("Clusters")
plt.ylim(0, max(df_cluster_pct['Percentage']) * 1.1)

for bar, pct, eff in zip(bars, df_cluster_pct['Percentage'], df_cluster_pct['Effective']):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{pct}%\n({eff})',
        ha='center', va='bottom',
        fontweight='bold',
        fontsize=11
    )

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 2. GRAPHIQUE EN CAMEMBERT (PIE CHART)
plt.figure(figsize=(8, 6))
wedges, texts, autotexts = plt.pie(
    df_cluster_pct['Percentage'],
    labels=df_cluster_pct['Cluster'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=[0.05] * len(df_cluster_pct),
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)

plt.title(
    "Proportional distribution of individuals by cluster (K=5)",
    fontsize=14,
    fontweight='bold'
)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Export des résultats K=5
df_cluster_pct.to_csv(
    r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\pourcentages_clusters_k5.csv",
    index=False
)
print("-> Pourcentages exportés dans 'pourcentages_clusters_k5.csv'")

#Random Forest SANS data leakage - version corrigée

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# -- Features (PCA) et targets (variables originales)
numeric_cols = ['Flow rate (m³/s)', 'Water velocity (m/s)', 'Width (m)', 'Depth (m)',
                    'Roughness', 'Slope (%)', 'Water temperature (°C)', 'Siltation (%)']

# CORRECTION : X = composantes PCA (fit sur données nettoyées), Y = variables originales
X = X_pca[:, :4]  # 4 premières composantes (65-70% variance typiquement)
Y = data_hydraulique_clean[numeric_cols]

# Split TRAIN/TEST sur les données existantes (pour validation)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# -- Modélisation MultiOutput Random Forest
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)

# Prédiction sur train et test
Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

# -- Métriques pour chaque variable (maintenant réalistes)
print("=== Random Forest (PCA -> Variables hydrauliques) ===")
for i, col in enumerate(numeric_cols):
    mse_train = mean_squared_error(Y_train.iloc[:, i], Y_pred_train[:, i])
    mse_test = mean_squared_error(Y_test.iloc[:, i], Y_pred_test[:, i])
    r2_train = r2_score(Y_train.iloc[:, i], Y_pred_train[:, i])
    r2_test = r2_score(Y_test.iloc[:, i], Y_pred_test[:, i])
    print(f"{col}:")
    print(f"  MSE train: {mse_train:.2f}, test: {mse_test:.2f}")
    print(f"  R2 train:  {r2_train:.3f}, test: {r2_test:.3f}\n")

# -- Fusion avec clusters (ton code existant)
projets = data_hydraulique_clean.reset_index(drop=True)
acp = individus_pca.reset_index()  

nouveau_tableau = projets.merge(
    acp[['Channel', 'Component_1', 'Component_2', 'Cluster_K4']],
    left_on='Channel', right_on='Channel', how='left'
)

nouveau_tableau.rename(columns={
    'Component_1': 'PC1',
    'Component_2': 'PC2',
    'Cluster_K4': 'Cluster'
}, inplace=True)

colonnes_ordre = [
    'Channel','Flow rate (m³/s)', 'Water velocity (m/s)', 'Width (m)', 'Depth (m)',
                        'Roughness', 'Slope (%)', 'Water temperature (°C)', 'Siltation (%)','PC1','PC2','Cluster'
]							

nouveau_tableau = nouveau_tableau[colonnes_ordre]
print("\nTableau final (hyraulique + PCA + clusters):")
print(nouveau_tableau.head())

# Export
nouveau_tableau.to_excel(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\projets_clusters_complet.xlsx", index=False)
print("-> Exporté : projets_clusters_complet.xlsx")


import seaborn as sns
import matplotlib.pyplot as plt

# Sélectionner juste les colonnes numériques, comme tu as dans ton tableau
cols_heatmap = [
    'Flow rate (m³/s)', 
    'Water velocity (m/s)', 
    'Width (m)', 
    'Depth (m)',
    'Roughness', 
    'Slope (%)', 
    'Water temperature (°C)', 
    'Siltation (%)',
    'PC1',
    'PC2',
    'Cluster'
]

# Pour que les noms des projets s'affichent en ligne
plt.figure(figsize=(30, 100))
sns.heatmap(
    nouveau_tableau[cols_heatmap],
    annot=True,
    fmt=".2f",
    cmap="YlOrBr",
    linewidths=0.8,
    linecolor='grey',
    cbar_kws={'shrink': 0.8},
    #annot_kws={"size": 6}
)
plt.title("Dataset Channels", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(ticks=range(len(nouveau_tableau)), labels=nouveau_tableau['Channel'], rotation=0)
plt.tight_layout()
plt.show()

#decoupage 25 * 4
cols_heatmap = [
    'Flow rate (m³/s)', 
    'Water velocity (m/s)', 
    'Width (m)', 
    'Depth (m)',
    'Roughness', 
    'Slope (%)', 
    'Water temperature (°C)', 
    'Siltation (%)',
    'PC1',
    'PC2',
    'Cluster'
]

# paramètres
n_rows = len(nouveau_tableau)
step = 25   # 25 canaux par figure

for start in range(0, n_rows, step):
    end = min(start + step, n_rows)
    sub = nouveau_tableau.iloc[start:end]   # sous-tableau [web:27]

    plt.figure(figsize=(18, 12))            # taille adaptée pour 25 lignes [web:31]
    sns.heatmap(
        sub[cols_heatmap],
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        linewidths=0.8,
        linecolor='grey',
        cbar_kws={'shrink': 0.8},
        # annot_kws={"size": 6}  # décommente si tu veux plus petit
    )

    plt.title(f"Channels {sub['Channel'].iloc[0]} - {sub['Channel'].iloc[-1]}", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(
        ticks=range(len(sub)),
        labels=sub['Channel'],
        rotation=0
    )
    plt.tight_layout()                      # ajuste les marges [web:30]
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Entrée des nouveaux 
nouv_hydraulique = pd.DataFrame({
    "Channel": [101, 102, 103, 104, 105],
    "Flow rate (m³/s)": [35.2, 50.1, 65.3, 75.8, 20.4],
    "Water velocity (m/s)": [1.2, 2.8, 1.9, 3.1, 0.8],
    "Width (m)": [12.5, 28.3, 18.7, 22.1, 15.6],
    "Depth (m)": [2.1, 4.5, 3.8, 6.2, 1.9],
    "Roughness": [0.012, 0.028, 0.019, 0.034, 0.008],
    "Slope (%)": [1.2, 2.3, 0.9, 3.1, 1.8],
    "Water temperature (°C)": [18.5, 22.1, 15.9, 27.3, 12.7],
    "Siltation (%)": [25.1, 11.8, 18.2, 15.4, 38.6]
})

# 2. Colonnes numériques
features_cols = [
    'Flow rate (m³/s)', 
    'Water velocity (m/s)', 
    'Width (m)', 
    'Depth (m)',
    'Roughness', 
    'Slope (%)', 
    'Water temperature (°C)', 
    'Siltation (%)'
]
X_nouv = nouv_hydraulique[features_cols]

# 3. Centrage-réduction avec ton scaler entraîné (remplace scaler_tmp par scaler réel)
nouv_hydraulique_scaled = scaler.transform(X_nouv)  # scaler: StandardScaler() entraîné sur ton dataset principal

# 4. Projection ACP avec ton PCA entraîné
nouv_hydraulique_pca = pca.transform(nouv_hydraulique_scaled)  # pca: PCA() entraîné sur ton dataset principal

# 5. Prédiction du cluster sur les deux premières CP
nouv_hydraulique_pca2 = nouv_hydraulique_pca[:, :2]
nouv_hydraulique_cluster = kmeans_pca4.predict(nouv_hydraulique_pca2)  # kmeans_pca3: KMeans(n_clusters=3) sur les CP

# 6. Ajout des PC1, PC2 et du Cluster au DataFrame
nouv_hydraulique['PC1'] = nouv_hydraulique_pca[:, 0]
nouv_hydraulique['PC2'] = nouv_hydraulique_pca[:, 1]
nouv_hydraulique['Cluster_K4'] = nouv_hydraulique_cluster

# 7. Standardiser toutes les colonnes quantitatives pour la heatmap (sauf cluster)
from sklearn.preprocessing import StandardScaler
scaler_visu = StandardScaler()

data_for_heatmap = nouv_hydraulique[features_cols + ['PC1', 'PC2']].copy()
data_for_heatmap_scaled = scaler_visu.fit_transform(data_for_heatmap)

# DataFrame standardisé pour les couleurs
heatmap_df = pd.DataFrame(
    data_for_heatmap_scaled,
    columns=features_cols + ['PC1', 'PC2'],
    index=nouv_hydraulique['Channel']
)

# On ajoute le cluster en colonne (il ne sert pas aux couleurs, mais affiche le groupe)
heatmap_df['Cluster_K4'] = nouv_hydraulique['Cluster_K4'].values

# DataFrame des valeurs ORIGINALES pour les annotations
annot_df = pd.concat(
    [nouv_hydraulique[features_cols + ['PC1', 'PC2']],  # valeurs originales
     nouv_hydraulique[['Cluster_K4']]],                # cluster original
    axis=1
)

# 8. Affichage heatmap stylisée
plt.figure(figsize=(11, 3.8))
sns.heatmap(
    heatmap_df[features_cols + ['PC1', 'PC2', 'Cluster_K4']],  # standardisé -> couleurs
    annot=annot_df.values,   # valeurs originales -> texte
    fmt=".2f",               # ou "" si tu veux le format brut
    cmap="coolwarm",
    linewidths=0.6,
    linecolor='grey',
    cbar_kws={'shrink': 0.65},
    annot_kws={"size": 10}
)
plt.title("New Channels")
plt.xticks(rotation=40, ha="right")
plt.yticks(ticks=range(len(heatmap_df)), labels=heatmap_df.index, rotation=0)
plt.tight_layout()
plt.show()


# 9. Affichage du tableau complet avec cluster pour ton rapport
print(nouv_hydraulique)

# 10. Export éventuel du tableau final
nouv_hydraulique.to_excel(r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports\nouveaux_canaux_clusters_forecast.xlsx", index=False)


import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PLAN FACTORIEL ACP : Canaux existants vs Nouveaux projets (Clusters K=4)
# =========================

plt.figure(figsize=(12, 9))

# Palette pour les 4 clusters (0,1,2,3)
palette = sns.color_palette("tab10", n_colors=4)

# 1) Anciens canaux : ronds colorés par cluster K=4
ax = sns.scatterplot(
    x=individus_pca['Component_1'],
    y=individus_pca['Component_2'],
    hue=individus_pca['Cluster_K4'],
    palette=palette,
    s=80,
    alpha=0.8,
    edgecolor='grey',
    linewidth=0.5
)

# Récupérer la légende automatique (clusters) puis la laisser
handles_old, labels_old = ax.get_legend_handles_labels()
ax.legend_.remove()  # on la reconstruit proprement après

# 2) Nouveaux canaux : losanges verts, bordure noire, texte 101–105
for idx, row in nouv_hydraulique.iterrows():
    cluster_id = int(row['Cluster_K4'])  # cluster K=4 prédit pour ce canal

    # point losange vert
    plt.scatter(
        row['PC1'], row['PC2'],
        s=220,
        color='#00cc44',        # vert vif
        edgecolors='black',
        marker='D',
        linewidths=1.8,
        zorder=10
    )

    # étiquette du canal, légèrement décalée
    plt.text(
        row['PC1'] + 0.05, row['PC2'] + 0.05,
        f"Channel {int(row['Channel'])}",
        fontsize=10,
        fontweight='bold',
        color='black',
        ha='left', va='bottom'
    )

# 3) Axes, titre, grille
plt.xlabel('(PC1)', fontsize=12, fontweight='bold')
plt.ylabel('(PC2)', fontsize=12, fontweight='bold')
plt.title('Factorial plane: Existing channels vs. New channels\n(Clusters K=4)',
          fontsize=14, fontweight='bold', pad=18)

plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

from matplotlib.lines import Line2D
import numpy as np

# 4) Légende : mapping explicite Cluster 0,1,2,3
cluster_ids = np.sort(individus_pca['Cluster_K4'].unique())

cluster_handles = []
cluster_labels = []
for c in cluster_ids:
    h = Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=palette[c],
        markeredgecolor='grey',
        markersize=8,
        linestyle='None'
    )
    cluster_handles.append(h)
    cluster_labels.append(f'Cluster {c}')

new_handle = Line2D(
    [0], [0],
    marker='D',
    color='w',
    markerfacecolor='#00cc44',
    markeredgecolor='black',
    markersize=9,
    linestyle='None',
    label='New channels'
)

all_handles = cluster_handles + [new_handle]
all_labels  = cluster_labels + ['New channels']

plt.legend(
    handles=all_handles,
    labels=all_labels,
    title='Clusters',
    title_fontsize=11,
    fontsize=10,
    loc='upper right'
)

plt.show()

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display  # pour tableaux stylisés

# ============================
# 0) Réglages généraux
# ============================
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)
pd.set_option("display.float_format", "{:,.6f}".format)

sns.set(style="white")

# Chemins
data_path = r"C:\Users\achra\Desktop\OuvragesHydrauliques\tableau_contingence.xlsx"
output_dir = r"C:\Users\achra\Desktop\OuvragesHydrauliques\exports"

# ============================
# 1) Lecture des données
# ============================
data = pd.read_excel(data_path, index_col=0)
print("1) Tableau de contingence (canaux × critères) :\n")
print(data)

# ============================
# 2) Matrice des fréquences P
# ============================
n = data.values.sum()
P = data / n

print("\n2) Matrice des fréquences P = data / n :\n")
print(P)

plt.figure(figsize=(18, 8))
sns.heatmap(
    P, cmap="Greens",
    xticklabels=True, yticklabels=True,
    annot=True, fmt=".003f",
    #annot_kws={"fontsize": 6}
)
plt.title("Frequency Matrix")
plt.xlabel("Criteria")
plt.ylabel("Channels")
plt.tight_layout()
plt.show()

# ============================
# 3) Masses lignes et colonnes
# ============================
r = P.sum(axis=1).values.reshape(-1, 1)   # masses lignes
c = P.sum(axis=0).values.reshape(1, -1)   # masses colonnes

masses_lignes_df = pd.DataFrame(r, index=data.index, columns=["Mass_row"])
masses_colonnes_df = pd.DataFrame(c.T, index=data.columns, columns=["Masse_column"])

print("\n3) Masses lignes (r) :\n")
print(masses_lignes_df)
print("\n3) Masses colonnes (c) :\n")
print(masses_colonnes_df)

# --- Masses lignes : tableau stylisé + heatmap
print("\nMasses lignes (tableau stylisé) :\n")
display(masses_lignes_df.style
        .format("{:.6f}")
        .background_gradient(cmap="Oranges"))

plt.figure(figsize=(4, 8))
sns.heatmap(
    masses_lignes_df,
    cmap="Oranges",
    annot=True, fmt=".004f",
    annot_kws={"fontsize": 7},
    cbar=True
)
plt.title("Row masses")
plt.tight_layout()
plt.show()

# --- Masses colonnes : tableau stylisé + heatmap
print("\nMasses colonnes (tableau stylisé) :\n")
display(masses_colonnes_df.style
        .format("{:.6f}")
        .background_gradient(cmap="Oranges"))

plt.figure(figsize=(10, 3))
sns.heatmap(
    masses_colonnes_df.T,     # 1 × 30
    cmap="Oranges",
    annot=True, fmt=".004f",
    annot_kws={"fontsize": 7},
    cbar=True
)
plt.title("Column masses")
plt.tight_layout()
plt.show()

# ============================
# 4) Matrice des écarts S
# ============================
expected = r @ c
S = (P - expected) / np.sqrt(expected)
S_df = pd.DataFrame(S, index=data.index, columns=data.columns)

print("\n4) Matrice des écarts à l'indépendance S :\n")
print(S_df)

plt.figure(figsize=(16, 8))
sns.heatmap(
    S_df,
    fmt=".003f",
    cmap="coolwarm",
    center=0,
    xticklabels=True, yticklabels=True,
    annot=True  # True si tu veux les valeurs (mais 30×30 -> illisible)
)
plt.title("Matrix of deviations from independence S")
plt.xlabel("Criteria")
plt.ylabel("Channels")
plt.tight_layout()
plt.show()

# ============================
# 5) SVD, valeurs propres, inertie
# ============================
U, s, Vt = np.linalg.svd(S, full_matrices=False)
eigenvalues = s**2
inertie_totale = eigenvalues.sum()
explained_inertia = eigenvalues / inertie_totale

print("\n5) Valeurs singulières (s) :\n", s)
print("\n6) Valeurs propres :\n", eigenvalues)
print("\n6) Inertie totale :", inertie_totale)
print("\n6) Inertie expliquée par axe :\n", explained_inertia)

valeurs_df = pd.DataFrame({
    "Singular_values": s,
    "Eigenvalues": eigenvalues,
    "Inertia_explained": explained_inertia
})

print("\nValeurs singulières / propres (tableau stylisé) :\n")
display(valeurs_df.style
        .format("{:.6f}")
        .background_gradient(cmap="Blues"))

vp = valeurs_df[["Eigenvalues"]]

plt.figure(figsize=(6, 4))
sns.heatmap(
    vp,
    cmap="Blues",
    annot=True, fmt=".004f",
    annot_kws={"fontsize": 7},
    cbar=True
)
plt.title("Eigenvalues")
plt.tight_layout()
plt.show()

# Heatmap verticale pour les valeurs singulières
vs = valeurs_df[["Singular_values"]]   # 30 lignes × 1 colonne

plt.figure(figsize=(6, 4))
sns.heatmap(
    vs,
    cmap="Purples",
    annot=True, fmt=".004f",
    annot_kws={"fontsize": 7},
    cbar=True,
    yticklabels=True,
    xticklabels=True
)
plt.title("Singular values")
plt.tight_layout()
plt.show()

# ============================
# Inertie expliquée : tableau stylisé
# ============================
inertie_df = pd.DataFrame({
    "Axe": np.arange(1, len(explained_inertia) + 1),
    "Inertia_explained": explained_inertia
})

print("\nInertie expliquée par axe (tableau stylisé) :\n")
display(inertie_df.style
        .format({"Inertia_explained": "{:.4f}"})
        .background_gradient(cmap="Greens"))

# ============================
# Heatmap verticale de l'inertie expliquée
# ============================
ie = inertie_df.set_index("Axe")[["Inertia_explained"]]  # index = Axe

plt.figure(figsize=(6, 4))
sns.heatmap(
    ie,
    cmap="Greens",
    annot=True, fmt=".004f",
    annot_kws={"fontsize": 7},
    cbar=True,
    yticklabels=True,
    xticklabels=True
)
plt.title("Inertia explained by axis")
plt.tight_layout()
plt.show()


# ============================
# Inertie expliquée : graphique
# ============================

# 1) Graphe en barres
plt.figure(figsize=(8, 4))
plt.bar(inertie_df["Axe"], inertie_df["Inertia_explained"],
        color="seagreen", edgecolor="black")
plt.xlabel("Factorial axe")
plt.ylabel("Inertia explained")
plt.title("Inertia explained per axe")
plt.xticks(inertie_df["Axe"])
plt.tight_layout()
plt.show()

# 2) (optionnel) Courbe cumulée
inertie_cumulee = inertie_df["Inertia_explained"].cumsum()

plt.figure(figsize=(8, 4))
plt.plot(inertie_df["Axe"], inertie_cumulee, marker="o", color="darkblue")
plt.xlabel("Factorial axe")
plt.ylabel("Cumulative inertia")
plt.title("Cumulative inertia of the axes")
plt.xticks(inertie_df["Axe"])
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



# ============================
# 6) Test du Khi-deux
# ============================
chi2_from_inertia = n * inertie_totale
chi2_scipy, p_value, ddl, expected_scipy = chi2_contingency(data)

print("\n7) Khi-deux via AFC : ", chi2_from_inertia)
print("7) Khi-deux via scipy :", chi2_scipy)
print("7) p-value :", p_value)
print("7) ddl :", ddl)

# Contributions de chaque cellule au chi²
chi2_cell = (data - expected * n)**2 / (expected * n)   # ou n * (P - expected)**2 / expected
chi2_cell_df = pd.DataFrame(chi2_cell, index=data.index, columns=data.columns)

plt.figure(figsize=(18, 8))
sns.heatmap(
    chi2_cell_df,
    cmap="Reds",
    annot=True, fmt=".3f",
    #annot_kws={"fontsize": 6},
    cbar=True
)
plt.title("Contributions of cells to χ²", fontsize=12)
plt.xlabel("Criteria")
plt.ylabel("Channels")
plt.tight_layout()
plt.show()

# Contribution totale de chaque ligne au χ²
chi2_lignes = chi2_cell_df.sum(axis=1)

plt.figure(figsize=(10, 5))
chi2_lignes.sort_values(ascending=False).plot(
    kind="bar", color="steelblue", edgecolor="black"
)
plt.ylabel("Part of the total χ²", fontsize=11)
plt.title("Contributions of rows to χ²", fontsize=12)
plt.xlabel("Channels")
plt.tight_layout()
plt.show()

# Contribution totale de chaque colonne au χ²
chi2_colonnes = chi2_cell_df.sum(axis=0)

plt.figure(figsize=(10, 5))
chi2_colonnes.sort_values(ascending=False).plot(
    kind="bar", color="indianred", edgecolor="black"
)
plt.ylabel("Part of the total χ²", fontsize=11)
plt.title("Contributions of columns to χ²", fontsize=12)
plt.xlabel("Criteria")
plt.tight_layout()
plt.show()

#Jauge / barre pour la p‑value
alpha = 0.05

plt.figure(figsize=(6, 3))
plt.hlines(1, 1e-25, alpha, colors="lightgray", linewidth=12, label="Threshold 0.05")
plt.hlines(1, 1e-25, p_value, colors="red", linewidth=12, label=f"p-value = {p_value:.2e}")
plt.xscale("log")
plt.xlabel("Significance level (log scale)")
plt.title("Position of the p-value relative to the 0.05 threshold")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


#Histogramme théorique vs χ² observé
from scipy.stats import chi2

# Densité théorique du chi² avec ddl degrés de liberté
x = np.linspace(0, chi2.ppf(0.999, ddl), 500)
y = chi2.pdf(x, ddl)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label=f"Lois χ² (ddl = {ddl})", color="steelblue")
plt.axvline(chi2_from_inertia, color="red", linestyle="--",
            label=f"χ² observed = {chi2_from_inertia:.1f}")
plt.xlabel("Value of χ²")
plt.ylabel("Density")
plt.title("Position of the observed statistic in the theoretical χ² distribution")
plt.legend()
plt.tight_layout()
plt.show()



# ============================
# 7) Coordonnées factorielles
# ============================
dim = 2
F = U[:, :dim] @ np.diag(s[:dim])           # lignes
G = Vt.T[:, :dim] @ np.diag(s[:dim])        # colonnes

coords_lignes = pd.DataFrame(F, index=data.index,
                             columns=[f"F{i+1}" for i in range(dim)])
coords_colonnes = pd.DataFrame(G, index=data.columns,
                               columns=[f"F{i+1}" for i in range(dim)])

print("\n8) Coordonnées factorielles des lignes (chantiers) :\n")
print(coords_lignes)
print("\n8) Coordonnées factorielles des colonnes (critères) :\n")
print(coords_colonnes)

print("\nCoordonnées factorielles des lignes (tableau stylisé) :\n")
display(coords_lignes.style
        .format("{:.4f}")
        .background_gradient(cmap="PuBu"))

plt.figure(figsize=(6, 8))
sns.heatmap(
    coords_lignes,
    cmap="PuBu",
    annot=True, fmt=".003f",
    annot_kws={"fontsize": 7},
    cbar=True
)
plt.title("Factorial coordinates of the rows (F1, F2)")
plt.tight_layout()
plt.show()

print("\nCoordonnées factorielles des colonnes (tableau stylisé) :\n")
display(coords_colonnes.style
        .format("{:.4f}")
        .background_gradient(cmap="PuRd"))

plt.figure(figsize=(10, 6))
sns.heatmap(
    coords_colonnes,
    cmap="PuRd",
    annot=True, fmt=".003f",
    annot_kws={"fontsize": 7},
    cbar=True
)
plt.title("Factorial coordinates of the columns (F1, F2)")
plt.tight_layout()
plt.show()

# ============================
# 8) Plan factoriel F1 × F2
# ============================
plt.figure(figsize=(8, 6))

# Lignes (chantiers)
plt.scatter(coords_lignes["F1"], coords_lignes["F2"],
            color="blue", label="Channels")
for i, name in enumerate(coords_lignes.index):
    plt.text(coords_lignes.iloc[i, 0], coords_lignes.iloc[i, 1],
             name, color="blue", fontsize=8)

# Colonnes (critères)
plt.scatter(coords_colonnes["F1"], coords_colonnes["F2"],
            color="red", marker="s", label="Criteria")
for j, name in enumerate(coords_colonnes.index):
    plt.text(coords_colonnes.iloc[j, 0], coords_colonnes.iloc[j, 1],
             name, color="red", fontsize=8)

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Factorial Plan of associations (F1 × F2)")
plt.legend()
plt.tight_layout()
plt.show()


# Détection d'anomalies avec Isolation Forest et LOF (8 CANAUX)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import pandas as pd
import numpy as np

# ============================
# 1) Charger les données cyber
# ============================
df_cyber = pd.read_excel(r"C:\Users\achra\Desktop\OuvragesHydrauliques\donnees_cyber_8canaux.xlsx", index_col=0)
print("Données cyber chargées (8 canaux) :\n")
print(df_cyber)

# Joindre avec ton tableau de contingence 'data' (8x8)
data_with_cyber = df_cyber.copy()
#data_with_cyber = data.join(df_cyber)
print("\n=== Data + variables cyber (8x8) ===\n")
print(data_with_cyber.head())

# ============================
# 2) Détection d'anomalies
# ============================
features = ["SCADA_Alerts",
            "DDoS_Attacks",
            "Command_latency",
            "Overall_risk"]				

X = data_with_cyber[features].values

# Normalisation
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
iso = IsolationForest(contamination=0.2, random_state=42)
data_with_cyber["IF_Prediction"] = iso.fit_predict(X_scaled)

# LOF
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.2)
data_with_cyber["LOF_Prediction"] = lof.fit_predict(X_scaled)

print("\n=== Canaux avec prédictions d'anomalies ===")
print(data_with_cyber[features + ["IF_Prediction", "LOF_Prediction"]])

# ============================
# 3) Tableau stylisé des anomalies
# ============================
anomaly_df = data_with_cyber[features + ["IF_Prediction", "LOF_Prediction"]]

def color_anomalies_if(col):
    return ['background-color: #ffcccc' if val == -1 else '' for val in col]

def color_anomalies_lof(col):
    return ['background-color: #ff9999' if val == -1 else '' for val in col]

print("\n=== Canaux à risque (ROUGE = Anomalie) ===\n")
display(
    anomaly_df.style
    .format({
        "SCADA_Alerts": "{:.0f}",
        "DDoS_Attacks": "{:.0f}",
        "Command_latency": "{:.0f}",
        "Overall_risk": "{:.2f}", 

    })
    .background_gradient(cmap="YlOrRd", subset=features)
    .apply(color_anomalies_if, subset=["IF_Prediction"], axis=0)
    .apply(color_anomalies_lof, subset=["LOF_Prediction"], axis=0)
)

# ============================
# 4) FIGURE 1 : Variables cyber par canal
# ============================
plt.figure(figsize=(10, 8))
sns.heatmap(
    anomaly_df[features],
    cmap="YlOrRd",
    annot=True, fmt=".1f",
    cbar_kws={'label': 'Security values'}
)
plt.title("Security values per channel")
plt.xlabel("Security values")
plt.ylabel("Channels")
plt.tight_layout()
plt.show()

# ============================
# 5) FIGURE 2 : Prédictions IF / LOF
# ============================
plt.figure(figsize=(8, 4))
pred_bin = data_with_cyber[["IF_Prediction", "LOF_Prediction"]]
sns.heatmap(
    pred_bin.T,
    cmap="RdYlGn_r",
    annot=True, fmt="d",
    cbar_kws={'label': '-1 = Anomalia, 1 = Normal'}
)
plt.title("Predictions IF / LOF per channel")
plt.xlabel("Channels")
plt.ylabel("Algorithmes")
plt.tight_layout()
plt.show()

# ============================
# 6) Graphiques de dispersion
# ============================
plt.figure(figsize=(12, 5))

# Isolation Forest
plt.subplot(1, 2, 1)
colors_if = np.where(data_with_cyber["IF_Prediction"] == -1, "red", "green")
plt.scatter(
    data_with_cyber["SCADA_Alerts"],
    data_with_cyber["DDoS_Attacks"],
    c=colors_if, s=100, edgecolor="black"
)
for canal in data_with_cyber.index:
    x = data_with_cyber.loc[canal, "SCADA_Alerts"]
    y = data_with_cyber.loc[canal, "DDoS_Attacks"]
    color = "red" if data_with_cyber.loc[canal, "IF_Prediction"] == -1 else "green"
    plt.text(x + 0.2, y + 0.2, canal, fontsize=9, color=color, fontweight="bold")

plt.xlabel("SCADA Alerts")
plt.ylabel("DDoS Attacks")
plt.title("Isolation Forest : Risky channels (RED)")

# LOF
plt.subplot(1, 2, 2)
colors_lof = np.where(data_with_cyber["LOF_Prediction"] == -1, "red", "green")
plt.scatter(
    data_with_cyber["SCADA_Alerts"],
    data_with_cyber["DDoS_Attacks"],
    c=colors_lof, s=100, edgecolor="black"
)
for canal in data_with_cyber.index:
    x = data_with_cyber.loc[canal, "SCADA_Alerts"]
    y = data_with_cyber.loc[canal, "DDoS_Attacks"]
    color = "red" if data_with_cyber.loc[canal, "LOF_Prediction"] == -1 else "green"
    plt.text(x + 0.2, y + 0.2, canal, fontsize=9, color=color, fontweight="bold")

plt.xlabel("SCADA Alerts")
plt.ylabel("DDoS Attacks")
plt.title("LOF : Risky channels (RED)")
plt.tight_layout()
plt.show()

# ============================
# 7) Résumé des anomalies
# ============================
anomalies = data_with_cyber[
    (data_with_cyber["IF_Prediction"] == -1) |
    (data_with_cyber["LOF_Prediction"] == -1)
]
print(f"\nCANAUX ANOMALES détectés : {len(anomalies)} / 8")
print(anomalies[["Overall_risk", "IF_Prediction", "LOF_Prediction"]])


# ============================
# 9) Export Excel (fichiers séparés)
# ============================
data.to_excel(f"{output_dir}\\01_Donnees_brutes.xlsx")
P.to_excel(f"{output_dir}\\02_Frequences_P.xlsx")
S_df.to_excel(f"{output_dir}\\03_Ecarts_S.xlsx")
masses_lignes_df.to_excel(f"{output_dir}\\04_Masses_lignes.xlsx")
masses_colonnes_df.to_excel(f"{output_dir}\\05_Masses_colonnes.xlsx")

inertie_df = pd.DataFrame({
    "Valeurs singulières": s,
    "Valeurs propres": eigenvalues,
    "Inertie expliquée": explained_inertia
})
inertie_df.to_excel(f"{output_dir}\\06_Inertie.xlsx")

chi2_df = pd.DataFrame({
    "Statistique": ["Chi² (AFC)", "Chi² (scipy)", "p-value", "ddl"],
    "Valeur": [chi2_from_inertia, chi2_scipy, p_value, ddl]
})
chi2_df.to_excel(f"{output_dir}\\07_Khi_deux.xlsx", index=False)

coords_lignes.to_excel(f"{output_dir}\\08_Coord_Lignes.xlsx")
coords_colonnes.to_excel(f"{output_dir}\\09_Coord_Colonnes.xlsx")
#data_with_cyber.to_excel(f"{output_dir}\\10_Chantiers_avec_cyber_et_anomalies.xlsx")

print("Tous les fichiers ont été exportés !")