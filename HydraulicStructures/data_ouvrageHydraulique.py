# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 18:34:57 2025

@author: achra
"""

# data_aero.py (version découpée)
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Élimine le warning KMeans Windows

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sys, os


from scipy.stats import chi2_contingency, chi2
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

def get_data_path(filename):
    if getattr(sys, 'frozen', False):
        # PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Développement normal
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)

sns.set(style="white")

#afc_data_path = r"C:\Users\achra\Desktop\4IIRG7\Analyse de données (Data Analysis)\Final PRoject DATA ANALYST Par Professeur EL MKHALET MOUNA\projet Ouvrages hydrauliques\tableau_contingence.xlsx"
#afc_output_dir = r"...\exports"
#cyber_data_path = r"C:\Users\achra\Desktop\4IIRG7\Analyse de données (Data Analysis)\Final PRoject DATA ANALYST Par Professeur EL MKHALET MOUNA\projet Ouvrages hydrauliques\donnees_cyber_8canaux.xlsx"

afc_data_path = get_data_path("tableau_contingence.xlsx")
cyber_data_path = get_data_path("donnees_cyber_8canaux.xlsx")


# ================== CHARGEMENT & PRETRAITEMENT GLOBAL ==================

#file_path = r'C:\Users\achra\Desktop\4IIRG7\Analyse de données (Data Analysis)\Final PRoject DATA ANALYST Par Professeur EL MKHALET MOUNA\projet Ouvrages hydrauliques\Ouvrages_hydrauliques.xlsx'
file_path = get_data_path("Ouvrages_hydrauliques.xlsx")
data_hydraulique = pd.read_excel(file_path, header=0)

data_hydraulique.columns = [
    'Channel', 'Flow rate (m³/s)', 'Water velocity (m/s)', 'Width (m)', 'Depth (m)',
                        'Roughness', 'Slope (%)', 'Water temperature (°C)', 'Siltation (%)'
]
						
numeric_cols = [
    'Flow rate (m³/s)', 'Water velocity (m/s)', 'Width (m)', 'Depth (m)',
                        'Roughness', 'Slope (%)', 'Water temperature (°C)', 'Siltation (%)'
]
    
for col in numeric_cols:
    data_hydraulique[col] = (
        data_hydraulique[col]
        .astype(str).str.strip().str.replace(' ', '')
        .str.replace(',', '.')          
        .replace('', np.nan)
    )
    data_hydraulique[col] = pd.to_numeric(data_hydraulique[col], errors='coerce') 

data_hydraulique_clean = data_hydraulique.dropna()

X = data_hydraulique_clean[numeric_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=numeric_cols,
    index=data_hydraulique_clean['Channel']  
)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

individus_pca = pd.DataFrame(
    X_pca,
    columns=[f"Component_{i+1}" for i in range(X_pca.shape[1])],
    index=data_hydraulique_clean['Channel']
)

variables_pca = pd.DataFrame(
    pca.components_,
    columns=numeric_cols,
    index=[f"Component_{i+1}" for i in range(len(numeric_cols))]
)

X_pca_plot = individus_pca.iloc[:, :2]
kmeans_pca4 = KMeans(n_clusters=4, random_state=42)
individus_pca['Cluster_K4'] = kmeans_pca4.fit_predict(X_pca_plot)

# ================== PARTIE I : FONCTIONS ==================

def afficher_moyenne_ecart_type():
    print("=== Tableau Moyenne et Écart-type ===\n")
    moyennes = X.mean()
    ecarts_type = X.std()
    for c in numeric_cols:
        print(f"{c} : Moyenne = {moyennes[c]:.2f}, Écart-type = {ecarts_type[c]:.2f}")
    print()
    
    
def get_moyenne_ecart_type():
    moyennes = X.mean()
    ecarts_type = X.std()
    df = pd.DataFrame({
        "Variable": numeric_cols,
        "Mean": [moyennes[c] for c in numeric_cols],
        "Standard deviation": [ecarts_type[c] for c in numeric_cols],
    })
    return df


def afficher_matrice_centree_reduite():
    print("=== Matrice centrée-réduite (extrait) ===\n")

    parts = [
        X_scaled_df.iloc[0:25, :],
        X_scaled_df.iloc[25:50, :],
        X_scaled_df.iloc[50:75, :],
        X_scaled_df.iloc[75:100, :],
    ]
    titles = ["Channels 1-25", "Channels 26-50", "Channels 51-75", "Channels 76-100"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))  # moins large mais 4 sous‑plots

    vmin = X_scaled_df.values.min()
    vmax = X_scaled_df.values.max()

    for part, ax, title in zip(parts, axes.ravel(), titles):
        sns.heatmap(
            part,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=vmin,
            vmax=vmax,          # même échelle de couleurs partout[web:42][web:47]
            linewidths=0.5,
            linecolor='gray',
            cbar=False,
            annot_kws={"size": 5},
            ax=ax
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    fig.suptitle("Standardized Data Matrix – 4 blocks of 25 channels", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def get_matrice_centree_reduite():
    """Retourne le DataFrame de la matrice centrée-réduite."""
    return X_scaled_df.copy()

def afficher_heatmap_correlation():
    print("=== Matrice de corrélation ===\n")

    corr_df = pd.DataFrame(
        np.corrcoef(X_scaled.T),
        index=numeric_cols,
        columns=numeric_cols
    )
    print(corr_df.round(2))

    fig, ax = plt.subplots(figsize=(12, 6))   # taille raisonnable (comme MCR réduite)

    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
        ax=ax
    )
    ax.set_title("Correlation Matrix", pad=10, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.tight_layout()
    return fig

def calculer_inerties():
    print("=== Valeurs propres et inerties ===\n")
    valeurs_propres = pca.explained_variance_
    inertie_expliquee = pca.explained_variance_ratio_
    inertie_cumulee = np.cumsum(inertie_expliquee)

    for i, val in enumerate(valeurs_propres, 1):
        print(
            f"Component {i} : {val:.4f} | Inertia : {inertie_expliquee[i-1]*100:.2f}% "
            f"(cumulée {inertie_cumulee[i-1]*100:.2f}%)"
        )
    print()

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(inertie_expliquee)+1),
        inertie_expliquee*100,
        marker='o',
        label='Inertia % per component'
    )
    plt.plot(
        range(1, len(inertie_cumulee)+1),
        inertie_cumulee*100,
        marker='s',
        label='Cumulative Inertia %'
    )
    plt.title("Inertia explained by principal components")
    plt.xlabel("Main components")
    plt.ylabel("Percentage of inertia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_inerties_table():
    valeurs_propres = pca.explained_variance_
    inertie_expliquee = pca.explained_variance_ratio_
    inertie_cumulee = np.cumsum(inertie_expliquee)

    df = pd.DataFrame({
        "Component": [f"Component {i+1}" for i in range(len(valeurs_propres))],
        "Eigenvalues": valeurs_propres,
        "Inertia (%)": inertie_expliquee * 100,
        "Cumulative Inertia (%)": inertie_cumulee * 100
    })
    return df

# ================== PARTIE II : FONCTIONS ==================

def plan_factoriel_individus():
    print("=== Plan factoriel contenant les individus ===\n")
    
    # Garder directement l’étiquette de l’index
    labels = [str(i) for i in individus_pca.index]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    #fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    ax.scatter(individus_pca['Component_1'], individus_pca['Component_2'])
    for i, txt in enumerate(labels):
        x = individus_pca.iloc[i, 0]
        y = individus_pca.iloc[i, 1]
        ax.annotate(txt, (x, y), fontsize=7)
    ax.set_title("Factorial plane of individuals")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)
    fig.tight_layout()
    return fig


def cercle_correlation():
    print("=== Cercle de corrélation dans le plan factoriel ===\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(numeric_cols)):
        ax.arrow(
            0, 0,
            variables_pca.loc["Component_1", numeric_cols[i]],
            variables_pca.loc["Component_2", numeric_cols[i]],
            head_width=0.03,
            head_length=0.03,
            color='red'
        )
        ax.text(
            variables_pca.loc["Component_1", numeric_cols[i]]*1.08,
            variables_pca.loc["Component_2", numeric_cols[i]]*1.08,
            numeric_cols[i],
            color='blue',
            fontsize=8
        )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.set_title("Correlation Cercle")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)
    fig.tight_layout()

    return fig


def qualite_representation():
    print("=== Tableau de qualité de représentation des individus ===\n")
    coord_carres = individus_pca**2
    qualite_rep = pd.DataFrame()
    qualite_rep['Quality Axe 1'] = coord_carres['Component_1'] / np.sum(coord_carres['Component_1'])
    qualite_rep['Quality Axe 2'] = coord_carres['Component_2'] / np.sum(coord_carres['Component_2'])
    qualite_rep['Factorial Plan Quality'] = qualite_rep['Quality Axe 1'] + qualite_rep['Quality Axe 2']

    print(qualite_rep.head().round(3))

    # Découpage en 4 parties comme la MCR (100 individus → 25 par bloc)
    parts = [
        qualite_rep.iloc[0:25],
        qualite_rep.iloc[25:50],
        qualite_rep.iloc[50:75],
        qualite_rep.iloc[75:100],
    ]
    titles = ["Individuals 1-25", "Individuals 26-50", "Individuals 51-75", "Individuals 76-100"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Même échelle de couleurs pour toutes les heatmaps
    vmin = qualite_rep.values.min()
    vmax = qualite_rep.values.max()

    for part, ax, title in zip(parts, axes.ravel(), titles):
        sns.heatmap(
            part,
            annot=True,
            fmt=".3f",
            cmap="crest",
            vmin=vmin,
            vmax=vmax,  # même échelle partout
            linewidths=0.5,
            linecolor='gray',
            cbar=False,  # une seule barre de couleur
            annot_kws={"size": 5},
            ax=ax
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    fig.suptitle("Quality of representation of individuals – 4 blocks of 25", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

# ================== PARTIE III : FONCTIONS ==================

# ===== KMeans sur PC1-PC2 : K=3,4,5 =====
X_pca_plot = individus_pca.iloc[:, :2]

kmeans_pca3 = KMeans(n_clusters=3, random_state=42, n_init="auto")
kmeans_pca4 = KMeans(n_clusters=4, random_state=42, n_init="auto")
kmeans_pca5 = KMeans(n_clusters=5, random_state=42, n_init="auto")

individus_pca["Cluster_K3"] = kmeans_pca3.fit_predict(X_pca_plot)
individus_pca["Cluster_K4"] = kmeans_pca4.fit_predict(X_pca_plot)
individus_pca["Cluster_K5"] = kmeans_pca5.fit_predict(X_pca_plot)

def prediction_nouveaux_individus(k=4):
    print(f"=== Prédiction / projection de nouveaux individus avec Random Forest + ACP (K={k}) ===\n")

    models = {3: kmeans_pca3, 4: kmeans_pca4, 5: kmeans_pca5}
    kmeans_model = models.get(k)
    if kmeans_model is None:
        raise ValueError("k doit être 3, 4 ou 5")

    col_cluster = f"Cluster_K{k}"

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

    # ----- Projection ACP des nouveaux individus -----
    X_nouv = nouv_hydraulique[numeric_cols]
    nouv_scaled = scaler.transform(X_nouv)
    nouv_pca = pca.transform(nouv_scaled)

    nouv_hydraulique["PC1"] = nouv_pca[:, 0]
    nouv_hydraulique["PC2"] = nouv_pca[:, 1]

    # ----- Cluster avec le modèle choisi (K=3/4/5) -----
    nouv_hydraulique[col_cluster] = kmeans_model.predict(nouv_pca[:, :2])

    # ----- Random Forest (inchangé) -----
    Y = data_hydraulique_clean[numeric_cols]
    X_rf = X_pca[:, :4]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_rf, Y)

    pred_nouv = model.predict(nouv_pca[:, :4])
    pred_df = pd.DataFrame(pred_nouv, columns=numeric_cols)
    pred_df.insert(0, "Channel", nouv_hydraulique["Channel"])

    print("Prédictions Random Forest pour les nouveaux individus :")
    print(pred_df.round(2))
    print()

    # ===== Figure ACP anciens + nouveaux =====
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=k)

    sns.scatterplot(
        x=individus_pca["Component_1"],
        y=individus_pca["Component_2"],
        hue=individus_pca[col_cluster],
        palette=palette,
        s=70,
        legend=False,
        alpha=0.7,
        edgecolor="grey",
        linewidth=0.3,
        ax=ax
    )

    for _, row in nouv_hydraulique.iterrows():
        ax.scatter(
            row["PC1"], row["PC2"],
            s=130,
            c=[palette[int(row[col_cluster])]],
            marker="P",
            edgecolor="black",
            linewidths=1.8,
            zorder=5
        )
        ax.text(
            row["PC1"], row["PC2"],
            f"Channel {int(row['Channel'])}",
            fontsize=9,
            color="black",
            ha="left",
            va="bottom",
            weight="bold"
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Old and new Channels (clusters K={k})")
    ax.grid(alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=palette[i], markersize=8,
                   label=f"Cluster {i}")
        for i in range(k)
    ] + [
        plt.Line2D([0], [0], marker="P", color="k",
                   label="New Channel", markersize=10,
                   linestyle="None")
    ]
    ax.legend(handles=handles, loc="best")

    fig.tight_layout()
    return fig

def afficher_clusters_k(k=4):
    print(f"=== Affichage des clusters (KMeans sur PC1-PC2, K={k}) ===\n")

    col = f"Cluster_K{k}"
    if col not in individus_pca.columns:
        raise ValueError(f"{col} n'existe pas. Vérifie que tu as bien calculé les clusters K={k}.")

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("Set2", n_colors=k)

    sns.scatterplot(
        x=individus_pca["Component_1"],
        y=individus_pca["Component_2"],
        hue=individus_pca[col],
        palette=palette,
        ax=ax
    )

    labels = [str(i) for i in individus_pca.index]
    for i, txt in enumerate(labels):
        ax.annotate(txt,
                    (individus_pca.iloc[i, 0], individus_pca.iloc[i, 1]),
                    fontsize=7)

    ax.set_title(f"Clusters K-Means (K={k})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)
    fig.tight_layout()
    return fig

def pourcentage_clusters(k=4):
    print(f"=== Pourcentage de chaque cluster (K={k}) ===\n")

    col = f"Cluster_K{k}"
    if col not in individus_pca.columns:
        raise ValueError(f"{col} n'existe pas. Vérifie que tu as bien calculé les clusters K={k}.")

    cluster_counts = individus_pca[col].value_counts().sort_index()
    cluster_percentages = (cluster_counts / len(individus_pca) * 100).round(2)

    for cluster, pct in cluster_percentages.items():
        print(f"Cluster {cluster}: {pct:.2f}% ({cluster_counts[cluster]} individus)")

    df_cluster_pct = pd.DataFrame({
        "Cluster": [f"Cluster {i}" for i in cluster_percentages.index],
        "Percentage": cluster_percentages.values,
        "Effective": cluster_counts.values
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#14b8a6", "#ec4899", "#fbbf24", "#6366f1", "#84cc16"]  # tu prends k couleurs
    bars = ax.bar(df_cluster_pct["Cluster"], df_cluster_pct["Percentage"],
                  color=colors[:k], alpha=0.85, edgecolor="black")

    #bars = ax.bar(df_cluster_pct["Cluster"], df_cluster_pct["Pourcentage"], alpha=0.85, edgecolor="black")

    ax.set_title(f"Distribution of individuals by cluster (K={k}) - Percentage")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Clusters")
    ax.set_ylim(0, max(df_cluster_pct["Percentage"]) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    for bar, pct, eff in zip(bars, df_cluster_pct["Percentage"], df_cluster_pct["Effective"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{pct}%\n({eff})",
                ha="center", va="bottom",
                fontweight="bold", fontsize=9)

    fig.tight_layout()
    return fig

def metriques_random_forest(k=4):
    print(f"=== Métriques Random Forest (ACP → variables originales) avec {k} composantes ===\n")

    Y = data_hydraulique_clean[numeric_cols]   # cibles
    X_rf = X_pca[:, :k]                        # k premières composantes

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_rf, Y, train_size=0.7, random_state=42
    )

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    lignes = []
    for i, col in enumerate(numeric_cols):
        mse_train = mean_squared_error(Y_train.iloc[:, i], Y_pred_train[:, i])
        mse_test = mean_squared_error(Y_test.iloc[:, i], Y_pred_test[:, i])
        r2_train = r2_score(Y_train.iloc[:, i], Y_pred_train[:, i])
        r2_test = r2_score(Y_test.iloc[:, i], Y_pred_test[:, i])

        lignes.append({
            "Variable": col,
            "MSE train": mse_train,
            "MSE test": mse_test,
            "R2 train": r2_train,
            "R2 test": r2_test,
        })

    return pd.DataFrame(lignes)

# ================== AFC & CYBER : PREPARATION GLOBALE ==================

afc_data = pd.read_excel(afc_data_path, index_col=0)
afc_data_affichage = afc_data.reset_index()
n_afc = afc_data.values.sum()
P_afc = afc_data / n_afc

r_afc = P_afc.sum(axis=1).values.reshape(-1, 1)
c_afc = P_afc.sum(axis=0).values.reshape(1, -1)
expected_afc = r_afc @ c_afc
S_afc = (P_afc - expected_afc) / np.sqrt(expected_afc)
U_afc, s_afc, Vt_afc = np.linalg.svd(S_afc, full_matrices=False)
eigen_afc = s_afc**2
inertie_totale_afc = eigen_afc.sum()
explained_inertia_afc = eigen_afc / inertie_totale_afc

def afc_import_data_tableau():
    """Import data + tableau de contingence (AFC)."""
    # ici on renvoie simplement le tableau brut
    return afc_data_affichage.copy()

def afc_matrice_frequences():
    """Figure de la matrice des fréquences P."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        P_afc,
        cmap="Greens",
        xticklabels=True, yticklabels=True,
        annot=True, fmt=".003f",
        ax=ax
    )
    ax.set_title("Frequency Matrix")
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Channels")
    fig.tight_layout()
    return fig

def afc_khi2_inertie_totale():
    chi2_from_inertia = n_afc * inertie_totale_afc
    chi2_scipy, p_value, ddl, expected_scipy = chi2_contingency(afc_data)

    seuil_critique = chi2.ppf(0.95, ddl)

    # ✅ Conclusion plus détaillée (sans ajouter de champs)
    if p_value < 0.05:
        conclusion_txt = (
            "SIGNIFICANT DEPENDENCE: p-value < 0.05, therefore we reject the hypothesis of independence "
            "between the channels (rows) and the criteria (columns). The discrepancies between Observed and Expected (under H0) "
            "are too important to be explained solely by chance."
        )
    else:
        conclusion_txt = (
            "NO SIGNIFICANT DEPENDENCE: p-value ≥ 0.05, therefore we do not reject the hypothesis "
            "of independence between the channels and the criteria. The observed vs. expected differences are consistent "
            "with random fluctuations at the 5% level."
        )

    df = pd.DataFrame({
        "Criteria": ["Observed Chi² (AFC)", "Observed Chi² (scipy)", "Critical threshold 5%", "Total Inertia",
                    "p-value", "Degrees of freedom (ddl)", "Conclusion"],
        "Values": [
            f"{chi2_from_inertia:.2f}",
            f"{chi2_scipy:.2f}",
            f"{seuil_critique:.2f}",
            f"{inertie_totale_afc:.4f}",
            f"{p_value:.2e}",
            str(ddl),
            conclusion_txt
        ]
    })
    return df

def afc_distances_khi2_cellules():
    """Heatmap des contributions des cellules au χ²."""
    chi2_cell = (afc_data - expected_afc * n_afc)**2 / (expected_afc * n_afc)
    chi2_cell_df = pd.DataFrame(chi2_cell, index=afc_data.index, columns=afc_data.columns)

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        chi2_cell_df,
        cmap="Reds",
        annot=True, fmt=".3f",
        cbar=True,
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    ax.set_title("Cell Contributions to χ² (AFC)")
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Channels")
    fig.tight_layout()
    return fig

def afc_plan_factoriel():
    """Plan factoriel F1 × F2 lignes & colonnes."""
    dim = 2
    F = U_afc[:, :dim] @ np.diag(s_afc[:dim])
    G = Vt_afc.T[:, :dim] @ np.diag(s_afc[:dim])

    coords_lignes = pd.DataFrame(F, index=afc_data.index, columns=["F1", "F2"])
    coords_colonnes = pd.DataFrame(G, index=afc_data.columns, columns=["F1", "F2"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords_lignes["F1"], coords_lignes["F2"], color="blue", label="Channels")
    for i, name in enumerate(coords_lignes.index):
        ax.text(coords_lignes.iloc[i, 0], coords_lignes.iloc[i, 1], name,
                color="blue", fontsize=8)

    ax.scatter(coords_colonnes["F1"], coords_colonnes["F2"],
               color="red", marker="s", label="Criteria")
    for j, name in enumerate(coords_colonnes.index):
        ax.text(coords_colonnes.iloc[j, 0], coords_colonnes.iloc[j, 1], name,
                color="red", fontsize=8)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_title("Factorial plane AFC (F1 × F2)")
    ax.legend()
    fig.tight_layout()
    return fig

def afc_test_khi2_interpretation():
    chi2_from_inertia = n_afc * inertie_totale_afc
    chi2_scipy, p_value, ddl, expected_scipy = chi2_contingency(afc_data)

    # ✅ Interprétation plus détaillée (sans ajouter de champs)
    if p_value < 0.05:
        interpretation = (
            "The chi-square test indicates a statistically significant association: the distribution of criteria "
            "depends on the channel. In other words, some channels present different criterion profiles than those "
            "expected if everything were independent."
        )
    else:
        interpretation = (
            "The chi-square test did not reveal a significant association: no solid statistical evidence "
            "that the distribution of criteria varies according to the channel, at the 5% threshold."
        )

    df = pd.DataFrame({
        "Index": ["Chi² (AFC)", "Chi² (scipy)", "p-value", "ddl", "Interpretation"],
        "Value": [chi2_from_inertia, chi2_scipy, p_value, ddl, interpretation]
    })
    return df


# ================== CYBER SECURITY ==================

def cyber_import_data():
    df_cyber = pd.read_excel(cyber_data_path, index_col=0)
    df_cyber_affichage = df_cyber.reset_index()             # Canal redevient colonne
    return df_cyber_affichage

def cyber_isolation_and_lof():
    df_cyber_aff = cyber_import_data()
    df_cyber = df_cyber_aff.set_index("Channel")

    data_with_cyber = df_cyber.copy()  # <-- cyber only

    features = ["SCADA_Alerts", "DDoS_Attacks", "Command_latency", "Overall_risk"]

    X = data_with_cyber[features].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.2, random_state=42)
    data_with_cyber["IF_Prediction"] = iso.fit_predict(X_scaled)

    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.2)
    data_with_cyber["LOF_Prediction"] = lof.fit_predict(X_scaled)

    return data_with_cyber

def cyber_heatmap_isolation_only():
    data_with_cyber = cyber_isolation_and_lof()
    pred_if = data_with_cyber[["IF_Prediction"]]

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(
        pred_if.T,
        cmap="RdYlGn_r",
        annot=True, fmt="d",
        cbar_kws={'label': '-1 = Anomalia, 1 = Normal'},
        ax=ax
    )
    ax.set_title("Predictions Isolation Forest per channel")
    ax.set_xlabel("Channels")
    ax.set_ylabel("Isolation Forest")
    fig.tight_layout()
    return fig

def cyber_heatmap_lof_only():
    data_with_cyber = cyber_isolation_and_lof()
    pred_lof = data_with_cyber[["LOF_Prediction"]]

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(
        pred_lof.T,
        cmap="RdYlGn_r",
        annot=True, fmt="d",
        cbar_kws={'label': '-1 = Anomalia, 1 = Normal'},
        ax=ax
    )
    ax.set_title("Predictions LOF per channel")
    ax.set_xlabel("Channels")
    ax.set_ylabel("LOF")
    fig.tight_layout()
    return fig

def cyber_high_risk_summary():
    data_with_cyber = cyber_isolation_and_lof()

    anomalies = data_with_cyber[
        (data_with_cyber["IF_Prediction"] == -1) |
        (data_with_cyber["LOF_Prediction"] == -1)
    ]

    df = anomalies[["Overall_risk", "IF_Prediction", "LOF_Prediction"]].copy()

    # ✅ Ajouter la colonne Canal (qui est l'index)
    df = df.reset_index()          # -> colonne "Canal" [web:104]
    # si jamais la colonne s'appelle "index" chez toi:
    # df = df.reset_index().rename(columns={"index": "Canal"})

    df["Recommended_action"] = "Audit the channel and strengthen security"
    return df



def cyber_if_lof_scatter():
    data_with_cyber = cyber_isolation_and_lof()

    # plus grand
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=120)

    # points + texte plus grands
    point_size = 160
    text_size = 10

    # --- Isolation Forest ---
    colors_if = np.where(data_with_cyber["IF_Prediction"] == -1, "red", "green")
    ax1.scatter(
        data_with_cyber["SCADA_Alerts"],
        data_with_cyber["DDoS_Attacks"],
        c=colors_if, s=point_size, edgecolor="black", linewidth=0.8
    )
    for canal in data_with_cyber.index:
        x = data_with_cyber.loc[canal, "SCADA_Alerts"]
        y = data_with_cyber.loc[canal, "DDoS_Attacks"]
        color = "red" if data_with_cyber.loc[canal, "IF_Prediction"] == -1 else "green"
        ax1.text(x + 0.15, y + 0.15, canal, fontsize=text_size, color=color, fontweight="bold")

    ax1.set_xlabel("SCADA Alerts", fontsize=12)
    ax1.set_ylabel("DDoS Attacks", fontsize=12)
    ax1.set_title("Isolation Forest : Risky channels (RED)", fontsize=13)
    ax1.tick_params(labelsize=10)

    # --- LOF ---
    colors_lof = np.where(data_with_cyber["LOF_Prediction"] == -1, "red", "green")
    ax2.scatter(
        data_with_cyber["SCADA_Alerts"],
        data_with_cyber["DDoS_Attacks"],
        c=colors_lof, s=point_size, edgecolor="black", linewidth=0.8
    )
    for canal in data_with_cyber.index:
        x = data_with_cyber.loc[canal, "SCADA_Alerts"]
        y = data_with_cyber.loc[canal, "DDoS_Attacks"]
        color = "red" if data_with_cyber.loc[canal, "LOF_Prediction"] == -1 else "green"
        ax2.text(x + 0.15, y + 0.15, canal, fontsize=text_size, color=color, fontweight="bold")

    ax2.set_xlabel("SCADA Alerts", fontsize=12)
    ax2.set_ylabel("DDoS Attacks", fontsize=12)
    ax2.set_title("LOF : Risky channels (RED)", fontsize=13)
    ax2.tick_params(labelsize=10)

    fig.tight_layout()
    return fig


def contribution_individus_variables():
    print("=== 5 - Contribution des individus et des variables sur les axes 1 et 2 ===\n")
    
    # =========================
    # 1) Contribution des individus
    # =========================
    coord_carres_12 = individus_pca[['Component_1', 'Component_2']]**2
    somme_coord = coord_carres_12.sum(axis=0)           # somme par axe
    contrib_ind = coord_carres_12.divide(somme_coord, axis=1) * 100

    print("Contribution of individuals (axes 1 et 2) – en % (extrait) :")
    print(contrib_ind.head().round(2))

    # Heatmaps contributions individus : 4 blocs de 25 individus
    parts = [
        contrib_ind.iloc[0:25, :],
        contrib_ind.iloc[25:50, :],
        contrib_ind.iloc[50:75, :],
        contrib_ind.iloc[75:100, :],
    ]
    titles = ["Individuals 1-25", "Individuals 26-50", "Individuals 51-75", "Individuals 76-100"]

    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
    vmin = contrib_ind.values.min()
    vmax = contrib_ind.values.max()

    for part, ax, title in zip(parts, axes.ravel(), titles):
        sns.heatmap(
            part,
            annot=True,
            fmt=".1f",
            cmap="OrRd",
            vmin=vmin, vmax=vmax,
            linewidths=0.5,
            linecolor="gray",
            cbar=False,
            annot_kws={"size": 5},
            ax=ax
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    fig1.suptitle("Contribution of individuals axes 1 and 2 (%)", fontsize=12)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    
    # =========================
    # 2) Contribution des variables
    # =========================
    comp_carres_12 = variables_pca.loc[["Component_1", "Component_2"], numeric_cols]**2
    somme_comp = comp_carres_12.sum(axis=1)             # somme par axe
    contrib_var = comp_carres_12.divide(somme_comp, axis=0) * 100

    print("\nContribution of variables axes 1 and 2 – en % :")
    print(contrib_var.round(2))
    
    # Heatmap contributions variables (une seule figure)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        contrib_var,
        annot=True,
        fmt=".1f",
        cmap="OrRd",
        linewidths=0.8,
        linecolor="gray",
        cbar_kws={"shrink": 0.8, "label": "Contribution (%)"},
        annot_kws={"size": 8},
        ax=ax2
    )
    ax2.set_title("Contribution of variables axes 1 and 2 (%)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    fig2.tight_layout()
    
    return fig1, fig2
