# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 18:18:28 2025

@author: achra
"""

import numpy as np
import pandas as pd

# Paramètres
np.random.seed(42)  # Reproductibilité
n_canaux = 100

# Génération des données respectant les bornes
data = {
    'Canal': [f'Canal_{i+1}' for i in range(n_canaux)],
    'Débit (m³/s)': np.random.uniform(0, 200, n_canaux),
    'Vitesse eau (m/s)': np.random.uniform(0, 5, n_canaux),
    'Largeur (m)': np.random.uniform(1, 50, n_canaux),
    'Profondeur (m)': np.random.uniform(0.5, 10, n_canaux),
    'Rugosité': np.random.uniform(0, 0.05, n_canaux),
    'Pente (%)': np.random.uniform(0, 5, n_canaux),
    'Température eau (°C)': np.random.uniform(5, 30, n_canaux),
    'Envasement (%)': np.random.uniform(0, 100, n_canaux)
}

# Création du DataFrame
df = pd.DataFrame(data)
df.set_index('Canal', inplace=True)

# Vérification des bornes respectées
print(" Aperçu des 10 premiers canaux :")
print(df.head(10).round(2))
print(f"\nDimensions : {df.shape}")
print("\nBornes MIN observées :")
print(df.min().round(2))
print("\nBornes MAX observées :")
print(df.max().round(2))

# Export Excel 
df.to_excel(r'C:\Users\achra\Desktop\4IIRG7\Analyse de données (Data Analysis)\Final PRoject DATA ANALYST Par Professeur EL MKHALET MOUNA\projet Ouvrages hydrauliques\Ouvrages_hydrauliques.xlsx', sheet_name='Canaux', index=True)

