#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_circles
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

# Répertoire de sauvegarde
output_dir = "MakeCircles_Projected_WithNoise"
os.makedirs(output_dir, exist_ok=True)

# Paramètres
dimensions = [10, 50, 200]
n_datasets_per_config = 50
n_samples = 2000  # 1000 par cluster (cercle)
total_datasets = 0

for dim in dimensions:
    for run in range(n_datasets_per_config):
        seed_data = 1000 + run
        seed_proj = 5000 + run
        rng = np.random.RandomState(seed_proj)

        # Étape 1 : Générer les données 2D make_circles SANS bruit
        X, y = make_circles(n_samples=n_samples, noise=0.0, factor=0.2, random_state=seed_data)

        # Étape 2 : Projection aléatoire vers un espace de dim dimensions
        projector = GaussianRandomProjection(n_components=dim, random_state=seed_proj)
        X_projected = projector.fit_transform(X)

        # Étape 3 : Normalisation z-score
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_projected)

        # Étape 4 : Ajout de bruit à 75% des colonnes
        num_features = X_normalized.shape[1]
        indices = np.arange(num_features)
        rng.shuffle(indices)

        n_25 = num_features // 4
        untouched = indices[0:n_25]
        noise_1 = indices[n_25:2*n_25]
        noise_05 = indices[2*n_25:3*n_25]
        noise_025 = indices[3*n_25:]

        X_noisy = X_normalized.copy()
        X_noisy[:, noise_1] += rng.normal(loc=0.0, scale=1.0, size=(X_noisy.shape[0], len(noise_1)))
        X_noisy[:, noise_05] += rng.normal(loc=0.0, scale=0.5, size=(X_noisy.shape[0], len(noise_05)))
        X_noisy[:, noise_025] += rng.normal(loc=0.0, scale=0.25, size=(X_noisy.shape[0], len(noise_025)))

        # Étape 5 : Créer le DataFrame
        columns = [f"Feature_{i+1}" for i in range(dim)]
        df = pd.DataFrame(X_noisy, columns=columns)
        df["Label"] = y

        # Étape 6 : Sauvegarde
        filename = f"make_circles_dim{dim}_run{run+1}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        total_datasets += 1

print(f"✅ {total_datasets} datasets générés avec 2 clusters (make_circles, bruit injecté) dans le dossier '{output_dir}'.")
