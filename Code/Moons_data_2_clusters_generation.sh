#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import os

# Paramètres
dimensions = [10, 50, 200]
n_datasets_per_config = 50
n_samples = 2000
output_dir = "MakeMoons_Projected_Datasets_WithNoise"

os.makedirs(output_dir, exist_ok=True)
dataset_counter = 0

for dim in dimensions:
    for i in range(n_datasets_per_config):
        seed_data = 1000 + i
        seed_proj = 5000 + i
        rng = np.random.RandomState(seed_proj)

        # Étape 1 : Générer make_moons SANS bruit
        X, y = make_moons(n_samples=n_samples, noise=0.0, random_state=seed_data)

        # Étape 2 : Projection aléatoire
        projector = GaussianRandomProjection(n_components=dim, random_state=seed_proj)
        X_projected = projector.fit_transform(X)

        # Étape 3 : Normaliser chaque variable (z-score)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_projected)

        # Étape 4 : Ajouter bruit à 75% des colonnes
        num_features = X_normalized.shape[1]
        indices = np.arange(num_features)
        rng.shuffle(indices)

        n_25 = num_features // 4

        # Répartition : 25% untouched, 25% bruit s=1, 25% bruit s=0.5, 25% bruit s=0.25
        untouched = indices[0:n_25]
        noise_1 = indices[n_25:2*n_25]
        noise_05 = indices[2*n_25:3*n_25]
        noise_025 = indices[3*n_25:]

        # Ajouter bruit
        X_noisy = X_normalized.copy()
        X_noisy[:, noise_1] += rng.normal(loc=0.0, scale=1.0, size=(n_samples, len(noise_1)))
        X_noisy[:, noise_05] += rng.normal(loc=0.0, scale=0.5, size=(n_samples, len(noise_05)))
        X_noisy[:, noise_025] += rng.normal(loc=0.0, scale=0.25, size=(n_samples, len(noise_025)))

        # Étape 5 : Créer DataFrame et sauvegarder
        columns = [f"Feature_{j+1}" for j in range(dim)]
        df = pd.DataFrame(X_noisy, columns=columns)
        df["Label"] = y

        filename = f"make_moons_dim{dim}_run{i+1}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        dataset_counter += 1

print(f"✅ {dataset_counter} datasets générés dans '{output_dir}' avec bruit injecté (25% colonnes intactes).")
