#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_circles
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

# Répertoire de sortie
output_dir = "MakeCircles_5Clusters_Projected_WithNoise"
os.makedirs(output_dir, exist_ok=True)

# Paramètres
dimensions = [10, 50, 200]
n_datasets_per_config = 50
n_samples_per_circle = 400  # 400 × 5 cercles = 2000 points
scaling_factors = [1, 2, 3.5, 5, 7]

total_datasets = 0

for dim in dimensions:
    for run in range(n_datasets_per_config):
        seed_base = 1000 + run
        seed_proj = 5000 + run
        rng = np.random.RandomState(seed_proj)

        X_list, y_list = [], []

        for i, scale in enumerate(scaling_factors):
            seed_circle = seed_base + i * 50
            X, _ = make_circles(n_samples=n_samples_per_circle, noise=0, factor=0.8, random_state=seed_circle)  # noise supprimé
            X *= scale  # Mise à l’échelle du cercle
            X_list.append(X)
            y_list.append(np.full(X.shape[0], i))

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)

        # Projection aléatoire
        projector = GaussianRandomProjection(n_components=dim, random_state=seed_proj)
        X_projected = projector.fit_transform(X_all)

        # Normalisation z-score
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_projected)

        # Injection de bruit
        num_features = X_normalized.shape[1]
        indices = np.arange(num_features)
        rng.shuffle(indices)

        n_25 = num_features // 4
        untouched = indices[0:n_25]
        noise_1 = indices[n_25:2*n_25]
        noise_05 = indices[2*n_25:3*n_25]
        noise_025 = indices[3*n_25:]

        X_noisy = X_normalized.copy()
        X_noisy[:, noise_1] += rng.normal(loc=0.0, scale=0.5/4, size=(X_noisy.shape[0], len(noise_1)))
        X_noisy[:, noise_05] += rng.normal(loc=0.0, scale=0.25/4, size=(X_noisy.shape[0], len(noise_05)))
        X_noisy[:, noise_025] += rng.normal(loc=0.0, scale=0.125/4, size=(X_noisy.shape[0], len(noise_025)))

        # DataFrame
        columns = [f'Feature_{i+1}' for i in range(dim)]
        df = pd.DataFrame(X_noisy, columns=columns)
        df["Label"] = y_all

        # Sauvegarde
        filename = f"make_circles_C5_dim{dim}_run{run+1}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)

        total_datasets += 1

print(f"✅ {total_datasets} datasets générés (5 cercles, bruit injecté, 3 dimensions, 50 runs chacun) dans le dossier '{output_dir}'.")