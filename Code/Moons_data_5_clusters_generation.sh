#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_moons
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

# Répertoire de sortie
output_dir = "MakeMoons_5Clusters_Projected_WithNoise"
os.makedirs(output_dir, exist_ok=True)

# Fonction : générer un arc personnalisé sans bruit
def generate_face_arc(n_samples, stretch_factor, angle, shift_x=0, shift_y=0, seed=None):
    X, y = make_moons(n_samples=n_samples * 2, noise=0.0, random_state=seed)  # NOISE = 0
    X = X[y == 0]
    X[:, 0] *= stretch_factor
    np.random.seed(seed)
    selected_idx = np.random.choice(len(X), size=n_samples, replace=False)
    X = X[selected_idx]

    theta = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = X @ rotation_matrix.T

    X[:, 0] += shift_x
    X[:, 1] += shift_y
    return X

# Paramètres
n_samples_per_cluster = 400
angles = [-160, 160, 10, -10, 180]
shifts = [(4, 1.5), (-4, 1.5), (2, 1.2), (-2, 1.2), (0, 1)]
stretch_factors = [1.5, 1.5, 1.5, 1.5, 1]

dimensions = [10, 50, 200]
n_datasets_per_config = 50
total_datasets = 0

for dim in dimensions:
    for run in range(n_datasets_per_config):
        seed_data = 1000 + run
        seed_proj = 5000 + run
        rng = np.random.RandomState(seed_proj)

        # Génération des 5 clusters
        X_list, y_list = [], []
        for i in range(5):
            arc = generate_face_arc(n_samples=n_samples_per_cluster,
                                    stretch_factor=stretch_factors[i],
                                    angle=angles[i],
                                    shift_x=shifts[i][0],
                                    shift_y=shifts[i][1],
                                    seed=seed_data + i * 10)
            X_list.append(arc)
            y_list.append(np.full(arc.shape[0], i))

        X_2d = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Projection aléatoire
        projector = GaussianRandomProjection(n_components=dim, random_state=seed_proj)
        X_projected = projector.fit_transform(X_2d)

        # Normalisation z-score
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_projected)

        # Ajout de bruit
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

        # Création du DataFrame
        columns = [f'Feature_{i+1}' for i in range(dim)]
        df = pd.DataFrame(X_noisy, columns=columns)
        df["Label"] = y

        # Sauvegarde
        filename = f"make_moons_C5_dim{dim}_run{run+1}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        total_datasets += 1

print(f"✅ {total_datasets} datasets (5 clusters, bruit injecté, 3 dimensions, 50 runs chacun) générés dans le dossier '{output_dir}'.")