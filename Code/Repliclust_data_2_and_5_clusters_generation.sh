#!/usr/bin/env python
# coding: utf-8


import repliclust
import numpy as np
import pandas as pd
import os

def random_config(seed):
    rng = np.random.default_rng(seed)

    min_ov = rng.uniform(0.0005, 0.005)
    max_ov = rng.uniform(min_ov + 0.001, 0.02)  # max > min

    return {
        'min_overlap': min_ov,
        'max_overlap': max_ov,
        'aspect_ref': rng.uniform(1.01, 2.0),         # > 1
        'aspect_maxmin': rng.uniform(1.0, 3.0),       # ≥ 1
        'radius_maxmin': rng.uniform(1.0, 3.0),       # ≥ 1
        'imbalance_ratio': rng.uniform(1.0, 2.0),     # ✅ ≥ 1
        'distributions': ['normal']
    }


# Paramètres fixes
dimensions = [10, 50, 200]
clusters = [2, 5]
n_repeats = 50

root_dir = "repliclust_datasets"
os.makedirs(root_dir, exist_ok=True)

for dim in dimensions:
    for n_clusters in clusters:
        if n_clusters == 2:
            n_samples_per_cluster = 1000
        elif n_clusters == 5:
            n_samples_per_cluster = 400
        
        total_samples = n_clusters * n_samples_per_cluster
        key = f"D{dim}_K{n_clusters}"
        output_dir = os.path.join(root_dir, key)
        os.makedirs(output_dir, exist_ok=True)

        for seed in range(n_repeats):
            config = random_config(seed)
            repliclust.set_seed(seed)

            archetype = repliclust.Archetype(
                n_clusters=n_clusters,
                dim=dim,
                n_samples=total_samples,
                **config
            )

            generator = repliclust.DataGenerator(archetype=archetype)
            data, labels, _ = generator.synthesize(quiet=True)

            df = pd.DataFrame(data, columns=[f"X{i+1}" for i in range(dim)])
            df["label"] = labels

            filename = f"{key}_run{seed}.csv"
            filepath = os.path.join(output_dir, filename)

            df.to_csv(filepath, index=False)
            print(f"💾 Saved: {filepath}")
