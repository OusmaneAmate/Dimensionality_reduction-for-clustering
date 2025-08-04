#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import re
import logging

logging.basicConfig(level=logging.INFO)

# === Chemin vers les fichiers
folder_path = "/home/ousmane/C50_200Features"
output_csv = "ari_scores_rodriguez_C50D200_optics.csv"
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt') or f.endswith('.arff')]

# === Réductions à tester
reduction_methods = ["PCA", "KernelPCA", "Isomap", "Autoencoder", "MDS"]
n_components_list = [9, 50, 100]

def load_arff(file_path):
    try:
        data = []
        data_started = False
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().lower() == '@data':
                    data_started = True
                    continue
                if data_started:
                    cleaned_line = re.sub(r'\s+', ' ', line.strip())
                    if cleaned_line:  # éviter les lignes vides
                        data.append(cleaned_line.split(','))
        df = pd.DataFrame(data)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        logging.error(f"[ERREUR] Chargement ARFF {file_path}: {e}")
        return None, None


# === Chargement TXT
def load_txt(file_path):
    try:
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() == "" or line.strip().startswith("%"):
                    continue
                parts = re.sub(r'\s+', ' ', line.strip()).split(' ')
                data.append(parts)
        df = pd.DataFrame(data)
        X = df.iloc[:, :-1].astype(float).values
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        logging.error(f"[ERREUR] Chargement fichier {file_path}: {e}")
        return None, None

# === Normalisation
def normalize_data(X):
    return StandardScaler().fit_transform(X)

# === Autoencoder
def autoencoder_reduction(X, n_components):
    try:
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        input_dim = X_train.shape[1]
        input_layer = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        latent = Dense(n_components, activation='sigmoid')(x)
        x = Dense(64, activation='relu')(latent)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(input_dim, activation='sigmoid')(x)
        autoencoder = Model(input_layer, output)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_train, X_train, epochs=100, batch_size=64,
                        shuffle=False, validation_data=(X_test, X_test), verbose=1)
        encoder = Model(input_layer, latent)
        return encoder.predict(X)
    except Exception as e:
        logging.error(f"[ERREUR] Réduction Autoencoder: {e}")
        return None

# === Application de la réduction
def apply_reduction(X, method, n_components):
    X = normalize_data(X)
    try:
        if method == "PCA":
            return PCA(n_components=n_components).fit_transform(X)
        elif method == "KernelPCA":
            return KernelPCA(n_components=n_components, kernel='rbf').fit_transform(X)
        elif method == "Isomap":
            return Isomap(n_components=n_components, n_neighbors=10).fit_transform(X)
        elif method == "MDS":
            return MDS(n_components=n_components).fit_transform(X)
        elif method == "Autoencoder":
            return autoencoder_reduction(X, n_components)
        else:
            return None
    except Exception as e:
        logging.error(f"[ERREUR] Réduction {method} n_components={n_components}: {e}")
        return None

# === Clustering
def cluster_ari(X, y, k):
    try:
        model = OPTICS(min_samples = 5, min_cluster_size = 0.1, xi = 0.01)
        y_pred = model.fit_predict(X)
        return ari(y, y_pred)
    except Exception as e:
        logging.error(f"[ERREUR] Clustering ARI: {e}")
        return None

# === Extraction du nombre de clusters à partir du nom de fichier
def extract_k_from_filename(filename):
    match = re.search(r'C(\d+)', filename)
    if match:
        k = int(match.group(1))
        return 10 if k == 50 else k
    return None

# === Lancement
df_results = {}

for file in file_list:
    path = os.path.join(folder_path, file)

    if file.endswith(".txt"):
        X, y = load_txt(path)
    elif file.endswith(".arff"):
        X, y = load_arff(path)
    else:
        logging.warning(f"[IGNORÉ] Format non pris en charge : {file}")
        continue

    if X is None or y is None:
        continue

    k = extract_k_from_filename(file)
    if k is None:
        continue

    row = {}
    score = cluster_ari(X, y, k)
    row["No Reduction"] = score

    for method in reduction_methods:
        for n in n_components_list:
            reduced = apply_reduction(X, method, n)
            if reduced is not None:
                score = cluster_ari(reduced, y, k)
                row[f"{method}{n}"] = score
            else:
                row[f"{method}{n}"] = np.nan  # marquer les échecs de réduction

    df_results[file] = row

# === Construction DataFrame
df_ari = pd.DataFrame.from_dict(df_results, orient="index")
df_ari.index.name = "File"
df_ari.to_csv(output_csv)
print(f"[✅] Fichier sauvegardé : {output_csv}")

#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import re
import logging

logging.basicConfig(level=logging.INFO)

# === Chemin vers les fichiers
folder_path = "/home/ousmane/C10_200Features"
output_csv = "ari_scores_rodriguez_C10D200_optics.csv"
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt') or f.endswith('.arff')]

# === Réductions à tester
reduction_methods = ["PCA", "KernelPCA", "Isomap", "Autoencoder", "MDS"]
n_components_list = [9, 50, 100]

def load_arff(file_path):
    try:
        data = []
        data_started = False
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().lower() == '@data':
                    data_started = True
                    continue
                if data_started:
                    cleaned_line = re.sub(r'\s+', ' ', line.strip())
                    if cleaned_line:  # éviter les lignes vides
                        data.append(cleaned_line.split(','))
        df = pd.DataFrame(data)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        logging.error(f"[ERREUR] Chargement ARFF {file_path}: {e}")
        return None, None


# === Chargement TXT
def load_txt(file_path):
    try:
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() == "" or line.strip().startswith("%"):
                    continue
                parts = re.sub(r'\s+', ' ', line.strip()).split(' ')
                data.append(parts)
        df = pd.DataFrame(data)
        X = df.iloc[:, :-1].astype(float).values
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        logging.error(f"[ERREUR] Chargement fichier {file_path}: {e}")
        return None, None

# === Normalisation
def normalize_data(X):
    return StandardScaler().fit_transform(X)

# === Autoencoder
def autoencoder_reduction(X, n_components):
    try:
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        input_dim = X_train.shape[1]
        input_layer = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        latent = Dense(n_components, activation='sigmoid')(x)
        x = Dense(64, activation='relu')(latent)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(input_dim, activation='sigmoid')(x)
        autoencoder = Model(input_layer, output)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_train, X_train, epochs=100, batch_size=64,
                        shuffle=False, validation_data=(X_test, X_test), verbose=1)
        encoder = Model(input_layer, latent)
        return encoder.predict(X)
    except Exception as e:
        logging.error(f"[ERREUR] Réduction Autoencoder: {e}")
        return None

# === Application de la réduction
def apply_reduction(X, method, n_components):
    X = normalize_data(X)
    try:
        if method == "PCA":
            return PCA(n_components=n_components).fit_transform(X)
        elif method == "KernelPCA":
            return KernelPCA(n_components=n_components, kernel='rbf').fit_transform(X)
        elif method == "Isomap":
            return Isomap(n_components=n_components, n_neighbors=10).fit_transform(X)
        elif method == "MDS":
            return MDS(n_components=n_components).fit_transform(X)
        elif method == "Autoencoder":
            return autoencoder_reduction(X, n_components)
        else:
            return None
    except Exception as e:
        logging.error(f"[ERREUR] Réduction {method} n_components={n_components}: {e}")
        return None

# === Clustering
def cluster_ari(X, y, k):
    try:
        model = OPTICS(min_samples = 5, min_cluster_size = 0.1, xi = 0.01)
        y_pred = model.fit_predict(X)
        return ari(y, y_pred)
    except Exception as e:
        logging.error(f"[ERREUR] Clustering ARI: {e}")
        return None

# === Extraction du nombre de clusters à partir du nom de fichier
def extract_k_from_filename(filename):
    match = re.search(r'C(\d+)', filename)
    if match:
        k = int(match.group(1))
        return 10 if k == 50 else k
    return None

# === Lancement
df_results = {}

for file in file_list:
    path = os.path.join(folder_path, file)

    if file.endswith(".txt"):
        X, y = load_txt(path)
    elif file.endswith(".arff"):
        X, y = load_arff(path)
    else:
        logging.warning(f"[IGNORÉ] Format non pris en charge : {file}")
        continue

    if X is None or y is None:
        continue

    k = extract_k_from_filename(file)
    if k is None:
        continue

    row = {}
    score = cluster_ari(X, y, k)
    row["No Reduction"] = score

    for method in reduction_methods:
        for n in n_components_list:
            reduced = apply_reduction(X, method, n)
            if reduced is not None:
                score = cluster_ari(reduced, y, k)
                row[f"{method}{n}"] = score
            else:
                row[f"{method}{n}"] = np.nan  # marquer les échecs de réduction

    df_results[file] = row

# === Construction DataFrame
df_ari = pd.DataFrame.from_dict(df_results, orient="index")
df_ari.index.name = "File"
df_ari.to_csv(output_csv)
print(f"[✅] Fichier sauvegardé : {output_csv}")

