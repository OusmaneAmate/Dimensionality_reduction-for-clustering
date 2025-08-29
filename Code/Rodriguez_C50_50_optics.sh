#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import re
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

# === Chemin vers les fichiers
folder_path = "/home/ousmane/C50_50Features"
output_csv = "ari_scores_rodriguez_C50D50_optics.csv"
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# === Réductions à tester
reduction_methods = ["PCA", "KernelPCA", "Isomap", "Autoencoder", "MDS"]
n_components_list = [9, 12, 25]

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

# Réduction de dimension avec VAE
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, input_dim, n_components):
        super(VAE, self).__init__()
        self.encoder = self.build_encoder(input_dim, n_components)
        self.decoder = self.build_decoder(input_dim, n_components)

    def build_encoder(self, input_dim, n_components):
        inputs = tf.keras.Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        #x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        z_mean = Dense(n_components, name='z_mean')(x)
        z_log_var = Dense(n_components, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self, input_dim, n_components):
        latent_inputs = tf.keras.Input(shape=(n_components,))
        x = Dense(64, activation='relu')(latent_inputs)
        x = Dense(128, activation='relu')(x)
        #x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(input_dim, activation='sigmoid')(x)
        return tf.keras.Model(latent_inputs, outputs, name='decoder')

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

def vae_autoencoder_reduction(X, n_components):
    input_dim = X.shape[1]
    vae = VAE(input_dim, n_components)
    vae.compile(optimizer='adam', loss='mse')

    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

    # Entraîner l'autoencodeur et enregistrer l'historique des pertes
    history = vae.fit(X_train, X_train, epochs=100, batch_size=64, shuffle=False, validation_data=(X_test, X_test), verbose=1, callbacks=[tf.keras.callbacks.TerminateOnNaN()])

    # Tracer la courbe d'erreur (loss curve)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Courbe d\'erreur de l\'autoencodeur (VAE)')
    plt.legend()
    plt.show()

    encoder_model = vae.encoder
    X_reduced = encoder_model.predict(X)[0]  # Récupérer z_mean
    return X_reduced



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
            return vae_autoencoder_reduction(X, n_components)
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
    X, y = load_txt(path)
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

    df_results[file] = row

# === Construction DataFrame
df_ari = pd.DataFrame.from_dict(df_results, orient="index")
df_ari.index.name = "File"
df_ari.to_csv(output_csv)
print(f"[✅] Fichier sauvegardé : {output_csv}")