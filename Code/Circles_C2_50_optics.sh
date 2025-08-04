#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from collections import defaultdict
from joblib import Parallel, delayed
import traceback

# === Répertoire contenant les fichiers CSV ===
folder_path = '/home/ousmane/Make_circles_C2_50Features'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
print(f"[INFO] Nombre de fichiers CSV trouvés : {len(csv_files)}")
if csv_files:
    print(f"[INFO] Exemple de fichier : {csv_files[0]}")
else:
    print("[ERREUR] Aucun fichier CSV trouvé. Vérifie le chemin.")

# Liste des méthodes de réduction de dimension
reduction_methods = ['PCA', 'KernelPCA', 'VAE_autoencoder', 'Isomap', 'MDS']
n_components_list = [1, 12, 25]

# === Méthodes de réduction ===
def apply_dimensionality_reduction(X, method, n_components):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'KernelPCA':
        reducer = KernelPCA(n_components=n_components, kernel='rbf')
    elif method == 'Isomap':
        reducer = Isomap(n_components=n_components, n_neighbors=5)
    elif method == 'MDS':
        reducer = MDS(n_components=n_components, random_state=10, n_init=50)
    elif method == 'VAE_autoencoder':
        return vae_autoencoder_reduction(X, n_components)
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(X)

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
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        z_mean = Dense(n_components, name='z_mean')(x)
        z_log_var = Dense(n_components, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self, input_dim, n_components):
        latent_inputs = Input(shape=(n_components,))
        x = Dense(32, activation='relu')(latent_inputs)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        outputs = Dense(input_dim, activation='sigmoid')(x)
        return Model(latent_inputs, outputs, name='decoder')

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

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
    #plt.savefig(f'C:/Users/Ousmane/Desktop/UQÀM/Plots/Plot_loss_curve_agg_vae_wisconsin.png')
    plt.show()

    encoder_model = vae.encoder
    X_reduced = encoder_model.predict(X)[0]  # Récupérer z_mean
    return X_reduced


def process_file(file_path):
    filename = os.path.basename(file_path)
    print(f"Traitement de {filename}")
    try:
        data = pd.read_csv(file_path)
        labels = data['Label']
        data = data.drop('Label', axis=1)
        X_scaled = StandardScaler().fit_transform(data)

        scores = {}

        # Clustering sans réduction
        optics = OPTICS(min_samples=5, xi=0.02, min_cluster_size=0.05)
        y_pred_original = optics.fit_predict(X_scaled)
        ari_original = ari(labels, y_pred_original)
        scores['No Reduction'] = ari_original

        # Parallélisation des méthodes sauf VAE_autoencoder
        def process_reduction(method, n_components):
            key = f"{method}{n_components}"
            try:
                X_reduced = apply_dimensionality_reduction(X_scaled, method, n_components)
                optics_reduced = OPTICS(min_samples=5, xi=0.02, min_cluster_size=0.05)
                y_pred = optics_reduced.fit_predict(X_reduced)
                score = ari(labels, y_pred)
                return (key, score)
            except Exception as e:
                print(f"[ERREUR] {method}({n_components}) - {filename} : {e}")
                traceback.print_exc()
                return (key, np.nan)

        # Séparer méthodes avec et sans VAE
        non_vae_methods = [m for m in reduction_methods if m != 'VAE_autoencoder']
        vae_methods = ['VAE_autoencoder']

        # === Parallèle pour méthodes non-VAE
        parallel_results = Parallel(n_jobs=-1)(
            delayed(process_reduction)(method, n)
            for method in non_vae_methods
            for n in n_components_list
        )

        # === Séquentiel pour VAE_autoencoder
        vae_results = [
            process_reduction(method, n)
            for method in vae_methods
            for n in n_components_list
        ]

        # Fusionner les résultats
        all_results = parallel_results + vae_results
        for key, score in all_results:
            scores[key] = score

        return filename, scores, ari_original

    except Exception as e:
        print(f"[ERREUR] Fichier {filename} : {e}")
        traceback.print_exc()
        return filename, {}, np.nan



def main():
    results = []
    for file_path in csv_files:
        results.append(process_file(file_path))

    ari_scores_per_file = {}
    ari_original_list = []
    ari_scores_dict = defaultdict(list)

    for filename, scores, ari_original in results:
        ari_scores_per_file[filename] = scores
        if not np.isnan(ari_original):
            ari_original_list.append(ari_original)
        for key, val in scores.items():
            if key != 'No Reduction' and not np.isnan(val):
                method = ''.join(filter(str.isalpha, key))
                n = int(''.join(filter(str.isdigit, key)))
                ari_scores_dict[(method, n)].append(val)

    ari_original_mean = np.mean(ari_original_list)
    ari_scores_avg = {k: np.mean(v) for k, v in ari_scores_dict.items()}

    print("\nScores ARI moyens :")
    print(f"- Avant réduction de dimension (No Reduction) : {round(ari_original_mean, 4)}")

    for method in reduction_methods:
        print(f"\nMéthode : {method}")
        for n_components in n_components_list:
            key = (method, n_components)
            if key in ari_scores_avg:
                print(f"  - n_components = {n_components} ➔ ARI moyen = {round(ari_scores_avg[key], 4)}")
            else:
                print(f"  - n_components = {n_components} ➔ (aucun score disponible)")

    df_ari_per_file = pd.DataFrame.from_dict(ari_scores_per_file, orient='index')
    columns_order = ['No Reduction'] + [f"{method}{n}" for method in reduction_methods for n in n_components_list]
    existing_columns = df_ari_per_file.columns.tolist()
    ordered_existing_columns = [col for col in columns_order if col in existing_columns]

    missing_cols = [col for col in columns_order if col not in existing_columns]
    if missing_cols:
        print(f"[INFO] Colonnes manquantes dans df_ari_per_file : {missing_cols}")

    df_ari_per_file = df_ari_per_file[ordered_existing_columns]

    pd.set_option("display.max_columns", None)
    print("\n📊 DataFrame des scores ARI par fichier :")
    print(df_ari_per_file)
    df_ari_per_file.to_csv("ari_scores_circles_C2D50_optics.csv")

    # Plotly
    bar_width = 0.2
    x = np.arange(len(n_components_list)) * 1.8
    x_no_reduction = -1.2

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[x_no_reduction],
        y=[ari_original_mean],
        text=[f'{round(ari_original_mean, 2)}'],
        textposition='outside',
        marker_color='orange',
        name='No Reduction',
        width=bar_width
    ))

    colors = {'PCA': 'skyblue', 'KernelPCA': 'lightgreen', 'VAE_autoencoder': 'salmon', 'Isomap': 'purple', 'MDS': 'red'}
    for i, method in enumerate(reduction_methods):
        x_method = x + (i * bar_width)
        y_vals = [ari_scores_avg.get((method, n), 0) for n in n_components_list]
        fig.add_trace(go.Bar(
            x=x_method,
            y=y_vals,
            text=[f'{round(val, 2)}' for val in y_vals],
            textposition='outside',
            marker_color=colors[method],
            name=method,
            width=bar_width
        ))

    fig.add_shape(
        type="line",
        x0=x_no_reduction + bar_width + 0.1,
        y0=0,
        x1=x_no_reduction + bar_width + 0.1,
        y1=1,
        line=dict(color="black", width=2, dash="dash")
    )

    xticks = [x_no_reduction] + list(x + 2.0 * bar_width for x in x)
    xtick_labels = ['No Reduction'] + [str(n) for n in n_components_list]
    fig.update_layout(
        xaxis=dict(
            tickvals=xticks,
            ticktext=xtick_labels,
            range=[-1.5, max(xticks) + 1.0],
            tickfont=dict(size=40),
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror='allticks',
            showgrid=False,
            ticks='outside',
            tickmode='array'
        ),
        yaxis=dict(
            range=[0, 1],
            tickvals=[i / 10 for i in range(11)],
            tickfont=dict(size=40),
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror='allticks',
            showgrid=False,
            ticks='outside',
            tickmode='array'
        ),
        barmode='group',
        bargap=0.5,
        bargroupgap=0.7,
        title='ARI Scores for Agglomerative Clustering with and without Dimensionality Reduction',
        xaxis_title='number of features',
        yaxis_title='ARI',
        legend_title='Methods',
        height=2000,
        width=3000,
        plot_bgcolor='white',
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=40)
        ),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        font=dict(size=40),
        uniformtext_minsize=40,
        uniformtext_mode='show'
    )

    fig.show()
    output_path = '/home/ousmane/results/Plot_clustering_optics_vae_circles_C2_50D.png'
    fig.write_image(output_path)
    print(f"Figure saved as {output_path}")

if __name__ == '__main__':
    main()

