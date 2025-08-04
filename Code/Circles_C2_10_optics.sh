#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.callbacks import TerminateOnNaN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from collections import defaultdict


# === Répertoire contenant les fichiers CSV ===
folder_path = '/home/ousmane/Make_circles_C2_10Features'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# === Stocker les ARI ===
ari_original_list = []
ari_scores_dict = defaultdict(list)  # clé = (method, n_components)

# === Initialisation du DataFrame pour stocker les résultats par fichier
ari_scores_per_file = {}

# Liste des méthodes de réduction de dimension
reduction_methods = ['PCA', 'KernelPCA', 'VAE_autoencoder', 'Isomap', 'MDS']
n_components_list = [1, 3, 5]
ari_scores = []

# Méthodes de réduction de dimensionnalité
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
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        #x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        z_mean = Dense(n_components, name='z_mean')(x)
        z_log_var = Dense(n_components, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self, input_dim, n_components):
        latent_inputs = tf.keras.Input(shape=(n_components,))
        x = Dense(32, activation='relu')(latent_inputs)
        x = Dense(64, activation='relu')(x)
        #x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
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
    #plt.savefig(f'C:/Users/Ousmane/Desktop/UQÀM/Plots/Plot_loss_curve_agg_vae_wisconsin.png')
    plt.show()

    encoder_model = vae.encoder
    X_reduced = encoder_model.predict(X)[0]  # Récupérer z_mean
    return X_reduced




for file_path in csv_files:
    filename = os.path.basename(file_path)
    print(f"🔍 Traitement de {filename}")

    # Charger et préparer les données
    data = pd.read_csv(file_path)
    labels = data['Label']
    data = data.drop('Label', axis=1)

    # Normaliser
    X_scaled = StandardScaler().fit_transform(data)

    # Dictionnaire temporaire pour stocker les ARIs pour ce fichier
    scores = {}

    # Clustering sans réduction
    optics = OPTICS(min_samples=5, xi=0.02, min_cluster_size=0.05)
    y_pred_original = optics.fit_predict(X_scaled)
    ari_original = ari(labels, y_pred_original)
    scores['No Reduction'] = ari_original
    ari_original_list.append(ari_original)

    # Réduction + clustering
    for method in reduction_methods:
        for n_components in n_components_list:
            key = f"{method}{n_components}"
            try:
                X_reduced = apply_dimensionality_reduction(X_scaled, method, n_components)
                optics_reduced = OPTICS(min_samples=5, xi=0.02, min_cluster_size=0.05)
                y_pred_reduced = optics_reduced.fit_predict(X_reduced)
                ari_reduced = ari(labels, y_pred_reduced)
                scores[key] = ari_reduced
                ari_scores_dict[(method, n_components)].append(ari_reduced)
            except Exception as e:
                print(f"⚠️ Erreur avec {method} ({n_components}) pour {filename} : {e}")
                scores[key] = np.nan  # NaN si erreur

    # Stocker les résultats de ce fichier
    ari_scores_per_file[filename] = scores




# === Moyennes des scores ARI ===
ari_original_mean = np.mean(ari_original_list)
ari_scores_avg = {k: np.mean(v) for k, v in ari_scores_dict.items()}

print("\n🎯 Scores ARI moyens :")
print(f"- Avant réduction de dimension (No Reduction) : {round(ari_original_mean, 4)}")

for method in reduction_methods:
    print(f"\nMéthode : {method}")
    for n_components in n_components_list:
        key = (method, n_components)
        if key in ari_scores_avg:
            print(f"  - n_components = {n_components} ➝ ARI moyen = {round(ari_scores_avg[key], 4)}")
        else:
            print(f"  - n_components = {n_components} ➝ (aucun score disponible)")



#Créer le DataFrame à partir du dictionnaire
df_ari_per_file = pd.DataFrame.from_dict(ari_scores_per_file, orient='index')

# Réorganiser les colonnes
columns_order = ['No Reduction']
for method in reduction_methods:
    for n in n_components_list:
        columns_order.append(f"{method}{n}")

df_ari_per_file = df_ari_per_file[columns_order]

# Afficher le DataFrame
import pandas as pd
pd.set_option("display.max_columns", None)
print("\n📊 DataFrame des scores ARI par fichier :")
print(df_ari_per_file)
df_ari_per_file.to_csv("ari_scores_circles_C2D10_optics.csv")



# Tracer les résultats avec Plotly
bar_width = 0.2
x = np.arange(len(n_components_list)) * 1.8
x_no_reduction = -1.2

fig = go.Figure()

# Ajout du score sans réduction
fig.add_trace(go.Bar(
    x=[x_no_reduction],
    y=[ari_original_mean],
    text=[f'{round(ari_original_mean, 2)}'],
    textposition='outside',
    marker_color='orange',
    name='No Reduction',
    width=bar_width
))

# Ajout des scores avec réduction
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

# Ajouter une ligne verticale entre "No Reduction" et les autres barres
fig.add_shape(
    type="line",
    x0=x_no_reduction + bar_width + 0.1, 
    y0=0,
    x1=x_no_reduction + bar_width + 0.1,
    y1=1,
    line=dict(
        color="black",
        width=2,
        dash="dash"
    ),
)

# Ajuster les axes
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
    title='ARI Scores for Optics Clustering with and without Dimensionality Reduction',
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
            font=dict(
                size=40  
            )
        ),
        xaxis_showgrid=False,  
        yaxis_showgrid=False,  
        font=dict(size=40),  
        uniformtext_minsize=40,  
        uniformtext_mode='show'  
)

# Afficher le graphique
fig.show()

# Enregistrer l'image
output_path = '/home/ousmane/results/Plot_clustering_optics_vae_circles_C2_10D.png'
fig.write_image(output_path)
print(f"Figure saved as {output_path}")