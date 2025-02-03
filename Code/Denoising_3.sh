#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import re
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.losses import CategoricalCrossentropy
import logging
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler
import umap
import plotly.graph_objects as go

os.environ['OMP_NUM_THREADS'] = '2'
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the directory containing the files
extraction_dir = '/home/ousmane/C2_50Features/'
subdir_files = [f for f in os.listdir(extraction_dir) if f.endswith('.arff') or f.endswith('.txt')]

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
                    data.append(cleaned_line.split(','))
        df = pd.DataFrame(data)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        logging.error(f"Error loading ARFF file {file_path}: {e}")
        return None, None

def load_txt(file_path):
    try:
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() == "" or line.strip().startswith("%"):
                    continue
                cleaned_line = re.sub(r'\s+', ' ', line.strip())
                data.append(cleaned_line.split(' '))
        df = pd.DataFrame(data)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        logging.error(f"Error loading TXT file {file_path}: {e}")
        return None, None

def process_file(file_path):
    try:
        logging.debug(f"Processing file: {file_path}")
        if file_path.endswith('.arff'):
            X, y = load_arff(file_path)
        else:
            X, y = load_txt(file_path)
        logging.debug(f"Finished processing file: {file_path}, X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None, None

def normalize_data(X):
    scaler = StandardScaler()  # Initialiser StandardScaler
    X_scaled = scaler.fit_transform(X)  # Normaliser X
    return X_scaled

def add_noise(X, noise_factor=0.2):
    """
    Ajoute du bruit gaussien aux données.
    """
    X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    X_noisy = np.clip(X_noisy, 0., 1.)  # S'assure que les valeurs restent entre 0 et 1
    return X_noisy

def denoising_autoencoder_reduction(X, n_components, noise_factor=0.2):
    try:
        # Ajouter du bruit aux données
        X_noisy = add_noise(X, noise_factor)
        
        # Séparer les données d'entraînement et de validation
        X_train, X_val, X_train_noisy, X_val_noisy = train_test_split(X, X_noisy, test_size=0.3, random_state=42)

        input_dim = X_train.shape[1]
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoder = Dense(128, activation='relu')(input_layer)
        encoder = Dense(64, activation='relu')(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.2)(encoder)
       
        latent_layer = Dense(n_components, activation='sigmoid')(encoder)

        # Decoder
        decoder = Dense(64, activation='relu')(latent_layer)
        decoder = Dense(128, activation='relu')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.2)(decoder)


        output_layer = Dense(input_dim, activation='sigmoid')(decoder)

        # Modèle Denoising Autoencoder
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Entraîner le Denoising Autoencoder
        history = autoencoder.fit(X_train_noisy, X_train,
                        epochs=200,
                        batch_size=64,
                        shuffle=False,
                        validation_data=(X_val_noisy, X_val),
                        verbose=0)

        # Affichage de la courbe de perte
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig(f'/home/ousmane/results/Plot_loss_curve_denoising_optics.png')
        plt.show()

        # Extraire la représentation réduite
        encoder_model = Model(inputs=input_layer, outputs=latent_layer)
        X_reduced = encoder_model.predict(X)
        return X_reduced
    except Exception as e:
        logging.error(f"Error in denoising autoencoder reduction: {e}")
        return None

def apply_dimensionality_reduction(X, y, method, n_components):
    try:
        # Normaliser X avant la réduction
        X = normalize_data(X)

        if method == 'PCA':
            reducer = PCA(n_components=n_components)
        elif method == 'KernelPCA':
            reducer = KernelPCA(n_components=n_components, kernel='rbf')
        elif method == 'Denoising_autoencoder':
            return denoising_autoencoder_reduction(X, n_components)
        elif method == 'Isomap':
            reducer = Isomap(n_components=n_components, n_neighbors=10)
        elif method == 'MDS':
            reducer = MDS(n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return reducer.fit_transform(X)
    except Exception as e:
        logging.error(f"Error applying dimensionality reduction with {method}: {e}")
        return None


def cluster_and_evaluate(X, y, n_clusters):
    try:
        model = OPTICS(min_samples = 6, min_cluster_size = 0.1, xi = 0.05)
        y_pred = model.fit_predict(X)
        score = ari(y, y_pred)
        return score
    except Exception as e:
        logging.error(f"Error in clustering and evaluating: {e}")
        return None

# List of dimensionality reduction methods
reduction_methods = ['PCA', 'KernelPCA', 'Denoising_autoencoder', 'Isomap', 'MDS']

# List of n_components to try
n_components_list = [1, 12, 25]

def extract_k_from_filename(file_name):
    # Utilise une expression régulière pour capturer le nombre de classes après le "C"
    match = re.search(r'C(\d+)', file_name)
    if match:
        k = int(match.group(1))
        # Appliquer l'exception si k est 50
        if k == 50:
            k = 10
        return k
    else:
        logging.error(f"Could not determine the number of clusters from filename: {file_name}")
        return None
        
def process_file_for_clustering(file_name):
    file_path = os.path.join(extraction_dir, file_name)
    X, y = process_file(file_path)
    
    if X is not None and y is not None:
        # Extraire la valeur de k à partir du nom du fichier
        k = extract_k_from_filename(file_name)
        if k is None:
            return None
        
        # Calcul de l'ARI sans réduction
        ari_no_reduction = cluster_and_evaluate(X, y, k)
        #logging.debug(f"File: {file_name}, Method: No Reduction, ARI: {ari_no_reduction}")
        results = {'No Reduction': ari_no_reduction}
        
        # Calcul de l'ARI avec les différentes méthodes de réduction
        for method in reduction_methods:
            for n_components in n_components_list:
                X_reduced = apply_dimensionality_reduction(X, y, method, n_components)
                if X_reduced is not None:
                    ari_score = cluster_and_evaluate(X_reduced, y, k)
                    #logging.debug(f"File: {file_name}, Method: {method}, Components: {n_components}, ARI: {ari_score}")
                    results[(method, n_components)] = ari_score
        return results
    else:
        return None



def main():
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file_for_clustering, file_name) for file_name in subdir_files]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Aggregate and plot the results
    aggregated_results = {'No Reduction': []}
    for method in reduction_methods:
        for n_components in n_components_list:
            aggregated_results[(method, n_components)] = []

    for result in results:
        for key, score in result.items():
            aggregated_results[key].append(score)

    mean_ari_scores = {key: np.mean(scores) for key, scores in aggregated_results.items()}

    # Plot using Plotly
    bar_width = 0.20
    x = np.arange(len(n_components_list)) * 1.2  # Multiply by 2 to add more space between groups
    x_no_reduction = -1.2  # Keep the "No Reduction" bar offset

    fig = go.Figure()

    # Plotting no reduction with adjusted text settings
    fig.add_trace(go.Bar(
       x=[x_no_reduction], 
       y=[mean_ari_scores['No Reduction']],
       text=[f'{round(mean_ari_scores["No Reduction"], 2)}'], 
       textposition='outside',
       marker_color='orange',
       name='No Reduction',
       width=bar_width
    ))

    # Plotting with reduction methods and adjusted text
    colors = {'PCA': 'skyblue', 'KernelPCA': 'lightgreen', 'Denoising_autoencoder': 'salmon', 'Isomap': 'purple', 'MDS': 'red'}
    for i, method in enumerate(reduction_methods):
        x_method = x + (i * bar_width)
        fig.add_trace(go.Bar(
            x=x_method, 
            y=[mean_ari_scores[(method, n_components)] for n_components in n_components_list],
            text=[f'{round(mean_ari_scores[(method, n_components)], 2)}' for n_components in n_components_list],
            textposition='outside',
            marker_color=colors[method],
            name=method,
            width=bar_width
        ))

    # Add a dashed vertical line between "No Reduction" and the other bars
    fig.add_shape(
        type="line",
        x0=x_no_reduction + bar_width + 0.05,  # Shift the line slightly to the right
        y0=0,
        x1=x_no_reduction + bar_width + 0.05,  # Same x-position for a vertical line
        y1=1,  # Top of the plot
        line=dict(
           color="black",
           width=2,
           dash="dash",  # Make the line dashed
        ),
    )

    #Adjust x-axis to have space for "No Reduction" bar
    xticks = [x_no_reduction] + list(x + 2.0 * bar_width for x in x)  
    xtick_labels = ['No Reduction'] + [str(n) for n in n_components_list]
    fig.update_layout(
        xaxis=dict(
           tickvals=xticks,
           ticktext=xtick_labels,
           range=[-1.5, max(xticks) + 1.0],  # Adjust the range to fit all bars
           tickfont=dict(size=40),  # Match font size with text above bars
           showline=True,  # Afficher la ligne de l'axe x
           linecolor='black',  # Couleur de la ligne de l'axe x
           linewidth=1,  # Épaisseur de la ligne de l'axe x
           mirror='allticks',  # Ajoute une ligne en haut et en bas pour encadrer
           showgrid=False,  # Afficher la grille sur l'axe x
           ticks='outside',  # Positionner les petits traits à l'intérieur de l'axe
           tickmode='array'  # Mode de placement des ticks
        ),
        yaxis=dict(
           range=[0, 1.1], 
           tickvals=[i / 10 for i in range(11)],
           tickfont=dict(size=40),  # Match font size with text above bars
           showline=True,  # Afficher la ligne de l'axe x
           linecolor='black',  # Couleur de la ligne de l'axe x
           linewidth=1,  # Épaisseur de la ligne de l'axe x
           mirror='allticks',  # Ajoute une ligne en haut et en bas pour encadrer
           showgrid=False,  # Afficher la grille sur l'axe x
           ticks='outside',  # Positionner les petits traits à l'intérieur de l'axe
           tickmode='array'  # Mode de placement des ticks
        ),
        barmode='group',
        bargap=0.02,  # Increase the gap between bars within the same group
        bargroupgap=0.1,  # Increase the gap between bar groups
        title='Mean ARI Scores for Optics Clustering with and without Dimensionality Reduction',
        xaxis_title='number of features',
        yaxis_title='Mean ARI',
        legend_title='Methods',
        height=2000,  # Further increase figure height
        width=3600,  # Further increase figure width
        plot_bgcolor='white',  # Keep background color white
        legend=dict(
            x=1,  # Horizontal position (1 is the far right)
            y=1,  # Vertical position (1 is the top)
            xanchor='right',  # Anchor the legend to the right
            yanchor='top',  # Anchor the legend to the top
            bgcolor='rgba(255, 255, 255, 0.5)',  # Background color with transparency
            bordercolor='black',  # Border color
            borderwidth=1,  # Border width
            font=dict(
                size=40  # Taille des écritures dans la légende
            )
        ),
        xaxis_showgrid=False,  # Hide gridlines
        yaxis_showgrid=False,  # Hide gridlines
        font=dict(size=40),  # Ensure consistent font size for the entire plot
        uniformtext_minsize=40,  # Set a minimum text size for all elements
        uniformtext_mode='show'  # Ensure uniform text display
    )

    # Show the figure
    fig.show()

    # Save the figure as an image
    output_path = '/home/ousmane/results/Plot_clustering_denoising_optics.png'
    fig.write_image(output_path)
    print(f"Figure saved as {output_path}")


if __name__ == "__main__":
    main()