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
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
import re
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler
import umap
import plotly.graph_objects as go

os.environ['OMP_NUM_THREADS'] = '1'
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the directory containing the files
extraction_dir = '/home/ousmane/200_Features/'
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

def autoencoder_reduction(X, n_components):
    try:
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

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

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Entraînement du modèle
        history = autoencoder.fit(X_train, X_train,
                                  epochs=200,
                                  batch_size=64,
                                  shuffle=False,
                                  validation_data=(X_test,X_test),
                                  verbose=1)
        
        # Affichage de la courbe de perte
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig(f'/home/ousmane/results/Plot_loss_curve_agglomerative.png')
        plt.show()


        encoder_model = Model(inputs=input_layer, outputs=latent_layer)
        X_reduced = encoder_model.predict(X)
        return X_reduced
    except Exception as e:
        logging.error(f"Error in autoencoder reduction: {e}")
        return None


def apply_dimensionality_reduction(X, y, method, n_components):
    try:
        # Normaliser X avant la réduction
        X = normalize_data(X)

        if method == 'PCA':
            reducer = PCA(n_components=n_components)
        elif method == 'KernelPCA':
            reducer = KernelPCA(n_components=n_components, kernel='rbf')
        elif method == 'Autoencoder':
            return autoencoder_reduction(X, n_components)
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
        model = AgglomerativeClustering(n_clusters=n_clusters, metric = "cosine", linkage = "average")
        y_pred = model.fit_predict(X)
        score = ari(y, y_pred)
        return score
    except Exception as e:
        logging.error(f"Error in clustering and evaluating: {e}")
        return None

# List of dimensionality reduction methods
reduction_methods = ['PCA', 'KernelPCA', 'Autoencoder', 'Isomap', 'MDS']

# List of n_components to try
n_components_list = [2, 4, 9]

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
        
        # Appliquer le clustering avec la valeur de k extraite
        best_score_no_reduction = -1
        score = cluster_and_evaluate(X, y, k)
        if score is not None and score > best_score_no_reduction:
            best_score_no_reduction = score
        
        results = {'No Reduction': best_score_no_reduction}
        
        for method in reduction_methods:
            for n_components in n_components_list:
                X_reduced = apply_dimensionality_reduction(X, y, method, n_components)
                if X_reduced is not None:
                    best_score_reduction = -1
                    score = cluster_and_evaluate(X_reduced, y, k)
                    if score is not None and score > best_score_reduction:
                        best_score_reduction = score
                    results[(method, n_components)] = best_score_reduction
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
    x = np.arange(len(n_components_list)) * 1.8  # Multiply by 2 to add more space between groups
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
    colors = {'PCA': 'skyblue', 'KernelPCA': 'lightgreen', 'Autoencoder': 'salmon', 'Isomap': 'purple', 'MDS': 'red'}
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
        x0=x_no_reduction + bar_width + 0.1,  # Shift the line slightly to the right
        y0=0,
        x1=x_no_reduction + bar_width + 0.1,  # Same x-position for a vertical line
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
           range=[0, 1], 
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
        bargap=0.5,  # Increase the gap between bars within the same group
        bargroupgap=0.7,  # Increase the gap between bar groups
        title='Mean ARI Scores for Agglomerative Clustering with and without Dimensionality Reduction',
        xaxis_title='number of features',
        yaxis_title='Mean ARI',
        legend_title='Methods',
        height=2000,  # Further increase figure height
        width=3000,  # Further increase figure width
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
    output_path = '/home/ousmane/results/Plot_clustering_agglomerative.png'
    fig.write_image(output_path)
    print(f"Figure saved as {output_path}")


if __name__ == "__main__":
    main()