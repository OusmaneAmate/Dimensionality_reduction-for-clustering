# Dimensionality_reduction-for-clustering

# 1. Introduction
- Project context and objective:
  This project aims to investigate whether the use of dimensionality reduction techniques can improve clustering, i.e., achieve better-separated clusters.
# 2. Methodology
- Data used:
  We conducted experiments on both artificial and real data.
- Data normalization:
  Before applying clustering, the data is normalized using StandardScaler from the scikit-learn library in Python.
# 3. Clustering algorithms
- Algorithms used:
  We used four different clustering algorithms:
  - Kmeans
  - Agglomerative Clustering
  - Gaussian Mixture
  - Optics
- Evaluation of the results
  Clustering results are evaluated using the Adjusted Rand Index (ARI) metric, which measures the similarity between two data groupings.
# 4. Dimensionality reduction
- Dimensionality reduction methods
  Several dimensionality reduction methods were applied:
  - Linear methods: PCA (Principal Component Analysis)
  - Non-linear methods: KernelPCA, Isomap, MDS (Multidimensional Scaling)
  - Autoencoders: Classic Autoencoder, Variational Autoencoder, Denoising Autoencoder
- Choice of target dimensions
  The chosen target dimensions for each method are:
  - r = K-1 (where K is the number of clusters)
  - r = 25% of D (where D is the initial dimension of the data)
  - r = 50% of D
# 5. Comparison of results
- Clustering on original data:
  Clustering is first performed on the normalized original data.
- Clustering after dimensionality reduction:
  Once the dimensionality is reduced, clustering is performed again on the data obtained from each dimensionality reduction method.
- Comparison of ARI scores:
  The ARI scores obtained for each dimensionality reduction method are compared with those from the original data to determine which methods improve clustering, i.e., have 
better ARI scores than the data before dimensionality reduction.

# 6. Case study: Image segmentation
- Dataset presentation:
  The "Image Segmentation" dataset, downloaded from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/50/image+segmentation), contains 2310 observations, 19 features, and 7 classes. Each class corresponds to a cluster, the original data has 19 dimensions (D=19).
- Target dimensions for the experiment:
  According to the experiment, the target dimensions are 5, 6, and 10.
- Results and interpretation:
  The experiment results on this dataset are summarized in the table below.





<table>
  <tr>
    <th></th>
    <th>No Reduction</th>
    <th colspan="3">PCA</th>
    <th colspan="3">KernelPCA</th>
    <th colspan="3">Variational Autoencoder</th>
    <th colspan="3">Isomap</th>
    <th colspan="3">MDS</th>
  </tr>
  <tr>
    <td>Number of features</td>
    <td>19</td>
    <td>5</td>
    <td>6</td>
    <td>10</td>
    <td>5</td>
    <td>6</td>
    <td>10</td>
    <td>5</td>
    <td>6</td>
    <td>10</td>
    <td>5</td>
    <td>6</td>
    <td>10</td>
    <td>5</td>
    <td>6</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Kmeans</td>
    <td>0.47</td>
    <td>0.48</td>
    <td>0.43</td>
    <td>0.47</td>
    <td>0.24</td>
    <td>0.47</td>
    <td>0.45</td>
    <td>0.44</td>
    <td>0.47</td>
    <td>0.5</td>
    <td>0.45</td>
    <td>0.44</td>
    <td>0.46</td>
    <td>0.46</td>
    <td>0.47</td>
    <td>0.47</td>
  </tr>
  <tr>
    <td>Agglomerative</td>
    <td>0.35</td>
    <td>0.45</td>
    <td>0.45</td>
    <td>0.35</td>
    <td>0.4</td>
    <td>0.43</td>
    <td>0.44</td>
    <td>0.47</td>
    <td>0.44</td>
    <td>0.47</td>
    <td>0.34</td>
    <td>0.37</td>
    <td>0.39</td>
    <td>0.45</td>
    <td>0.43</td>
    <td>0.36</td>
  </tr>
  <tr>
    <td>Gaussian mixture</td>
    <td>0.43</td>
    <td>0.45</td>
    <td>0.4</td>
    <td>0.47</td>
    <td>0.38</td>
    <td>0.41</td>
    <td>0.49</td>
    <td>0.42</td>
    <td>0.45</td>
    <td>0.46</td>
    <td>0.53</td>
    <td>0.4</td>
    <td>0.53</td>
    <td>0.44</td>
    <td>0.51</td>
    <td>0.4</td>
  </tr>
  <tr>
    <td>Optics</td>
    <td>0.25</td>
    <td>0.27</td>
    <td>0.22</td>
    <td>0.33</td>
    <td>0.31</td>
    <td>0.3</td>
    <td>0.32</td>
    <td>0.26</td>
    <td>0.27</td>
    <td>0.47</td>
    <td>0.21</td>
    <td>0.2</td>
    <td>0.27</td>
    <td>0.32</td>
    <td>0.32</td>
    <td>0.34</td>
  </tr>
</table>


