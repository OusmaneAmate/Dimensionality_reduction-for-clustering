# Dimensionality_reduction-for-clustering
The goal of this project is to see if the use of dimension reduction techniques can improve clustering, in other words have better separated clusters. To do this, we carried out experiments on artificial data and on real data. These experiments first involve importing the data and then normalizing them using StandardScaler from the scikit-learn library in python, once the data is normalized we apply clustering to this data.
To better interpret and compare the results we used four different clustering algorithms namely Kmeans, Agglomerative clustering, Gaussian mixture and Optics. First we do the clustering on the original data normalized with the different clustering algorithms mentioned above and then we evaluate the clustering results with the Adjusted Rand Index (ARI) metric which is a measure used to evaluate the similarity between two data groupings by comparing pairs of samples assigned to the same group or to different groups.
The value of ARI score is between -1 and 1, with 0 indicating random labeling and 1 indicating identical clusters. After clustering on the original data, dimension reduction is applied with linear methods for dimension reduction such as PCA (Principal Component Analysis), nonlinear methods such as KernelPCA, Isomap, and MDS (Multidimensional Scaling) and the Autoencoder (Classic Autoencoder, Variational Autoencoder and Denoising Autoencoder). Regarding the target dimension, we have as values ​​r = K-1, r = 25% of D and r = 50% of D with r the target dimension, K the number of clusters and D the initial dimension of the data.
Once the dimension is reduced through these different dimension reduction techniques, we perform the clustering again on the new data obtained using each method with the same algorithms as those used on the original data, then we calculate the ARI score on the clustering for each dimension reduction method. And finally we compare the ARI score of the clustering on the original data to those of the clustering on the data of each method to see which methods allow us to have a better clustering, that is to say an ARI score better than that of the data before the dimension reduction.
Let's take the example of the Image segmentation dataset that we downloaded from UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/50/image+segmentation) containing instances randomly drawn from a database of 7 outdoor images, these images were manually segmented to create a classification for each pixel. The dataset contains 2310 observations, 19 features and 7 classes. The number of classes corresponds to the number of clusters (7 clusters), the number of features corresponds to the dimension (D=19) of the original data. So according to the experiment presented above, the target dimensions must be equal to 5, 6 and 10. The results of the experiment are grouped in the table below.


<table>
  <tr>
    <th>Algorithms</th>
    <th>No Reduction</th>
    <th colspan="3">PCA</th>
    <th colspan="3">KernelPCA</th>
    <th colspan="3">VAE Autoencoder</th>
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


