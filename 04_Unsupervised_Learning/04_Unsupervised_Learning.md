# Supervised Machine Learning: Regression

These are my notes and the code of the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) offered by IBM & Coursera.

The Specialization is divided in 6 courses, and each of them has its own folder with its guide & notebooks:

1. [Exploratory Data Analysis for Machine Learning](https://www.coursera.org/learn/ibm-exploratory-data-analysis-for-machine-learning?specialization=ibm-machine-learning)
2. [Supervised Machine Learning: Regression](https://www.coursera.org/learn/supervised-machine-learning-regression?specialization=ibm-machine-learning)
3. [Supervised Machine Learning: Classification](https://www.coursera.org/learn/supervised-machine-learning-classification?specialization=ibm-machine-learning)
4. [Unsupervised Machine Learning](https://www.coursera.org/learn/ibm-unsupervised-machine-learning?specialization=ibm-machine-learning)
5. [Deep Learning and Reinforcement Learning](https://www.coursera.org/learn/deep-learning-reinforcement-learning?specialization=ibm-machine-learning)
6. [Specialized Models: Time Series and Survival Analysis](https://www.coursera.org/learn/time-series-survival-analysis?specialization=ibm-machine-learning)

This file focuses on the **fourth course: Unsupervised Machine Learning**

Mikel Sagardia, 2022.  
No guarantees

## Overview of Contents

- [Supervised Machine Learning: Regression](#supervised-machine-learning-regression)
  - [Overview of Contents](#overview-of-contents)
  - [1. Introduction to Unsupervised Learning](#1-introduction-to-unsupervised-learning)
    - [1.1 Curse of Dimensionality](#11-curse-of-dimensionality)
    - [1.2 Examples](#12-examples)
    - [1.3 Common Use Cases](#13-common-use-cases)
  - [2. K-Means Clustering](#2-k-means-clustering)
    - [2.1 Smart Initialization](#21-smart-initialization)
    - [2.2 Metrics for Choosing the Right Number of Clusters `K` and the Correct Clustering](#22-metrics-for-choosing-the-right-number-of-clusters-k-and-the-correct-clustering)
    - [2.3 Python Implementation](#23-python-implementation)
    - [2.4 Python Lab: K-Means](#24-python-lab-k-means)
    - [2.5 Python Notebook: K-Means](#25-python-notebook-k-means)
    - [2.6 Gaussian Mixture Models (GMM)](#26-gaussian-mixture-models-gmm)
      - [Applications of the Gaussian Mixture Models](#applications-of-the-gaussian-mixture-models)
      - [Python Syntax](#python-syntax)
    - [2.7 Python Lab: Gaussian Mixture Models](#27-python-lab-gaussian-mixture-models)
  - [3. Computational Difficulties of Clustering Algorithms: Distance Measures](#3-computational-difficulties-of-clustering-algorithms-distance-measures)
    - [3.1 Cosine and Jaccard Distance](#31-cosine-and-jaccard-distance)
    - [3.2 Python Demo: Curse of Dimensionality](#32-python-demo-curse-of-dimensionality)
    - [3.3 Python Notebook: Distance Metrics](#33-python-notebook-distance-metrics)
  - [4. Common Clustering Algorithms](#4-common-clustering-algorithms)
    - [4.1 Hierarchical Agglomerative Clustering](#41-hierarchical-agglomerative-clustering)
    - [4.2 Hierarchical Linkage Types](#42-hierarchical-linkage-types)
    - [4.3 Python Syntax for Hierarchical Agglomerative Clustering](#43-python-syntax-for-hierarchical-agglomerative-clustering)
    - [4.4 DBSCAN: Density-Based Spatial Clustering of Applications with Noise](#44-dbscan-density-based-spatial-clustering-of-applications-with-noise)
      - [Algorithm](#algorithm)
      - [Discussion](#discussion)
      - [Python Syntax](#python-syntax-1)
    - [4.5 Python Lab: DBSCAN](#45-python-lab-dbscan)
    - [4.5 Mean Shift](#45-mean-shift)
      - [Discussion](#discussion-1)
      - [Python Syntax](#python-syntax-2)
    - [4.6 Python Lab: Mean Shift](#46-python-lab-mean-shift)
  - [5. Comparing Clustering Algorithms](#5-comparing-clustering-algorithms)
    - [5.X Comparison Table](#5x-comparison-table)

## 1. Introduction to Unsupervised Learning

In supervised learning data points have a known outcome: the label. In contrast, in **unsupervised learning**, we have no known outcome and we try to learn about the structure of the dataset. There two major applications:

1. Clustering: identify unknown structure. Examples:
   - K-means
   - Hierarchical Agglomerative Clustering
   - DBSCAN
   - Mean Shift
2. Dimensionality reduction: use structural characteristics to simplify the dataset. Examples:
   - Principal Component Analysis (PCA)
   - Non-negative Matrix Factorization

### 1.1 Curse of Dimensionality

Why would we want to decrease the dimensionality?

In practice, high dimensions have many drawbacks and we talk about the *curse of dimensionality*:

- With more features, the risk of having correlations between them increases, and that correlation might be only in the training set.
- More features might introduce noise we need to learn to filter, otherwise we overfit
- Imagine KNN or any other distance based algorithm: with more dimensions, the number of data-points we need to cover the complete feature space increases exponentially; for a data point to get its proper nearest neighbors, we need many training points.
- We need more computational power to train on datasets with high dimensionality.
- Higher incidence of outliers.

In the image, a variable of 1D has 10 categories; if we add more variables like that, the number of data-points to cover the feature spaces increases exponentially.

![Curse of Dimensionality](./pics/curse_of_dimensionality.jpg)

We can decrease the dimensionality with:

- Feature selection.
- Dimensionality reduction, e.g., with PCA.

### 1.2 Examples

Example 1: Customer Churn, which has originally 54 features:

- We can cluster similar customers.
- We can apply dimensionality reduction.

![Overview of Application](./pics/unsupervised_learning_overview_1.jpg)

Example 2: News articles grouping by topics.

### 1.3 Common Use Cases

Clustering:

- Classification: spam filter
- Anomaly detection: fraudulent transactions
- Customer segmentation
- **Improvement of supervised models: find clusters and apply supervised models for each cluster! That doesn't work always, but it's often worth trying.**

Dimension Reduction:

- Compress high resolution images; with compressed images the performance of object detection algorithms is accelerated
- etc.

## 2. K-Means Clustering

In K-Means, we set the number of `K` clusters we'd like to segment our dataset. Then the algorithm works as follows:

1. Create `K` random centroids in feature space.
2. For each data point, compute the distance to each cluster centroid `C` and assign the closest one; we have (re-)clustered all the data points.
3. Recompute the cluster centroids as the mean of all the data points in each cluster.
4. Repeat 2-3 until convergence (cluster centroids don't move anymore) or maximum number of iterations.

![K-Means](./pics/k_means.jpg)

With K-means there can be multiple clustering solutions that converge successfully.

### 2.1 Smart Initialization

The selection of the initial random centroids `C` is key to prevent local optima and improve the convergence speed:

- Naive: We can take random data points.
- Better: **K-Means ++** (default in `sklearn`): We take random points far away from each other, as follows:
  - We set the first random centroid. 
  - Then, compute weights for all points such that `w = d^2 / sum(d^2)`, i.e., the larger the distance from point to centroid the larger the weight.
  - Then, we sum the weighted data points and we get the second centroid. 
  - If we have more centroids, we take the minimum distance to any centroid as `d`.

### 2.2 Metrics for Choosing the Right Number of Clusters `K` and the Correct Clustering

Sometimes the number of clusters `K` is given by the business problem.

But in other cases, we want to discover it! We can evaluate the clustering performance with the **inertia** metric. The inertia of cluster `k` with `k = 1 ... K`:

`inertia_k = sum(i = 1:n; (x_i - C_k)^2)`

with 

- centroids `C_k`
- `n` data points `x_i` in the cluster

Smaller values of the inertia correspond to tighter clusters, but the metric is sensitive to the number of points in the clusters, `n`.

Another metric is the **distortion**, the average of the **inertia**:

`distortion_k = (1/n) * inertia_k = (1/n) sum(i = 1:n; (x_i - C_k)^2)`

The distortion doesn't increase with the number of points.

Which one should we take?

- If we are concerned with the similarity of the points only, take **inertia**.
- If we want clusters with similar numbers of points, take **distortion**.

The general approach would be:

1. Initialize the random centroids with a given `K`
2. Fit the model and compute the selected metric: inertia / distortion.
3. Repeat 1-2 several times.
4. Change `K` and repeat 1-3.

At the end, we take the `K`-clustering combination with the lowest selected metric However, the reality is that as `K` increases, the metrics will decrease; we follow the *elbow method* to choose the smallest `K` possible that yields a low metric:

![Elbow Method](./pics/k_means_elbow.jpg)

Before the *elbow* or inflection point the inertia/distortion decrease dramatically, but after it they plateau.

Visual inspection of the labeled data points is always not possible, because of high dimensions; thus, we use the elbow method.

In the case the inertia/distortion curve is not clear for the elbow method, we can use the [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score).

### 2.3 Python Implementation

```python
# Alternative with mini-batches: MiniBatchKMeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3,
                init='k-means++')

kmeans.fit(X1)
y_pred = kmeans.predict(X2)

# Inertia
kmeans.inertia_

# Cluster centers
km.cluster_centers_
# Labels
km.labels_

# Elbow method
inertia = []
for k in range(10):
    kmeans = KMeans(n_clusters=k,
                    init='k-means++')
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)
```

### 2.4 Python Lab: K-Means

In this notebook,

`./lab/04a_LAB_KMeansClustering.ipynb`,

three applications are shown:

1. Clustering of a synthetic dataset with K-means.
2. Optimum clustering of a synthetic dataset with the elbow method using K-means.
3. Compression of an image with K-means.

The most important lines are summarized in the following:

```python

# Setup and imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

### -- 1. Clustering of a synthetic dataset with K-means

# Helper function that allows us to display data
# in 2 dimensions an highlights the clusters
def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk'
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1], c=color[0], alpha=alpha, s=s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0], X[km.labels_==i,1], c=color[i], alpha=alpha, s=s)
            plt.scatter(km.cluster_centers_[i][0], km.cluster_centers_[i][1], c=color[i], marker = 'x', s = 100)

# We define our dataset as a ring of points
# Infinite clusterings are possible due to rotation symmetry
angle = np.linspace(0,2*np.pi,20, endpoint = False)
X = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
# No model yet
display_cluster(X)

num_clusters = 2
# random_state controls the randomness of the initial state
# we can also modify init; look at the docu
km = KMeans(n_clusters=num_clusters,random_state=10,n_init=1) # n_init, number of times the K-mean algorithm will run
km.fit(X)
# Now we have a model
display_cluster(X,km,num_clusters)

# We change the random state
km = KMeans(n_clusters=num_clusters,random_state=20,n_init=1)
km.fit(X)
display_cluster(X,km,num_clusters)

### -- 2. Optimum clustering of a synthetic dataset with the elbow method using K-means

n_samples = 1000
n_bins = 4 
centers = [(-3, -3), (0, 0), (3, 3), (6, 6)]
# make_blobs takes centers and std and creates random points
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
# We display without model: all points in a color.
display_cluster(X)

# We run it with the original number of clusters
num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(X)
display_cluster(X,km,num_clusters)

# Get optimum number of clusters with the elbow method
# Note that in our case the optimum would be 4
# but the inertia ALWAYS decreases with K!
# Sometimes it is not clear where the elbow is,
# we should take it where the inertia considerably flattens.
inertia = []
list_num_clusters = list(range(1,11))
for num_clusters in list_num_clusters:
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    inertia.append(km.inertia_)
    
plt.plot(list_num_clusters,inertia)
plt.scatter(list_num_clusters,inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia');

### -- 3. Compression of an image with K-means.

img = plt.imread('peppers.jpg', format='jpeg')
plt.imshow(img)
plt.axis('off')

# Each pixel with its [R,G,B] values becomes a row
# -1 = img.shape[0]*img.shape[1], because we leave the 3 channels
# as the second dimension
img_flat = img.reshape(-1, 3)

img.shape # (480, 640, 3)
img_flat.shape # (307200, 3)

# Note that in reality we have 256^3 possible colors = 16,777,216
# but not all of them are used.
# All the unique/used colors
len(np.unique(img_flat,axis=0)) # 98452

# K=8 clusters: we allow 8 colors
kmeans = KMeans(n_clusters=8, random_state=0).fit(img_flat)

# Loop for each cluster center
# Assign to all pixels with the cluster label
# the color of the cluster == the cluster centroid
img_flat2 = img_flat.copy()
for i in np.unique(kmeans.labels_):
    img_flat2[kmeans.labels_==i,:] = kmeans.cluster_centers_[i]

img2 = img_flat2.reshape(img.shape)
plt.imshow(img2)
plt.axis('off');

# Function which compresses an image to k colors
def image_cluster(img, k):
    img_flat = img.reshape(img.shape[0]*img.shape[1],3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flat)
    img_flat2 = img_flat.copy()

    # loops for each cluster center
    for i in np.unique(kmeans.labels_):
        img_flat2[kmeans.labels_==i,:] = kmeans.cluster_centers_[i]
        
    img2 = img_flat2.reshape(img.shape)
    return img2, kmeans.inertia_

# Call the function for k between 2 and 20,
# and draw an inertia curve
k_vals = list(range(2,21,2))
img_list = []
inertia = []
for k in k_vals:
    img2, ine = image_cluster(img,k)
    img_list.append(img2)
    inertia.append(ine)

# Plot to find optimal number of clusters
plt.plot(k_vals,inertia)
plt.scatter(k_vals,inertia)
plt.xlabel('k')
plt.ylabel('Inertia');
```

### 2.5 Python Notebook: K-Means

In this notebook,

`./lab/KMeansClustering.ipynb`,

two applications are shown:

1. Customer segmentation, based on 4 features; a scatterplot of two features shows 5 clear clusters.
2. Image compression.

However, the notebook doesn't introduce anything new compared to the previous one.

### 2.6 Gaussian Mixture Models (GMM)

This section has no videos, just a reading. I summarize it here.

[Gaussian Mixture Models (GMM)](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95) consists of `K` Gaussian blobs computed in clusters; we have one blob `i` for each detected cluster with three parameters:

- the centroid (mean vectors: `mu_i`),
- the density (covariance matrix: `sigma_i`),
- and the mixing coefficient (`pi_i`), which is the weight of each blob (all weights add up to 1).

Each blob tells us the probability of a point in the feature space of belonging to the blob-cluster. The model has a complete distribution, which is the *sum* of all blobs.

![Gaussian Mixture Models: 1D](./pics/contreras_medium_gmm.png)

When all parameters are computed, the PDF of the mixture model is formulated as:

![GMM: PDF](./pics/gmm_pdf.jpg)

That formula can be used for **anomaly detection**.

While `p(x)` predicts the probability of a point belonging to any cluster, we can also obtain the probability of a point `x_n` of belonging to the cluster `i`: `p(i|x_n)`.

![GMM: Probabilities](./pics/gmm_probabilities.png)

The blobs are obtained with the [**Expectation Maximization**](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm) algorithm, which iteratively finds the optimum parameters. Usually, an initial guess of the clusters is provided (e.g., provided via K-means), and then those blobs are optimized. The optimization works with the derivatives of the equations.

#### Applications of the Gaussian Mixture Models

- Recommender systems: by similar user clustering.
- Anomaly detection: identification of data that are out of the general normal distribution.
- Clustering.

The main difference with K-means: 

- K-means is a *hard* clustering algorithm: it says whether a point belongs to a cluster or not.
- GMM is a *soft* clustering algorithm: it tells the probability of a point of belonging to different clusters.
- It is more informative to have a probability, since we can decide to take different clusters for a point depending on the business case.

#### Python Syntax

We need to define

- the number of blobs we'd like to detect: `n_components`,
- and the type of covariance matrix: `covariance_type`:
  - `full`: each component has its own general covariance matrix.
  - `tied`: all components share the same general covariance matrix.
  - `diag`: each component has its own diagonal covariance matrix.
  - `spherical`: each component has its own single variance.

Additionally, we can set `init_params`, which defines the initial clustering method; by default it's `kmeans`.

```python
from sklearn.mixture import GaussianMixture as GMM

# covariance_type
# full: each component has its own general covariance matrix.
# tied: all components share the same general covariance matrix.
# diag: each component has its own diagonal covariance matrix.
# spherical: each component has its own single variance.
gmm = GMM(n_components=3, covariance_type='tied', init_params='kmeans')
gmm.fit(X)

labels = gmm.predict(X)
probs = GMM.predict_proba(X)
```

### 2.7 Python Lab: Gaussian Mixture Models

In this notebook,

`./lab/GMM_v2.ipynb`,

these examples are shown:

1. Conceptual case with a univariate (1D) dataset; the case used in the reading.
2. Conceptual case with a bivariate (2D) dataset; the case used in the reading. This is a nice example to see the difference between the different `covariance_type` classes. An interesting plotting function is defined.
3. Image compression.
4. Market segmentation.

Nothing very new is learned, because the syntax is very similar to other clustering algorithms, e.g., K-means.

Below, I add the 1D and 2D plotting functions and the code related to the market segmentation example, which are the most interesting parts.

```python

###
### -- Plotting Functions
###

# This function will allow us to easily plot data taking in x values, y values, and a title
def plot_univariate_mixture(means, stds, weights, N = 10000, seed=10):
    
    """
    returns the simulated 1d dataset X, a figure, and the figure's ax
    
    """
    np.random.seed(seed)
    if not len(means)==len(stds)==len(weights):
        raise Exception("Length of mean, std, and weights don't match.") 
    K = len(means)
    
    mixture_idx = np.random.choice(K, size=N, replace=True, p=weights)
    # generate N possible values of the mixture
    X = np.fromiter((ss.norm.rvs(loc=means[i], scale=stds[i]) for i in mixture_idx), dtype=np.float64)
      
    # generate values on the x axis of the plot
    xs = np.linspace(X.min(), X.max(), 300)
    ps = np.zeros_like(xs)
    
    for mu, s, w in zip(means, stds, weights):
        ps += ss.norm.pdf(xs, loc=mu, scale=s) * w
    
    fig, ax = plt.subplots()
    ax.plot(xs, ps, label='pdf of the Gaussian mixture')
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("P", fontsize=15)
    ax.set_title("Univariate Gaussian mixture", fontsize=15)
    #plt.show()
    
    return X.reshape(-1,1), fig, ax
    
    
def plot_bivariate_mixture(means, covs, weights, N = 10000, seed=10):
    
    """
    returns the simulated 2d dataset X and a scatter plot is shown
    
    """
    np.random.seed(seed)
    if not len(mean)==len(covs)==len(weights):
        raise Exception("Length of mean, std, and weights don't match.") 
    K = len(means)
    M = len(means[0])
    
    mixture_idx = np.random.choice(K, size=N, replace=True, p=weights)
    
    # generate N possible values of the mixture
    X = np.fromiter(chain.from_iterable(multivariate_normal.rvs(mean=means[i], cov=covs[i]) for i in mixture_idx), 
                dtype=float)
    X.shape = N, M
    
    xs1 = X[:,0] 
    xs2 = X[:,1]
    
    plt.scatter(xs1, xs2, label="data")
    
    L = len(means)
    for l, pair in enumerate(means):
        plt.scatter(pair[0], pair[1], color='red')
        if l == L-1:
            break
    plt.scatter(pair[0], pair[1], color='red', label="mean")
    
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Scatter plot of the bivariate Gaussian mixture")
    plt.legend()
    plt.show()
    
    return X


def draw_ellipse(position, covariance, ax=None, **kwargs):
    
    """
    Draw an ellipse with a given position and covariance
    
    """
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
        
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

###
### -- Market Segmentation
###

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

data = pd.read_csv("customers.csv")
data.head()
data.shape # (2216, 19)

# Scale
SS = StandardScaler()
X = SS.fit(data).transform(data)

# PCA: Reduce to 2D to plot
pca2 = PCA(n_components=2)
reduced_2_PCA = pca2.fit(X).transform(X)

model = GaussianMixture(n_components=4, random_state=0)
model.fit(reduced_2_PCA)
PCA_2_pred = model.predict(reduced_2_PCA)

# Plot
x = reduced_2_PCA[:,0]
y = reduced_2_PCA[:,1]
plt.scatter(x, y, c=PCA_2_pred)
plt.title("2d visualization of the clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

# PCA with n=3 and clustering
pca3 = PCA(n_components=3)
reduced_3_PCA = pca3.fit(X).transform(X)
mod = GaussianMixture(n_components=4, random_state=0)
PCA_3_pred = mod.fit(reduced_3_PCA).predict(reduced_3_PCA)

# 3D Plotting
reduced_3_PCA = pd.DataFrame(reduced_3_PCA, columns=(['PCA 1', 'PCA 2', 'PCA 3']))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(reduced_3_PCA['PCA 1'],reduced_3_PCA['PCA 2'],reduced_3_PCA['PCA 3'], c=PCA_3_pred)
ax.set_title("3D projection of the clusters")
```

## 3. Computational Difficulties of Clustering Algorithms: Distance Measures

Clustering methods rely very heavily on distance measures. There are several distance metrics and each one has pros & cons.

The typical distance measure is the **Euclidean Distance** or **L2**:

`d(x,y) = srqt(sum((x-y)^2))`

Another distance measure: **Manhattan, L1**: `d(x,y) = sum(abs(x-y))`

- It will be larger than L2.
- It is often used in cases with very high dimensionality, because distance values between point pairs become more unique than L2.

### 3.1 Cosine and Jaccard Distance

**Cosine Distance** takes to data point vectors and computes the angle between both in the feature space:

![Cosine Distance](./pics/cosine_distance.jpg)

The output is `phi = arccos(cos(phi))`, with these properties:

- `cos = 0`: 90 degrees between the vectors
- `cos = 1`: 0 degrees between the vectors, i.e., the are aligned
- `cos = -1`: opposite direction
- The scaling from the origin is not relevant anymore: the end points can be very far away from each other but have the same cosine close to 1.
- We are more interested between the **relationships between the features** than the distance between the data points in feature space! **How different are the relationships of features between two data points?** `cos = 1` very similar, `cos = 0` orthogonal, `cos = -1` opposite.

However, we should **always check the formula/definition of the cosine distance in the used library**; for instance, in `scipy` the cosine distance is defined as

`d = 1 - cos`, thus it belongs to the range `[0,2]`, with

- $0$: "in the same direction"
- $1$: "perpendicular"
- $2$: "in the opposite direction."

We often distinguish between **cosine similarity** and **cosine distance**; for instance, `sklearn` (and in other frameworks, too):

`cosine_distance = 1 - cosine_similarity`

When should we use the cosine distance?

- Euclidean distance is useful for coordinate-based measurements, when the location of the end-point is important.
- Cosine is better for data that can contain many similar vectors, e.g., text: one text can be the summary of another so the vector end-points are relatively far from each other, but they are aligned! In other words, the location/occurrence/coordinate of the end-point is not important.
- Euclidean distance is more sensitive to the curse of dimensionality.

**Jaccard Distance** is another distance measure used also with text or, in general, with sets of sequences: it measures the intersection over union of unique words/items (i.e., sets) that appear in two texts/sequences:

![Jaccard Distance](.(pics/jaccard_distance.jpg))

Thus, it measures the similarity of two texts/sequences based on their words/items:

- $1$ means the two sets have nothing in common.
- $0$ means the two sets are identical.

Note: `sklearn.metrics.jaccard_score` calculates the **Jaccard similarity score**, which is **1 - Jaccard distance**.

### 3.2 Python Demo: Curse of Dimensionality

In this notebook,

`./lab/04b_DEMO_Distance_Dimensionality.ipynb`,

the instructor shows why in higher dimensions the data points start being more far ways.

The analogy with a sphere is used:

- If we circumscribe a circumference of radius R into a square os side R, 21% of the points are outside the circle. Let's call those outside points *distant* points.
- If we have a sphere and a cube, the distant points are 48%. Thus, the probability of having points that are far away from each other increases.
- Then, a simulation is done and we see that the percentage dramatically increases with higher dimensions. In fact, with 14 dimensions, less than `2.9e-03 %` of the points are inside the hypersphere. The decrease seems exponential (is it exponential?).

Some other experiments are done with synthetic datasets created with `make_classification`, but the notebook code is not that interesting in terms of practical usage.

### 3.3 Python Notebook: Distance Metrics

In this notebook,

`./lab/DistanceMetrics.ipynb`,

different distance measures are analyzed.

First two auxiliary functions are defined:

1. Average distance: given 2 X & Y datasets with rows of feature vectors, the distances between each x in X to each y in Y are computed, and then, the average.

![Average Distance](./pics/avg_distance.png)

2. Pairwise distance: given 2 X & Y datasets with rows of feature vectors, the distance between paired rows is computed, and then, the average.

![Pairwise Distance](./pics/pairwise_distance.png)

Then, functions and datasets are passed to those functions to see the effect of using one or another distance measure.

The notebook is not that interesting; the most relevant section to me is an example in which two groups or strata with categorical features are compared using the Jaccard distance: In deed, if we one-hot encode categorical columns, we can compute distances between datasets/samples.

```python
# Scipy API of distance functions
# cityblock = Manhattan
from scipy.spatial.distance import euclidean, cityblock, cosine

# Scikit-Learn API of distance functions
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_distances, paired_euclidean_distances, paired_manhattan_distances, cosine_similarity

# This function will allow us to find the average distance between two sets of data
def avg_distance(X1, X2, distance_func):
    from sklearn.metrics import jaccard_score
    #print(distance_func)
    res = 0
    for x1 in X1:
        for x2 in X2:
            if distance_func == jaccard_score: # the jaccard_score function only returns jaccard_similarity
                res += 1 - distance_func(x1, x2)
            else:
                res += distance_func(x1, x2)
    return res / (len(X1) * len(X2))

# This function will allow us to find the average pairwise distance
def avg_pairwise_distance(X1, X2, distance_func):
    return sum(map(distance_func, X1, X2)) / min(len(X1), len(X2))

# The function avg_pairwise_distance(X1, X2, distance_func)
# is equivalent to
paired_euclidean_distances(X1, X2).mean()

####

# Jaccard distance for categorical datasets
df = pd.read_csv('breast-cancer.csv')
df.head()
df.columns # All categorical, even age
# 'Class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
# 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'

print(sorted(df['age'].unique()))
# ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

# One-hot encode all columns except age
from sklearn.preprocessing import OneHotEncoder
OH = OneHotEncoder()
X = OH.fit_transform(df.loc[:, df.columns != 'age']).toarray()

# Take two strata: two age groups
X30to39 = X[df[df.age == '30-39'].index]
X60to69 = X[df[df.age == '60-69'].index]
X30to39.shape, X60to69.shape
# ((36, 39), (57, 39))

avg_distance(X30to39, X30to39, jaccard_score)
# 0.6435631883548536

avg_distance(X60to69, X60to69, jaccard_score)
# 0.6182114564956281

avg_distance(X30to39, X60to69, jaccard_score)
# 0.7324778699972173
```

## 4. Common Clustering Algorithms

### 4.1 Hierarchical Agglomerative Clustering

In hierarchical agglomerative clustering we iteratively find the two closest items in our dataset and cluster them together; an item can be

- a data point
- or a cluster.

In order for that to work, we need to define:

1. A good distance metric.
2. A type of linkage for computing distances between point-cluster or cluster-cluster; e.g.:
   - distance from point to average point in cluster,
   - distance between closest points between clusters,
   - etc.

With those defined, the algorithm simply finds the next two closest items in the dataset and it clusters them together. In the beginning, we'll have 2 points clustered together:

![Hierarchical Agglomerative Clustering: Start](./pics/hierarchical_agglomerative_clustering_1.jpg)

As we continue, new 2-point clusters will emerge, but at some point, clusters will start merging, since there will be no point pair closer than the actually merged point-cluster or cluster-cluster pair:

![Hierarchical Agglomerative Clustering: Merge Clusters](./pics/hierarchical_agglomerative_clustering_2.jpg)

At some point, we have only clusters and they start merging. Thus, We need a stop criterium, otherwise the algorithm makes a big unique cluster! That criterium is a distance threshold: when all clusters are further than that threshold from each other, we stop.

![Hierarchical Agglomerative Clustering: Merge Clusters](./pics/hierarchical_agglomerative_clustering_3.jpg)

### 4.2 Hierarchical Linkage Types

We can use at least 4 linkage types:

- Single: minimum pairwise distance between clusters, i.e., distance between closest points in different clusters.
  - Pro: we can achieve clear boundaries.
  - Con: susceptible to noise.
- Complete: maximum pairwise distance between clusters, i.e., compute the furthest point pairs in different clusters and select the minimum pair.
  - Pro: less susceptible to noise.
  - Con: clusters are taken apart.
- Average: distances between cluster centroids are used.
  - Pros & Cons: a mixture of the single & complete approach.
- Ward: the inertia of each cluster is computed and we merged based on best inertia, i.e., the pair which minimizes the inertia is merged; this is similar to K-means.
  - Pros & Cons: a mixture of the single & complete approach.

![Hierarchical Linkage Types: Single](./pics/linkage_single.jpg)

![Hierarchical Linkage Types: Complete](./pics/linkage_complete.jpg)

![Hierarchical Linkage Types: Average](./pics/linkage_average.jpg)

![Hierarchical Linkage Types: Ward](./pics/linkage_ward.jpg)

### 4.3 Python Syntax for Hierarchical Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering

# We can decide the number of clusters or a distance threshold as criterium:
# distance_threshold, n_clusters
agg = AgglomerativeClustering(  n_clusters=3, 
                                affinity='euclidean', # distance metric
                                linkage='ward')

agg.fit(X1)
y_pred = agg.predict(X2)
```

### 4.4 DBSCAN: Density-Based Spatial Clustering of Applications with Noise

A key feature of this algorithm is that it truly finds clusters of data, i.e., we do not partition the data:

- We can have points that don't belong to any cluster.
- It finds core points in high density regions and expands clusters from them, adding points that are at least at a given distance.
- The algorithm ends when all points have been classified in a cluster or as noise.

Inputs for DBSCAN:

- Distance metric.
- `epsilon`: radius of local neighborhood
- `n_clu`: density threshold (for a fixed `epsilon`); core points are those which have more than `n_clu` neighbors in their local `epsilon`-neighborhood.

There are 3 possible labels for any point in DBSCAN:

- Core: those which have more than `n_clu` neighbors in their local `epsilon`-neighborhood.
- Density-reachable: an `epsilon`-neighbor of a core point that has fewer than `n_clu` neighbors itself. It's still part of the cluster, because it's in the `epsilon`-neighborhood.
- Noise: point that is not part of any cluster, because it has no core points in its `epsilon`-neighborhood.

Thus, clusters are connected core and density-reachable points.

#### Algorithm

1. We take a random point that has not been labelled and insert it to a queue (e.g., FIFO)
2. We pop a/the point from the queues and draw a circle or radius `epsilon` around it.
3. If there are at least `n_clu` core points inside, it's a core point, else a density-reachable.
4. We insert all the points inside into the queue.
5. We repeat 2-4 until the queues is empty. Then, we go to step 1, and repeat. The algorithm ends when all points have been labelled.

![DBSCAN Algorithm](./pics/dbscan_algorithm.jpg)

![DBSCAN Example](./pics/dbscan_example.jpg)

#### Discussion

Strengths:

- No need to set number of clusters.
- Allows for noise.
- Can handle arbitrary-shaped clusters.

Weaknesses:

- Requires two parameters, and finding appropriate values for them can be difficult.
- Doesn't do well with clusters of different densities.

#### Python Syntax

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=3,
            min_samples=3)

df.fit(X)
# You cannot call predict,
# instead, you get the clusters for the current dataset 
# labels: -1, 0, 1, 2, ...
clusters = db.labels_
```

### 4.5 Python Lab: DBSCAN

In this notebook,

`./lab/DBSCAN.ipynb`,

two examples are shown:

1. Proving Someone Has Bad Handwriting.
2. Clustering of a Noisy Dataset.

The first uses the `8x8` MNIST dataset from `sklearn`, which is reduced to 2D with T-SNE. Then, DBSCAN is applied to identify clusters and noise/rare data points. The last 3 data points are digits of a "friend" to whom "we'd like to proof that he's got a bad handwriting"; indeed, the digits are either miss-classified or classified as noise.

The second exercise is similar to the first one, but with a synthetic dataset of shape `(1000, 2)`, i.e., no T-SNE needs to be applied.

![DBSCAN Clustering Example](./pics/dbscan_clustering_example.png)

The code of the second example is the following:

```python
df = pd.read_csv('DBSCAN_exercises.csv')
df.head()
df.shape # (1000, 2)

plt.scatter(df['x'], df['y'])
plt.show()

cluster = DBSCAN(eps=4, min_samples=4)
cluster.fit(df)
print(len(set(cluster.labels_) - {1})) # 6

print(f'{100 * (cluster.labels_ == -1).sum() / len(cluster.labels_)}%') # 0%

plt.rcParams['figure.figsize'] = (20,15)
unique_labels = set(cluster.labels_)
n_labels = len(unique_labels)
cmap = plt.cm.get_cmap('brg', n_labels)
for l in unique_labels:
    plt.scatter(
        df['x'][cluster.labels_ == l],
        df['y'][cluster.labels_ == l],
        c=[cmap(l)],
        marker='ov'[l%2],
        alpha=0.75,
        s=100,
        label=f'Cluster {l}' if l >= 0 else 'Noise')
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

```

### 4.5 Mean Shift

The **Mean Shift** clustering algorithm is very similar to the K-Means algorithm, but the cluster centroid is not shifted to the center of mass, but to the point with highest local density. The algorithm ends when all points are assigned to a cluster.

In order to measure the point with highest local density, a **window** is defined centered in each point, implemented as the **standard deviation**; then, the weighted mean of the points within that window is measured to compute the new window center. Weighting occurs according to a kernel function which computes a weight based on the distance between the point in the window and the previous center. For point, the window wanders until a convergence point, called the **mode**. the process is repeated for every point in the dataset; at the end, all the points have a mode assigned - points with the same mode belong to the same cluster.

![Mean Shift: Algorithm](./pics/mean_shift_algorithm.jpg)

In practice, for each point, the window moves in the direction of the density gradient until a **mode** is reached, i.e., the window doesn't move anymore. Note that the final centroids or modes don't need to be dataset points; they are the coordinates in which the local density is highest.

![Mean Shift: Modes](./pics/mean_shift_modes.jpg)

One common kernel for the weight computation is the RBF: Radial Basis Function, or Gaussian; the closest points to the previous centroid have more weight.

![Mean Shift: Weighted Mean, Kernel](./pics/mean_shift_weighted_mean.jpg)

Note: the window is depicted as a square in the slides; I understand the shape depends on the underlying data structure used to access the neighboring points. For instance, it could be rather a circle/sphere/hypersphre.

#### Discussion

Strengths:

- Model-free: no assumption on number or shape of clusters.
- Only one parameter: **window size or bandwidth; it is modeled as the standard deviation.**
- Robust to outliers: outliers have their own clusters.

Weaknesses:

- Result depends on windows size; it's not easy to get the correct value.
- Slow: complexity is `m*n^2`, with `m` iterations and `n` points.

#### Python Syntax

```python
from sklearn.cluster import MeanShift

ms = MeanShift(bandwidth=2)
ms.fit(X1)
y_pred = ms.predict(X2)
```

### 4.6 Python Lab: Mean Shift

In this notebook,

`./lab/Mean_Shift_Clustering_v2.ipynb`,

three examples are shown:

1. Image segmentation with color clustering.
2. People clustering in the Titanic dataset.
3. Notes and examples on how the Mean Shift algorithm works.

In particular, the first examples seems very well suited for mean shift, even though nothing really new is shown here. Perhaps, it is interesting that `sklearn` has a bandwidth estimation algorithm, which comes handy.

```python
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

##
## --- 1. Image segmentation with color clustering
##

# Load the image
img = cv.imread('peppers.jpeg')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

# Smooth the image to make the segmentation easier
img = cv.medianBlur(img, 7)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

# Show the 3D coordinates or the RGB pixels
ax = plt.axes(projection ="3d")
ax.scatter3D(img[:,:,0],img[:,:,1],img[:,:,2])
ax.set_title('Pixel Values ')
plt.show()

# Reshape and convert to float: (194, 259, 3) -> (50246, 3)
X = img.reshape((-1,3))
X = np.float32(X)

# Estimate the bandwidth, parameters:
# - X: (n_samples, n_features)
# - quantile: float, default=0.3 Should be between [0, 1]; 0.5 = median of all pairwise distances used.
# - n_samples: int, number of samples to be used; if not given, all samples used.
bandwidth = estimate_bandwidth(X, quantile=.06, n_samples=3000)

# Mean Shift, parameters:
# - max_itert: (default=300) maximum number of iterations per seed point, if not converged.
# - bin_seeding :if True, initial kernel locations are not locations of all points, but rather the location of the discretized version of points.
ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(X)

# Get labels for each data point
labeled=ms.labels_
# Predict clusters
clusters=ms.predict(X)
# In this case, clusters and labeled are the same
sum(clusters-labeled) # 0

# Get all unique clusters
np.unique(labeled) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# Get cluster centers
ms.cluster_centers_ # (12, 3) array
# Convert to int
cluster_int8=np.uint8(ms.cluster_centers_)

# Plot cluster centers in 3D
ax = plt.axes(projection ="3d")
ax.set_title('Pixel Cluster Values  ')
ax.scatter3D(cluster_int8[:,0],cluster_int8[:,1],cluster_int8[:,2],color='red')
plt.show()

# Draw the segmented image
result=np.zeros(X.shape,dtype=np.uint8)
for label in np.unique(labeled):
    result[labeled==label,:]=cluster_int8[label,:]    
result=result.reshape(img.shape)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()


##
## --- 2. People clustering in the Titanic dataset
##

df = pd.read_csv("titanic.csv") # (891, 12)

# Take features, replace values, impute, scale
df = df.drop(columns=['Name','Ticket','Cabin','PassengerId','Embarked'])
df.loc[df['Sex']!='male','Sex']=0
df.loc[df['Sex']=='male','Sex']=1
df['Age'].fillna(df['Age'].mean(),inplace=True)
X = df.apply(lambda x: (x-x.mean())/(x.std()+0.0000001), axis=0)
# Final dataset: all numerical, scaled, 
X.shape # (891, 7)

# Estimate bandwidth and apply Mean Shift
bandwidth = estimate_bandwidth(X)
ms = MeanShift(bandwidth=bandwidth , bin_seeding=True)
ms.fit(X)

# Append cluster prediction
X['cluster']=ms.labels_
df['cluster']=ms.labels_

# Group by clusters and interpret
# ...
df.groupby('cluster').mean().sort_values(by=['Survived'], ascending=False)

```


## 5. Comparing Clustering Algorithms

### 5.X Comparison Table

Text

|Algorithm 	  |Principle   	        |Parameters   	|Pros             	|Cons              	|
|---	        |---	                |---	          |---	              |---	              |
|K-Means      | Text               	|`a` <br> `b`                  	|- a <br> - b                  	|- a <br> - b                  	|
