import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA

# Load the Digits dataset
digits = datasets.load_digits()
data = digits.data  # Featurespip
target = digits.target  # Target labels (not used for clustering)

# Reduce dimensionality for visualization (optional)
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# Define the number of clusters (you can change this)
num_clusters = 10

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(data)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering of Digits Dataset')
plt.legend()
plt.show()

# Analyze the results as needed