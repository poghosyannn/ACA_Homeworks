import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


centers = [[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]]
X_blobs, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=0.3,random_state=0)

class Spectral_Clustering:
    
    def __init__(self, k):
        self.k = k
    
    def fit(self, X):
        distances = euclidean_distances(X)  
        affinity_matrix = np.exp(-distances ** 2)

        W = normalize(affinity_matrix, norm='l1', axis=1)
        D = np.diag(np.sum(W, axis=1))
        laplacian_matrix = D - W

        eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        indices_sorted = np.argsort(eigenvalues)
        eigenvalues_sorted = eigenvalues[indices_sorted]
        eigenvectors_sorted = eigenvectors[:, indices_sorted]
        
        eigenvectors_sorted = eigenvectors_sorted[:, 1:self.k+1]
        
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(eigenvectors_sorted)
        
        self.labels = kmeans.labels_
        
    def predict(self, X):
        return self.labels
    
spectral = Spectral_Clustering(k=5)
spectral.fit(X_blobs)
y_pred = spectral.predict(X_blobs)
plt.scatter(X_blobs[:,0], X_blobs[:,1], c=y_pred)
plt.show()
