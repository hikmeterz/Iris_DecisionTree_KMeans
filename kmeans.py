import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class KMeansClusterClassifier:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = None
        self.labels = None
    
    def fit(self, X, y=None, max_iters=100):
        random_indices = self.random_choice(len(X), self.n_clusters)
        self.centroids = [X[i] for i in random_indices]

        for _ in range(max_iters):
            distances = self.calculate_distances(X)
            self.labels = self.argmin(distances)

            new_centroids = [self.calculate_mean(X, self.labels, i) for i in range(self.n_clusters)]

            if self.has_converged(new_centroids):
                break

            self.centroids = new_centroids
    
    def predict(self, X):
        distances = self.calculate_distances(X)
        return self.argmin(distances)

    def calculate_distances(self, X):
        distances = []
        for centroid in self.centroids:
            centroid_distances = [self.euclidean_distance(centroid, x) for x in X]
            distances.append(centroid_distances)
        return distances

    def argmin(self, distances):
        return [min(range(self.n_clusters), key=lambda i: distances[i][j]) for j in range(len(distances[0]))]

    def calculate_mean(self, X, labels, cluster_idx):
        cluster_points = [X[i] for i in range(len(X)) if labels[i] == cluster_idx]
        if len(cluster_points) > 0:
            return [sum(col) / len(col) for col in zip(*cluster_points)]
        return self.centroids[cluster_idx]

    def has_converged(self, new_centroids, tol=1e-4):
        if self.centroids is not None:
            return all(self.euclidean_distance(new_centroids[i], self.centroids[i]) < tol for i in range(self.n_clusters))
        return False

    def random_choice(self, n, k):
        import random
        indices = list(range(n))
        random.shuffle(indices)
        return indices[:k]

    def euclidean_distance(self, a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def elbow_method(X, max_clusters, max_iters=100):
    ssd = []
    for k in range(1, max_clusters+1):
        kmeans = KMeansClusterClassifier(n_clusters=k)
        kmeans.fit(X, max_iters=max_iters)
        distances = np.sqrt(((X - kmeans.centroids[:, np.newaxis])**2).sum(axis=2))
        ssd.append(np.sum(np.min(distances, axis=0)**2))
    return ssd

