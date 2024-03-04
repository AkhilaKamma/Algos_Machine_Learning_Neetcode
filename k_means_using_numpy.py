import numpy as np

class KMeans:
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(range(len(X)), self.k, replace=False)]
        
        for i in range(self.max_iterations):
            # Assign each data point to the nearest centroid
            clusters = [[] for _ in range(self.k)]
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                cluster = np.argmin(distances)
                clusters[cluster].append(x)

            # Recalculate centroids
            prev_centroids = self.centroids
            self.centroids = []
            for cluster in clusters:
                if cluster:
                    self.centroids.append(np.mean(cluster, axis=0))
                else:
                    self.centroids.append(prev_centroids[np.random.choice(range(self.k))])

            # Check for convergence
            if np.allclose(prev_centroids, self.centroids):
                break

    def predict(self, X):
        distances = [np.linalg.norm(X - c, axis=1) for c in self.centroids]
        return np.argmin(distances, axis=0)
