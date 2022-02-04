import numpy as np
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def center_data(self, data):
        mean = np.sum(data, axis=0)/data.shape[0]
        centered_data = data - mean
        return centered_data

    def initialize_centroids(self, centered_data):
        sz = center_data.shape
        rand_intX = np.random.choice(sz[2], self.num_clusters, replace = False)
        rand_intY = np.random.choice(sz[1], self.num_clusters, replace = False)
        centroids = centered_data[:,rand_intY, rand_intX]
        print(centroids.shape)
        return centroids

    def assign_labels(self, centroids, centered_data):
        for i in range(self.num_clusters):
            diff = centered_data - centroids[i]
        return

    def fit(self, data):
        data = np.array(data)
        centered_data = self.center_data(data)
        print("from kmeans ",centered_data.shape)
        centroids = self.initialize_centroids(centered_data)
        self.assign_labels(centroids, data)
        # plt.imshow(centered_data[0])
        # plt.show()

        return
