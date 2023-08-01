import numpy as np 

# Step 1: Initialize the k-means cluster
def initialize_clusters(X, k):
    clusters = []
    for i in range(k):
        cluster = {
            'centroid': X[i],            
            'points': []
        }
        clusters.append(cluster)
    return clusters

# Step 2: Assign each data point to the closest cluster
def assign_points_to_clusters(X, clusters):
    for x in X:
        distances = [np.linalg.norm(x - cluster['centroid']) for cluster in clusters]
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx]['points'].append(x)
    
    return clusters

# Step 3: Compute new cluster centroids
def compute_cluster_centroids(clusters):
    for cluster in clusters:
        cluster['centroid'] = np.mean(cluster['points'], axis=0)
    
    return clusters

# Step 4: Repeat steps 2 and 3 until convergence
def kmeans(X, centroids, k):
    clusters = initialize_clusters(centroids, k)
    
    # converged = False
    # while not converged:
    #     clusters = assign_points_to_clusters(X, clusters)
    #     old_clusters = clusters
    #     clusters = compute_cluster_centroids(clusters)
    #     converged = np.array_equal(old_clusters, clusters)
     
    return assign_points_to_clusters(X, clusters)