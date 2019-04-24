"""Handy functions to have around.
"""
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from kml_data_utils import pair_generator, organize_by_class

def recall_at_k(embeddings, labels, k=1, metric='euclidean'):
    """Computes the Recall@K metric

    # Arguments
        embeddings: array of shape (num_samples, num_dims)
        labels: integer class labels with shape (num_samples,)
        k: Number of closest embeddings to consider
        metric: distance metric for the embedding space. Defaults to 'euclidean'.
    """
    if (k >= embeddings.shape[0]):
        raise ValueError('k cannot exceed the number of embeddings tested')

    closest_k = squareform(pdist(embeddings, metric=metric))
    np.fill_diagonal(closest_k, sys.float_info.max)
    closest_k = closest_k.argsort(axis=-1)[:, :k]
    labels = np.squeeze(labels)
    recall = np.any(labels[closest_k] == labels[:, np.newaxis], axis=-1).mean()

    return recall

def nmi(embeddings, labels, metric='euclidean'):
    """Computes the Normalized Mutual Information score

    # Arguments
        embeddings: array of shape (num_samples, num_dims)
        labels: integer class labels with shape (num_samples,)
        metric: distance metric for the embedding space. Defaults to 'euclidean'.
    """
    num_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    return normalized_mutual_info_score(labels, kmeans.labels_)

def plot_distance_distributions(embeddings, labels, num_pairs=10000, num_bins=100):
    negative_distances = []
    positive_distances = []
    pos_gen = pair_generator(organize_by_class(embeddings, labels), all_similar=True)
    while (len(positive_distances) < num_pairs/2):
        pairs = pos_gen.next()
        
        