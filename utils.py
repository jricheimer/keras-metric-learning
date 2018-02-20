"""Handy functions to have around.
"""
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

def recall_at_k(embeddings, labels, k=1, metric='euclidean'):
    if (k >= embeddings.shape[0]):
        raise ValueError('k cannot exceed the number of embeddings tested')

    closest_k = squareform(pdist(embeddings, metric=metric))
    np.fill_diagonal(closest_k, sys.float_info.max)
    closest_k = closest_k.argsort(axis=-1)[:,:k]
    recall = np.any(labels[closest_k] == labels[:,np.newaxis], axis=-1).mean()

    return recall

def nmi(embeddings, labels, metric='euclidean'):
    num_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    return normalized_mutual_info_score(labels, kmeans.labels_)