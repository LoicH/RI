# -*- coding: utf-8 -*-
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


class Clustering():
    def __init__(self):
        pass
    
    def cluster(self, query, ranking):
        """ Compute the clustering of a ranking
        :param query:
        :param ranking:
        :return: a clustering, index of the cluster each sample belongs to.

        """
        raise NotImplementedError("Abstract method")
        
class KMeansClustering(Clustering):
    def cluster(self, X, criterion="bic"):
        
        # Retrieve ranking for this query
        max_clusters = 20
        clusters_range = range(2, max_clusters)
        # Test different numbers of clusters
        best_cluster_labels = None
        best_score = -1
        for n_clusters in clusters_range:
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)
            sil_score = silhouette_score(X, cluster_labels)
            if sil_score > best_score:
                best_cluster_labels = cluster_labels
                best_score = sil_score
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", sil_score)
        return best_cluster_labels
        