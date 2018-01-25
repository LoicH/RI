# -*- coding: utf-8 -*-
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


class Clustering():
    def __init__(self):
        pass
    
    def cluster(self, X):
        """ Compute the clustering of a ranking
        :param X: The matrix of all docs
        :return: a clustering, index of the cluster each sample belongs to.

        """
        raise NotImplementedError("Abstract method")
        
class KMeansClustering(Clustering):
    def cluster(self, X, Nclusters=None, criterion="bic"):
        if Nclusters is None:
            # Find the best cluster
            maxClusters = 20
            clustersRange = range(2, maxClusters)
            bestClusterLabels = None
            bestScore = -1
            bestClusterNb = -1
            for Nclusters in clustersRange:
                clusterer = KMeans(n_clusters=Nclusters)
                clusterLabels = clusterer.fit_predict(X)
                silScore = silhouette_score(X, clusterLabels)
                if silScore > bestScore:
                    bestClusterLabels = clusterLabels
                    bestScore = silScore
                    bestClusterNb = Nclusters
                print("For n_clusters =", Nclusters,
                      "The average silhouette_score is :", silScore)
        else:
            clusterer = KMeans(n_clusters=Nclusters)
            bestClusterNb = Nclusters
            bestClusterLabels = clusterer.fit_predict(X)

            
        # Change the clustering format:
        clustering = []
        docsId = range(X.shape[0])
        for i in range(bestClusterNb):
            clustering.append([docId for docId in docsId if bestClusterLabels[docId] == i])
            
        return clustering
        