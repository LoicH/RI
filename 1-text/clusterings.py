# -*- coding: utf-8 -*-
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MeanShift
import numpy as np

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
    def cluster(self, X, Nclusters=None, maxClusters=20, criterion="bic", verbose=False):
        if Nclusters is None:
            # Find the best cluster
            maxClusters = maxClusters
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
                if verbose:
                    print("For n_clusters =", Nclusters,
                      "The average silhouette_score is :", silScore)
        else:
            clusterer = KMeans(n_clusters=Nclusters)
            bestClusterNb = Nclusters
            bestClusterLabels = clusterer.fit_predict(X)

            
        # Change the clustering format:
        if verbose:
            print("N_clusters = ", bestClusterNb)
        clustering = []
        docsId = range(X.shape[0])
        for i in range(bestClusterNb):
            clustering.append([docId for docId in docsId if bestClusterLabels[docId] == i])
            
        return clustering
        
    
class MeanShiftClustering(Clustering):
    def cluster(self, X, Nclusters=None, maxClusters=None, verbose=False):
        meanshift = MeanShift(bandwidth=1.5)
        meanshift.fit(X)
        labels = meanshift.labels_
        clustering = []
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        docsId = range(X.shape[0])
        for i in range(n_clusters_):
            clustering.append([docId for docId in docsId if labels[docId] == i])
            
        return clustering
