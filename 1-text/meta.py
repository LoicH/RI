# -*- coding: utf-8 -*-

"""
@author: Sébastien P
"""
from modeles import IRmodel
import graphes
import random
import numpy as np
import operator

class Featurer():
    """"""
    def __init__(self, index):
        self.index = index
        # A dictionnary of {docId : Score} for a given model
        self.features = {}

    def getFeatures(self, docId, query):
        """
        :param query: dict of {stem: frequency}
        """
        raise NotImplementedError ("Abstract method")
        
class DocLenFeaturer(Featurer):
    def __init__(self, index):
        super().__init__(index)
    def getFeatures(self, docId, query):
        return 1/self.index.getDocsLen(docId)
    
class QueryLenFeaturer(Featurer):
    def __init__(self, index):
        super().__init__(index)
    def getFeatures(self, docId, query):
        return 1/sum(query.values())
    
class PageRankFeaturer(Featurer):   
    """ Doesn't depend on a query"""
    def __init__(self, index):
        super().__init__(index)
        print("Init pagerank")
        self.pagerankScores = {}
        pagerank = graphes.PageRank(self.index, 
                                    seeds=self.index.getDocsID(), 
                                    prevNeighbours=0)
        self.pagerankScores = pagerank.getScores()
        # Already scaled
#        norm = IRmodel.dictNorm(self.pagerankScores)
#        for key, value in self.pagerankScores.items():
#            self.pagerankScores[key] = value/norm
        print("Done.")
        
        
    def getFeatures(self, docId, query):
        return self.pagerankScores[docId]


class FeaturerModel(Featurer):
    """"""
    def __init__(self, index, model):
        '''
        :param index: the reference to an index object
        :param model: the reference to a RI model (TfIdf weigther, Okapi, PageRank)
        '''
        super().__init__(index)
        self.model = model

    def getFeatures(self, idDoc, query):
        return self.model.score(query, idDoc)

class FeaturerList(Featurer):
    """Aggregates diverse features"""
    def __init__(self, index, featurers):
        super().__init__(index)
        self.featurers = featurers
        print("Featurers:", featurers)

    def getFeatures(self, idDoc, query):
        features = []
        for featurer in self.featurers:
            features.append(featurer.getFeatures(idDoc, query))
        return np.array(features)

class MetaModel(IRmodel):
    def __init__(self, index, featurer_list, stemmer):
        """
        :param featurer_list: FeaturerList object
        """
        super().__init__(index)
        self.featurer_list = featurer_list
        self.theta = None
        self.stemmer = stemmer

        
    def train(self, queries, max_iter, lr, regul_coef):
        """
        :param queries: List of Query object
        """
        # Init param theta
        randQueries = np.random.choice(queries, size=max_iter)
        for randQry in randQueries:
            relevantDocs = list(randQry.getRelevants().keys())
            irrelevantDocs = list(set(self.index.getDocsID()) - set(relevantDocs))
            # Get random documents 
            relDoc = random.choice(relevantDocs)
            not_relDoc = random.choice(irrelevantDocs)

            # Retrieve features for the query
            qryRepr = self.stemmer.getTextRepresentation(randQry.getText())
            relFeat = self.featurer_list.getFeatures(relDoc, qryRepr)
            not_relFeat = self.featurer_list.getFeatures(not_relDoc, qryRepr)

            # Init theta if not already done:
            if self.theta is None:
                self.theta = np.zeros_like(relFeat)
            
            # Update theta
            if 1 - self.score(qryRepr, relDoc) + self.score(qryRepr, not_relDoc) > 0:
                self.theta += lr*(relFeat - not_relFeat)
                # Scale theta:
                self.theta *= (1-2*lr*regul_coef)

    def score(self, query, docId):
        """ Return f_theta(d, q)
        """
        if self.theta is None:
            raise ValueError("Theta not initialized")
        else:
            xdq = self.featurer_list.getFeatures(docId, query)
            return self.theta.dot(xdq)
        
    def getScores(self, query, normalized=False):
        allDocs = self.index.getDocsID()
        scores = {}
        for docId in allDocs:
            if self.theta is None:
                raise ValueError("Theta not initialized")
            xdq = self.featurer_list.getFeatures(docId, query)
            scores[docId] = self.theta.dot(xdq)
        return scores

        
