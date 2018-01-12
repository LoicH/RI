# -*- coding: utf-8 -*-

"""
@author: Sébastien P
"""
from .modeles import IRmodel
import random
import numpy as np
import operator

class Featurer():
    """"""
    def __init__(self, index):
        '''
        :param
        '''
        self.index = index
        # A dictionnary of {docId : Score} for a given model
        self.features = {}

    def getFeatures(self):
        raise NotImplementedError ("Abstract method")

class FeaturerModel(Featurer):
    """"""
    def __init__(self, index, model):
        '''
        :param index: the reference to an index object
        :param model: the reference to a RI model (TfIdf weigther, Okapi, PageRank)
        '''
        super.__init__(index)
        self.model = model

    def getFeatures(self, idDoc, query):
        '''

        :param idDoc: the document id
        :param query: a query object
        :return: the scores of each document in the index for the query object given a specific model
        '''
        # Get the dictionnary {docId : Score}
        result = self.model.getScores(query)
        self.features = result
        return result

class FeaturerList(Featurer):
    def __init__(self, index, model_list):
        super.__init__(index)
        # A dictionnary of {model : {docId : score}}
        self.list_features = {}
        self.model_list = model_list

    def getFeatures(self, query):
        '''
        :param model_list: list of models
        :param idDoc: the document id
        :param query: a query object
        :return: update a dictionnary {model : features}, features = {docId : Score}
        '''
        for model in self.model_list:
            self.list_features[model] = model.getScores(query)

class MetaModel(IRmodel):
    def __init__(self, index, featurer_list):
        super.__init__(index)
        self.featurer_list = featurer_list
        self.theta = {}

    def gradientDescent(self, max_iter, lr, regul_coef):
        # Init param theta
        for model in self.featurer_list.list_features.keys():
            self.theta[model] = np.random.uniform(low=0.0, high=1.0)

        for iter in range(max_iter):
            # Randomly choosing a query
            # TODO check how to use only a subset of queries for training
            queryChosen = np.random.randint(1, 50, size=50)

            # Get relevant document d
            relevant_doc = random.choice.queryChosen.getRelevants().key()
            # Get not relevant document d'
            not_relevant_doc = random.choice.index.getDocsID()

            # While selected document is relevant select another one
            while not_relevant_doc in list(queryChosen.getRelevants().key()):
                not_relevant_doc = random.choice.index.getDocsID()

            # Retrieve not weigthed scores for the query
            self.featurer_list.getFeatures(queryChosen)
            # Retrieve weighted scores in a dict for each model
            score = self.getScores(queryChosen)
            # Param update
            for theta in self.theta.keys():
                if 1 - score[theta][relevant_doc] + score[theta][not_relevant_doc] > 0:
                    # Compute gradient
                    diff = self.featurers_list.list_features[theta][relevant_doc] - self.featurers_list.list_features[theta][not_relevant_doc]
                    theta[theta] = theta[theta] + lr * diff
                # Regularization
                theta[theta] = (1 - 2 * lr * regul_coef) * theta[theta]

    def getScores(self, query, normalized=True):
        score = {}
        # Init an empty dict
        # and fill it with the dot product of (theta,score) for all models
        for model in self.featurer_list.list_features.keys():
            score[model] = self.theta[model] * self.featurers_list.list_features[model]

        if normalized:
            for model in self.featurer_list.list_features.keys():
                s = np.sum(list(score[model].values()))
                for docId, score in score[model].items():
                    score[model][docId] = score/s

        return score

    def getRanking(self, query):
        final_score = []
        scores = self.getScores(query, normalized=True)
        for docId in scores[self.featurer_list.model_list[0]].key():
            s = 0
            for model in self.featurer_list.model_list:
                s += scores[model][docId]
            final_score.append((docId,s))

        return sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

