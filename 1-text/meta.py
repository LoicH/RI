# -*- coding: utf-8 -*-

"""
@author: Sébastien P
"""
from .modeles import IRmodel

class Featurer():
    """"""
    def __init__(self, index):
        '''
        :param
        '''
        self.index = index
        self.features

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
        # TODO check if one has to use getRanking or getScores here
        result = self.model.getRanking(query)
        self.features = result
        return result

class FeaturerList():
    def __init__(self):
        self.list_features = {}

    def getScores(self,featurer_list, idDoc, query):
        '''

        :param featurer_list: list of featurer objects
        :param idDoc: the document id
        :param query: a query object
        :return: return a dictionnary {featurer : score list of each doc for a given query }
        '''
        for featurer in featurer_list:
            self.list_features[featurer] = featurer.getFeatures(idDoc, query)

class MetaModel(IRmodel):
    def __init__(self, index, featurers_list):
        super.__init__(index)
        self.featurers_list = featurers_list
        self.theta

    def getScores(self, query):


