# -*- coding: utf-8 -*-

"""
@author: Loïc Herbelot
"""

import numpy as np
import operator

class Weighter():
    def __init__(self, index):
        """param index: Index object"""
        self.index = index

    def getDocWeightsForDoc(self, docId):
        raise NotImplementedError("Abstract method.")

    def getDocWeightsForStem(self, stem):
        raise NotImplementedError("Abstract method.")

    def getWeightsForQuery(self, query):
        raise NotImplementedError("Abstract method.")

class BinaryWeighter(Weighter):
    def __init__(self, index):
        """param index: Index object"""
        super().__init__(index)

    def getDocWeightsForDoc(self, docId):
        stems = self.index.getTfsForDoc(docId)
        return stems

    def getDocWeightsForStem(self, stem):
        docFreq = self.index.getTfsForStem(stem)
        return docFreq

    def getWeightsForQuery(self, query):
        return {stem:1 for stem in query.keys() }


class TfidfWeighter(Weighter):
    def __init__(self, index):
        """param index: Index object"""
        super().__init__(index)

    def getDocWeightsForDoc(self, docId):
        stems = self.index.getTfsForDoc(docId)
        return stems

    def getDocWeightsForStem(self, stem):
        docFreq = self.index.getTfsForStem(stem)
        return docFreq

    def getWeightsForQuery(self, query):
        """
        :param query: dict representing the query: {stem:frequence}"""
        docsID = self.index.getDocsID()
        N = len(docsID)
        
        def idf(stem):
            n = len(self.index.getTfsForStem(stem))
            if n == 0:
                return 0
            else :
                return np.log(N/n)
        return {stem:idf(stem) for stem in query.keys() }



class IRmodel():
    def __init__(self, index):
        self.index = index

    def getScores(self, query, normalized):
        """
        :param query: Dictionary of term frequencies        
        :return: a dict {docID: score}"""
        raise NotImplementedError("Abstract method")
    
    def getRanking(self, query):
        """ Compute the of documents for the query
        :return: A list of tuples (doc id, score) sorted by score """
        scores = self.getScores(query, True)
        return sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

    def dictProduct(a,b):
        s = sum([a[i]*b[i] for i in a.keys() if i in b.keys()])
        return s

    def dictNorm(a):
        return np.linalg.norm(list(a.values()))

class Vectoriel(IRmodel):
    def __init__(self, index, weighter):
        super().__init__(index)
        self.weighter = weighter
        self.norms = {}


    def getScores(self, query, normalized=True):
        """
        :param query: dict {stem: frequence}
        :return: a dict {docID: score}"""
        docsID = self.index.getDocsID()
        scores = {}
        queryWeights = self.weighter.getWeightsForQuery(query)
        for i in docsID:
            docWeights = self.weighter.getDocWeightsForDoc(i)
            score = IRmodel.dictProduct(docWeights,
                                        queryWeights)
            norm = 1
            if normalized:
                if i in self.norms:
                    norm = self.norms[i]
                else:
                    norm = IRmodel.dictNorm(docWeights)
                    self.norms[i] = norm
                norm *= IRmodel.dictNorm(queryWeights)

                scores[i] = score/norm
        return scores



class LanguageModel(IRmodel):
    def __init__(self, index, txt_repr, regularization):
        """ Create a new unigram model.
        :param index: The Index object that parsed all files
        :param txt_repr: a TextRepresenter object
        :param regularization: The regularization parameter to avoid having 
        a null probability because of an unkown word.
        """
        super().__init__(index)
        self.txt_repr = txt_repr
        self.reg = regularization
        self.model_corpus = None
        # TODO: cache document models with a queue
       # self.models = {}
    
    def get_model_corpus(self):
        """ Return the unigram model for the entire collection, and create
        it if the model is not already computed.
        """ 
        if self.model_corpus is None:
            self.model_corpus = {}
            collec_length = 0
            for stem in self.index.getStems():
                term_freq_dic = self.index.getTfsForStem(stem)
                freq = sum(term_freq_dic.values())
                collec_length += freq
                self.model_corpus[stem] = freq
                
            for stem in self.model_corpus:
                self.model_corpus[stem] /= collec_length
            
        return self.model_corpus
    
    def score(self, query, doc_id):
        """ Compute the likelihood of a query inside a document. 
        :param query: The Query object
        :param doc: int or string, the ID of the document 
        :return: float, the likelihood a the query inside the document.
        May be -inf if a word of the query isn't found in the entire collection.
        """
        doc_model = self.index.getTfsForDoc(doc_id)
        self.get_model_corpus()
        doc_len = sum(doc_model.values())
        doc_model = {s:tf/doc_len for s,tf in doc_model.items()}
        score = 0
        query_repr = self.txt_repr.getTextRepresentation(query.getText())
        for word, q_freq in query_repr.items():
            if word in doc_model:
                in_log = self.reg * doc_model[word] + (1-self.reg) * self.model_corpus[word]
#                print("in log:", in_log)
                score += q_freq * np.log(in_log)
            else:
                score = -np.infty
                break
        return score
        
    