# -*- coding: utf-8 -*-

"""
@author: Loïc Herbelot
"""

import numpy as np
import operator
import graphes
import itertools

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
        
    def constructMatrix(self, docsList):
        docsWeights = {} # {docId: {stem representation}}
        docMap = {}# {docID: doc position in matrix (line)}
        stemMap = {} # {stem: position in matrix (column)}
        # We are going to store all the stems inside the docs
        Nwords = 0
        for docId in docsList:
            docWeight = self.getDocWeightsForDoc(docId)
            docsWeights[docId] = docWeight
#            print("docId:", docId)
#            print("docWeight:", docWeight)
            # Add the new word to the vocabulary
#            vocabSet.update(set(docWeight.keys()))
            for stem in docWeight:
                if stem not in stemMap:
                    stemMap[stem] = Nwords
                    Nwords += 1
            
        # Create the matrix
        Ndocs = len(docsList)
        X = np.zeros((Ndocs, Nwords))
        
        # Populate the matrix
        allDocId = list(docsWeights.keys())
        for i, docId in enumerate(allDocId):
            docRepr = docsWeights[docId]
            docMap[i] = docId
            for stem in docRepr:
                j = stemMap[stem]
                X[i, j] = docRepr[stem]
#        print(allDocId, stemMap)
#        print(X)
        return X

class BinaryWeighter(Weighter):
    def __init__(self, index):
        """param index: Index object"""
        super().__init__(index)

    def getDocWeightsForDoc(self, docId):
        stems = self.index.getTfsForDoc(docId)
        return stems

#    def getDocWeightsForStem(self, stem):
#        docFreq = self.index.getTfsForStem(stem)
#        return docFreq

    def getWeightsForQuery(self, query):
        return {stem:1 for stem in query.keys() }


class TfidfWeighter(Weighter):
    def __init__(self, index):
        """param index: Index object"""
        super().__init__(index)

    def getDocWeightsForDoc(self, docId):
        stems = self.index.getTfsForDoc(docId)
        return stems

#    def getDocWeightsForStem(self, stem):
#        docFreq = self.index.getTfsForStem(stem)
#        return docFreq

    def getWeightsForQuery(self, query):
        """
        :param query: dict representing the query: {stem:frequence}"""
        return {stem:self.index.computeIdf(stem) for stem in query.keys() }



class IRmodel():
    def __init__(self, index):
        self.index = index

    def score(self, query, docId):
        """Compute score between one query and one doc"""
        raise NotImplementedError("score() is an abstract method.")
        
    def getScores(self, query, normalized):
        """
        :param query: Dictionary of term frequencies        
        :return: a dict {docID: score}"""
        raise NotImplementedError("Abstract method.")

        
    def getRanking(self, query):
        """ Compute the of documents for the query
        :param query: dict of term frequencies
        :return: A list of tuples (doc id, score) sorted by score """
        scores = self.getScores(query)
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

    def getWeighter(self):
        return self.weighter
    
    def score(self, query, docId, normalized=True):
        queryWeights = self.weighter.getWeightsForQuery(query)
        docWeights = self.weighter.getDocWeightsForDoc(docId)
        norm = 1
        if normalized:
            norm = IRmodel.dictNorm(queryWeights) * IRmodel.dictNorm(docWeights)
        product = IRmodel.dictProduct(docWeights, queryWeights)
        return product / norm

    def getScores(self, query, normalized=True):
        """
        :param query: dict {stem: frequency}
        :return: a dict {docID: score}"""
        docsID = self.index.getDocsID()
        scores = {}
        queryWeights = self.weighter.getWeightsForQuery(query)
        for i in docsID:
            docWeights = self.weighter.getDocWeightsForDoc(i)
            norm = 1
            if normalized:
                norm = IRmodel.dictNorm(queryWeights) * IRmodel.dictNorm(docWeights)
            product = IRmodel.dictProduct(docWeights, queryWeights)

            scores[i] = product/norm

        return scores


class PRClustering(IRmodel):
    def __init__(self, index, baseModel, cluster, nDocs=100, 
                 clustersRank="rank", docRank="rank"):
        """
        :param baseModel: a vectoriel model, needs to have a getWeighter() method
        :param clustersRank: "rank", "querySim", "docsNbrAsc", "docsNbrDesc"
            The order of cluster retrieval
        :param docsRank: "rank", "centerSim"
            The order of documents retrieval inside a cluster
        """
        super().__init__(index)
        self.baseModel = baseModel
        self.cluster = cluster
        self.nDocs = nDocs
        self.clustersRank = clustersRank
        self.docRank = docRank
            
    def docQrySimilarity(self, docId, queryRepr):
        docRepr = self.baseModel.getWeighter().getDocWeightsForDoc(docId)
        return IRmodel.dictProduct(docRepr, queryRepr)
    
    def setNDocs(self, nDocs):
        self.nDocs = nDocs
    
    def getRanking(self, query, Nclusters=None, maxClusters=20, verbose=False):
        # Get ranking from base model, sorted list of (docsID, score)
        baseRanking = self.baseModel.getRanking(query)[:self.nDocs]
        docsScores = {docId:score for (docId, score) in baseRanking}
        docsList = [docId for (docId, score) in baseRanking]
        
        # Construct a matrix representation for the ranking
        weighter = self.baseModel.getWeighter()
        X = weighter.constructMatrix(docsList)
        
        # compute a clustering of these docs/vectors
        clustering = self.cluster.cluster(X, Nclusters=Nclusters, maxClusters=maxClusters)
        # put the original docs ID in the clustering instead of docs position
        for i, cluster in enumerate(clustering):
            clustering[i] = [docsList[d] for d in cluster]
        
        if verbose:
            print("\nClustering:", clustering)
        
        # return a ranking given a cluster/doc orderings
        clusterRankings = {"rank": lambda l:max([docsScores[d] for d in l]),
                "querySim": lambda l:max([self.docQrySimilarity(d, query) 
                            for d in l]),
                "docsNbrAsc": lambda l:len(l)}
        docRankings = {"rank": lambda d:docsScores[d],}
        
        clusterOrder = sorted(clustering, key=clusterRankings[self.clustersRank], reverse=True)
        if verbose:
            print("\nCluster order:", clusterOrder)
        
        for i, cluster in enumerate(clusterOrder):
            clusterOrder[i] = sorted(cluster, key=docRankings[self.docRank], reverse=True)
        
        if verbose:
            print("\nCluster ranking:", clusterOrder)
        
        zipped = itertools.zip_longest(*clusterOrder)
        ranking = []
        for tupl in zipped:
            ranking += [elt for elt in tupl if elt is not None]
        if verbose:
            print("\nRanking:", ranking)
        return ranking
        
class UnigramLanguage(IRmodel):
    def __init__(self, index, regularization=0.9):
        """ Create a new unigram model.
        :param index: The Index object that parsed all files
        :param txt_repr: a TextRepresenter object
        :param regularization: The regularization parameter to avoid having 
        a null probability because of an unkown word.
        :param args: Dictionary of {string:param}
        """
        super().__init__(index)
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
        for word, q_freq in query.items():
            if word not in self.model_corpus:
                continue
            in_log = (1-self.reg) * self.model_corpus[word]
            if word in doc_model:
                in_log += self.reg * doc_model[word]
#            print("in log:", in_log)
            if in_log == 0:
                print("Warning in_log = 0, docid=%s, word=%s" % (doc_id, word))
            score += q_freq * np.log(in_log)
#            else:
#                score = -np.infty
#                break
        return score
    
    def getScores(self, query, normalized=False):
        scores = {}
        norm = 0
        for doc in self.index.getDocsID():
            s = self.score(query, doc)   
            scores[doc] = s
            norm += s**2
        scores = {k:v/np.sqrt(norm) for k,v in scores.items()}
        
        return scores
        
class Okapi(IRmodel):
    def __init__(self, index, k=1, b=1):
        super().__init__(index)
        self.k = k
        self.b = b

    def get_params(self):
        return {"txt_repr":self.txt_repr,
                "k":self.k,
                "b":self.b}
        
    def score(self, query, doc_id, verbose=False):
        s = 0
        docStems = self.index.getTfsForDoc(doc_id)
        docLen = sum(docStems.values())        
        meanDocLen = self.index.getMeanDocLen()
        if verbose:
            print("Doc", doc_id, "len=", docLen, "meanLean=", meanDocLen)
        for word in query.keys():
            if word in docStems:
                tf = docStems[word]
            else:
                tf = 0
            numer = (self.k+1) * tf
            denom = self.k * ((1-self.b) + self.b*docLen/meanDocLen) + tf
            w = self.index.probIdf(word) * numer / denom
            if verbose:
                print("weight for "+word+":", w)
            s += w
#            if tf == 0:
#                print("%s not in doc" % word)
#            else:
#                print("word=%s, tf=%d, numer=%.3f, den=%.3f" % (word, tf, numer, denom))
        return s
        
    
    def getScores(self, query, normalized=False):
        scores = {}
        norm = 0
        for doc in self.index.getDocsID():
            s = self.score(query, doc)   
            scores[doc] = s
            norm += s**2
        scores = {k:v/np.sqrt(norm) for k,v in scores.items()}
        return scores
    
class PageRankModel(IRmodel):
    def __init__(self, index, baseModel, seedsNbr, parentsNbr):
        super().__init__(index)
        self.baseModel = baseModel
        self.seedsNbr = seedsNbr
        self.parentsNbr = parentsNbr
    
    def score(self, query, docId):
        baseRanking = self.baseModel.getRanking(query)
        seeds = [seed for (seed, score) in baseRanking[:self.seedsNbr]]
        if docId not in seeds:
            seeds.append(docId)
        pagerank = graphes.PageRank(self.index, seeds, self.parentsNbr)
        return pagerank.getScores(nIter=100, teleportProba=0.1)[docId]

    
    def getScores(self, query, normalized=True):
        baseRanking = self.baseModel.getRanking(query)
        seeds = [seed for (seed, score) in baseRanking[:self.seedsNbr]]
        pagerank = graphes.PageRank(self.index, seeds, self.parentsNbr)
        return pagerank.getScores(nIter=100, teleportProba=0.1)
        
class HitsModel(IRmodel):
    def __init__(self, index, baseModel, seedsNbr, parentsNbr):
        super().__init__(index)
        self.baseModel = baseModel
        self.seedsNbr = seedsNbr
        self.parentsNbr = parentsNbr
    
    
    def score(self, query, docId):
        baseRanking = self.baseModel.getRanking(query)
        seeds = [seed for (seed, score) in baseRanking[:self.seedsNbr]]
        if docId not in seeds:
            seeds.append(docId)
        hits = graphes.HITS(self.index, seeds, self.parentsNbr)
#        print("call getScores")
        return hits.getScores(nIter=10)[docId]

    
    def getScores(self, query, normalized=True):
#        print("retrieve base ranking")
        baseRanking = self.baseModel.getRanking(query)
        seeds = [seed for (seed, score) in baseRanking[:self.seedsNbr]]
#        print("retrieved base ranking, call HITS")
        hits = graphes.HITS(self.index, seeds, self.parentsNbr)
#        print("call getScores")
        return hits.getScores(nIter=10)
        