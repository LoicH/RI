#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import TextRepresenter

class IRList():
    """ Contains a query and the scores found for this query """
    def __init__(self, query, scores):
        """ Create new IRList
        :param query: The Query object
        :param scores: a list of tuples [(docId, score)]"""
        self.query = query
        self.scores = scores # List [(docId, score)]

    def getQuery(self):
        """ Return the Query object"""
        return self.query

    def getScores(self):
        return self.scores


class EvalMeasure:
    """ The abstract class for measure methods """
    def __init__(self, irlist):
        self.irlist = irlist

    def getRelevantResults(self):
        """ Return the <100 relevant results for the query in the
        IRList object"""
        # The relevant results of the query are the ones that have
        # a score higher than mean:
        #mean = np.mean([score for (docId, score) in self.irlist.getScores()])
        sortRes = sorted(self.irlist.getScores(), key=lambda tupl:tupl[1], reverse=True)
        relevantResults = [docId for (docId, score) in sortRes
                          #if score > mean
                        ]
        return relevantResults[:]

    def eval(self):
        raise NotImplementedError("Abstract method")


class PrecisionRecallMeasure(EvalMeasure):

    def __init__(self, irlist):
        super().__init__(irlist)

    def eval(self, verbose=False, nbLevel=11):
        """ Compute the recall and precision for an IRList
        :param verbose: bool, whether or not to display messages
        :param nbLevel: int, optional (default is 11). The number of 
        (precision, recall) values to compute
        :return: A sorted list of (recall, precision)"""
        rec_prec = {}
        # Truely relevant results for the query:
        trueRels = list(self.irlist.getQuery().getRelevants().keys())
        # Results we found for the query:
        results = super().getRelevantResults()
        trueRelsLen = len(trueRels)        
        
        if verbose:
            print("This query has %d relevant results" 
                % len(trueRels))
            # print("Scores for this query:", self.irlist.getScores())
            print("   i |found| precision | recall")
        i = 1
        relevantFound = 0.
        while i<len(results) and relevantFound<trueRelsLen:
            # Number of results we found that are really relevant:
            if int(results[i-1]) in trueRels:
                relevantFound += 1

            prec = relevantFound/i
            rec = relevantFound/trueRelsLen
            
            if verbose and int(results[i-1]) in trueRels:
                print("%5d|%4d | %5f  |%5f" % (i, relevantFound, prec, rec))
            
            
            if rec not in rec_prec.keys():
                rec_prec[rec] = prec
            elif prec > rec_prec[rec]:
                rec_prec[rec] = prec
            i +=1

        results = []
        keys = list(rec_prec.keys())
        for recall in np.linspace(0, 1, nbLevel):
            # Find the closest point to recall
            idx = np.argmin(list(map(lambda n:np.abs(n-recall), keys)))
            # print("Closest recall to %f is %f" % (recall, keys[idx]))
            # print("idx=%d, keys[idx]=%f" %(idx, keys[idx]))
            results.append((keys[idx], rec_prec[keys[idx]]))

        # print(results)
        return results


class AveragePrecision(EvalMeasure):

    def __init__(self, irlist):
        super().__init__(irlist)

    def eval(self, verbose=False):
        """ Compute the performance of a model.
        :return: The average precision at different ranks"""
        s = 0 # The sum of precisions. 
        # Truely relevant results for the query:
        trueRels = list(self.irlist.getQuery().getRelevants().keys())
        # Results we found for the query:
        results = super().getRelevantResults()

        if verbose:
            print("This query has %d relevant results" 
                % len(trueRels))
            # print("Scores for this query:", self.irlist.getScores())
            print("   i |found| precision")
            
        i = 1
        relevantFound = 0
        while i<len(results) and relevantFound<len(trueRels):
            prec = 0
            if int(results[i-1]) in trueRels:
                # Number of results we found that are really relevant:
                relevantFound = len(np.intersect1d(results[:i], trueRels))
                prec = relevantFound/i
                s += prec
                if verbose:
                    print("%5d|%4d | %5f" % (i, relevantFound, prec))
#            elif verbose:
#                print(results[i-1] + " not in trueRels")
            i += 1
        return s/len(trueRels)

class PrecisionNDocuments(EvalMeasure):
    def __init__(self, irlist):
        super().__init__(irlist)

    def eval(self, n, verbose=False):
        """ Compute the performance of a model.
                :return: The precision at the n'th rank"""
        # Truely relevant results for the query:
        trueRels = list(self.irlist.getQuery().getRelevants().keys())
        # Results we found for the query:
        results = super().getRelevantResults()
        precision = 0
        if len(trueRels) < n:
            for result in results[0:len(trueRels)]:
                if result in trueRels:
                    precision += 1
            return precision/len(trueRels)
        else:
            for result in results[0:n]:
                if result in trueRels[0:n]:
                    precision += 1
            return precision/n
# TODO add subthemes in the returned elements of 'getRelevantResults' to be able to compute the Cluster Recall
class ClusterRecallNDocuments(EvalMeasure):
    def __init__(self, irlist):
        super().__init__(irlist)

    def eval(self, n, verbose=False):
        """ Compute the performance of a model.
                :return: The cluster recall at the n'th rank"""
        # Truely relevant results for the query:
        trueRels = self.irlist.getQuery().getRelevants()
        # Results we found for the query:
        results = super().getRelevantResults()
        # Unique subthemes in the real relevant results
        subtheme_set = set()
        for elt in list(trueRels.values()):
            subtheme_set.add(elt["subtheme"])
        real_nb_cluster = len(subtheme_set)
        return None
        
class EvalIRModel():
    def __init__(self, queries, irmodels, measures, 
                 stemmer=TextRepresenter.PorterStemmer()):
        """
        :param queries: List of Query objects
        :param irmodels: dictionary of {name:IRmodel object}
        :param evals: dictionary of {name:EvalMeasure class}"""
        self.queries = queries
        self.irmodels = irmodels
        self.measures = measures
        self.stemmer = stemmer

    def eval(self, verbose=0):
        """ Compares different types of IR models and evaluation methods.
        :return: A dictionary of {(s1, s2):(n1, n2)}
                where 's1' is the name of the IR model
                      's2' is the name of the evaluation method
                      'n1' is the mean of scores over all queries
                      'n2' is the standard deviation
        """
        all_query_scores = {}
        results = {}
        for irmodel_name, irmodel in self.irmodels.items():
            if verbose:
                print("IRModel:", irmodel_name)
            for q in self.queries:
                q_scores = irmodel.getScores(self.stemmer.
                            getTextRepresentation(q.getText()))
                all_query_scores[q] = (list(q_scores.items()))
                
            for measure_name, measure_class in self.measures.items():
                if verbose:
                    print("Measure:", measure_name)
                eval_scores = []
                for query, scores_list in all_query_scores.items():
                    if verbose > 1:
                        print(20 * '-')
                        print(query)
                        print(scores_list[:10])
                    measure = measure_class(IRList(query, scores_list))
                    tmp_score = measure.eval(verbose=(verbose==2))
                    eval_scores.append(tmp_score)
                if verbose:
                    print((np.mean(eval_scores), np.std(eval_scores)))
                results[(irmodel_name, measure_name)] = (np.mean(eval_scores), np.std(eval_scores))
        return results
