#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import query
import numpy as np

class IRList():
    """ Contains a query and the scores found for this query """
    def __init__(self, query, scores):
        self.query = query
        self.scores = scores # List [(docId, score)]

    def getQuery(self):
        return self.query

    def getScores(self):
        return self.scores


class EvalMeasure:
    """ The abstract class for measure methods """
    def __init__(self, irlist):
        self.irlist = irlist

    def getRelevantResults(self):
        # The relevant results of the query are the ones that have
        # a score higher than mean:
        mean = np.mean([score for (docId, score) in self.irlist.getScores()])
        relevantResults = [docId for (docId, score) in self.irlist.getScores()
                          if score > mean]
        return relevantResults

    def eval(self):
        raise NotImplementedError("Abstract method")


class PrecisionRecallMeasure(EvalMeasure):

    def __init__(self, irlist):
        super().__init__(irlist)

    def eval(self, verbose=False, size=11):
        """ 
        :return: A sorted list of (recall, precision)"""
        rec_prec = {}
        # Truely relevant results for the query:
        trueRels = list(self.irlist.getQuery().getRelevants().keys())
        # Results we found for the query:
        results = super().getRelevantResults()

        if verbose:
            print("This query has %d relevant results" 
                % len(trueRels))
            # print("Scores for this query:", self.irlist.getScores())
            print("   i |found| precision | recall")
        for i in range(1, len(results)):
            # Number of results we found that are really relevant:
            relevantFound = len(np.intersect1d(results[:i], trueRels))
            prec = relevantFound/i
            rec = relevantFound/len(trueRels)
            if verbose:
                print("%5d|%4d | %5f  |%5f" % (i, relevantFound, prec, rec))

            if rec not in rec_prec.keys():
                rec_prec[rec] = prec
            elif prec > rec_prec[rec]:
                rec_prec[rec] = prec

        results = []
        keys = list(rec_prec.keys())
        for recall in np.linspace(0, 1, size):
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

    def eval(self, verbose=False, step=1):
        """
        :return: The average precision at different ranks"""
        mean = 0
        # Truely relevant results for the query:
        trueRels = list(self.irlist.getQuery().getRelevants().keys())
        # Results we found for the query:
        results = super().getRelevantResults()

        if verbose:
            print("This query has %d relevant results" 
                % len(trueRels))
            # print("Scores for this query:", self.irlist.getScores())
            print("   i |found| precision")
        for i in range(1, len(results), step):
            # Number of results we found that are really relevant:
            relevantFound = len(np.intersect1d(results[:i], trueRels))
            # print("Doc at rank %d is %s." % (i, results[i-1]))
            # print("doc in trueRels:", results[i-1] in trueRels)
            if int(results[i-1]) in trueRels:
                # if verbose:
                    # print("Document at rank %d is relevant" % i)
                prec = relevantFound/i
            else:
                prec = 0
            if verbose:
                print("%5d|%4d | %5f" % (i, relevantFound, prec))
            mean += prec

        return mean/len(trueRels)

class EvalIRModel():
    def __init__(self, queries, irmodels, evals):
        """
        :param queries: List of Query object
        :param irmodels: dictionary of {name:IRmodel object}
        :param evals: dictionary of {name:EvalMeasure object}"""
        self.queries = queries
        self.irmodels = irmodels
        self.evals = evals

    def eval(self):
        pass

