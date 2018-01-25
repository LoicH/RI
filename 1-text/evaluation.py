#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import TextRepresenter
import itertools

class IRList():
    """ Contains a query and the scores found for this query """
    def __init__(self, query, scores, ranking=None):
        """ Create new IRList
        :param query: The Query object
        :param scores: a list of tuples [(docId, score)]"""
        self.query = query
        self.scores = scores # List [(docId, score)]
        self.ranking = ranking

    def getQuery(self):
        """ Return the Query object"""
        return self.query

    def getScores(self):
        return self.scores
    
    def getRanking(self):
        """ Return sorted list of relevant results for the query in the
        IRList object"""
        if self.ranking is None:
            sortRes = sorted(self.getScores(), key=lambda tupl:tupl[1], reverse=True)
            self.ranking = [docId for (docId, score) in sortRes]
        return self.ranking




class EvalMeasure:
    """ The abstract class for measure methods """
    def __init__(self, irlist):
        self.irlist = irlist

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
        results = self.irlist.getRanking()
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
        results = self.irlist.getRanking()

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
        results = self.irlist.getRanking()
        precision = 0
        N = min(n, len(trueRels))
        for result in results[:N]:
            if verbose:
                print("Result: ", result)
            if int(result) in trueRels:
                precision += 1
                if verbose:
                    print("Relevant, found docs =", precision)
        return precision/N
        
class ClusterRecallNDocuments(EvalMeasure):
    def __init__(self, irlist):
        super().__init__(irlist)

    def eval(self, n, verbose=False):
        """ Compute the performance of a model.
                :return: The cluster recall at the n'th rank"""
        # Truely relevant results for the query:
        trueRels = self.irlist.getQuery().getRelevants()
        # trueRels is dict {docId(int) : {'subtheme': int, 'score': float},}}
        # Results we found for the query (sorted list of docs ID)
        results = self.irlist.getRanking()
        # Unique subthemes in the real relevant results
        subtheme_set = set()
        for elt in list(trueRels.values()):
            subtheme_set.add(elt["subtheme"])
        real_nb_cluster = len(subtheme_set)
        if verbose:
            print("There are %d clusters for this query" % real_nb_cluster)
        
        N = min(n, len(trueRels))
        found_clusters = set()
        for result in results[:N]:
            # result is a string: doc ID
            if int(result) in trueRels.keys():
                cluster = trueRels[int(result)]['subtheme']
                if verbose:
                    print("Result: %s, cluster: %d" % (result, cluster))
                found_clusters.add(cluster)
        return len(found_clusters)/real_nb_cluster

        
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
        N_models = len(self.irmodels)
        N_measures = len(self.measures)
        for i, (irmodel_name, irmodel) in enumerate(self.irmodels.items()):
            if verbose:
                print("[%3d/%3d] IRModel '%s'" % (i, N_models, irmodel_name))
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



def dict_combinations(dic):
    keys = dic.keys()
    #print(keys)
    values = [dic[key] for key in keys]
    #print("values:", list(values), ".")
    #for combination in itertools.product(*values):
        #print(combination)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations

def gridsearch(model_class, param_grid, queries, measure_object, verbose=False):
    """
    :param model_class: modeles.Vectoriel for instance (the class, not an instance)
    :param param_grid: dict of {string:iterable}
    :param queries: list of Query objects
    :param measure_class: evaluation.AveragePrecision() for instance
    """
    params = []
    irmodels = {}
    all_combinations = dict_combinations(param_grid)
    N_comb = len(all_combinations)
    for i, comb in enumerate(all_combinations):
        params.append(comb)
        irmodels[i] = model_class(**comb)
    eval_models = EvalIRModel(queries, irmodels, {'measure':measure_object})
    if verbose:
        print("Calling eval()")
        scores = eval_models.eval(verbose=verbose)
        for k,v in scores.items():
            print(params[k[0]])
            print("--->", v[0])
    else:
        scores = eval_models.eval()
    best_irmodel = max(scores.keys(), key=(lambda key: scores[key][0]))[0]
    return params[best_irmodel]
