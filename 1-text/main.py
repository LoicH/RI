#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import ParserCACM
import TextRepresenter
import indexation
import modeles
from query import QueryParserCACM
import evaluation
import os
import matplotlib.pyplot as plt
import numpy as np

""" Some info to remember when testing:
- Query 10 is about parallel computation
- Doc 46 is about parallelism too
"""

if __name__ == "__main__":

    # Constants:
    srcFolder = "cacm"
    srcFile = "cacm.txt"
    qryFile = "cacm.qry"
    relFile = "cacm.rel"
    gendata = "gendata"
    indexName = "cacm"
    docId = 46
    queryId = np.random.randint(1,60)
    wordTest = "logic"

    cacm_txt = os.path.join(srcFolder, srcFile)
    cacm_qry = os.path.join(srcFolder, qryFile)
    cacm_rel = os.path.join(srcFolder, relFile)

    # Construct the index:
    idx = indexation.Index(indexName, gendata)
    stemmer = TextRepresenter.PorterStemmer()
    idx.indexation(cacm_txt, ParserCACM.ParserCACM(),
                   stemmer)

    print("\n###### A bit of testing: ###### ")
    print("Retrieve stems in doc %d:" % docId)
    print(idx.getTfsForDoc(docId))
    print("""Should look like
    “Multiprogramming STRETCH: Feasibility Considerations
    .W
    The tendency towards increased parallelism in
    computers is noted.  Exploitation of this parallelism
    presents a number of new problems in machine design
    and in programming systems.”""")

    print("\nRetrieve docs that contains '%s':" % wordTest)
    print(idx.getTfsForStem(wordTest))
    print("Should include doc 63 and ",docId,"")

    print("\n###### Testing BinaryWeighter: ###### ")
    bw = modeles.BinaryWeighter(idx)
    print("bw.getDocWeightsForDoc(",docId,"):", 
        bw.getDocWeightsForDoc(docId))
    print("\nbw.getDocWeightsForStem('"+wordTest+"'):", 
        bw.getDocWeightsForStem(wordTest))
    query = stemmer.getTextRepresentation("parallelism in programming")
    print("\nbw.getWeightsForQuery('parallelism in programming'):", 
        bw.getWeightsForQuery(query))


    print("\n###### Testing Vectoriel with BinaryWeighter: ###### ")
    vect = modeles.Vectoriel(idx, bw)
    print("Top 10 documents for the query:")
    print(vect.getRanking(query)[:10])

    print("\n###### Testing BinaryWeighter: ###### ")
    tfidfWeighter = modeles.TfidfWeighter(idx)
    print("tfidfWeighter.getDocWeightsForDoc(", docId, "):",
        tfidfWeighter.getDocWeightsForDoc(docId))
    print("\ntfidfWeighter.getDocWeightsForStem(\"logic\"):",
        tfidfWeighter.getDocWeightsForStem(wordTest))
    print("\ntfidfWeighter.getWeightsForQuery(query):",
        tfidfWeighter.getWeightsForQuery(query))


    print("\n###### Testing Vectoriel with TfidfWeighter: ###### ")
    vect = modeles.Vectoriel(idx, tfidfWeighter)
    print("Top 10 documents for the query:")
    print(vect.getRanking(query)[:10])

    print("\n###### Testing QueryParserCACM: ###### ")
    qp = QueryParserCACM(cacm_qry, cacm_rel)
    query = qp.nextQuery()
    print("Searching for query #%d:" % queryId)
    while query is not None and query.getID() != str(queryId):
        print(query)
        print(20*'-')
        query = qp.nextQuery()

    print("Query:", query)
    queryTxt = stemmer.getTextRepresentation(query.getText())
    scores = vect.getRanking(queryTxt)
    print("Scores:", scores)
    
    print("\n###### Testing evaluation.PrecisionRecallMeasure: ###### ")
    irlist = evaluation.IRList(query, scores)
    precisRecall = evaluation.PrecisionRecallMeasure(irlist)
    pr = precisRecall.eval()
    plt.scatter(*zip(*pr))
    plt.show()

