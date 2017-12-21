# -*- coding: utf-8 -*-

""" Graph module
- PageRank
- HITS """

from scipy.sparse import dok_matrix
import numpy as np

class RandomSurfer():
    def __init__(self, index, seeds, prevNumber):
        """
        :param seeds: List of docs ID (strings)
        """
        self.index = index
        self.seeds = seeds
        self.prevNumber = prevNumber
        self.nodes = set() # sets of node id
        self.graph = None
    
    def addDoc(self, docTitle, addNew=False):
        """ Add a document in the graph.
        :param docTitle: The title of the doc (typically string between 1 and docNbr)
        :param addNew: optional, bool (default is False)
            Whether or not to add the new neighbors in the nodes set.
        """
        docId = int(docTitle) - 1
        self.nodes.add(docId)
        for succTitle in self.index.getSuccNodes(docTitle):
            succId = int(succTitle) -1
            if addNew or succId in self.nodes:
                self.nodes.add(succId)
                self.graph[docId, succId] = 1
            
        
    def getGraph(self):
        if self.graph is None:
            nbDocs = len(self.index.getDocsID())
            self.graph = dok_matrix((nbDocs, nbDocs))
            for docTitle in self.seeds:
                self.addDoc(docTitle, addNew=True)
                for prevTitle in np.random.choice(self.index.getPrevNodes(docTitle), 
                                                  size=self.prevNumber):
                    self.addDoc(prevTitle, addNew=False)
        return self.graph
        


class PageRank(RandomSurfer):
    def __init__(self, index, seeds, prevNumber):
        super().__init__(index, seeds, prevNumber)
    
    def solve(self, nIter=100):
        graph = self.getGraph().copy()
        for i in np.unique(graph.nonzero()[0]):
            s = (graph[i].sum())
            for j in graph[i].nonzero()[1]:
                graph[i,j] /= s
        

        u = np.zeros(graph.shape[0])
        u[list(self.nodes)] = 1/(len(self.nodes))
        eigenVect = u*(graph**nIter)
        return {str(nodeId+1):eigenVect[nodeId] for nodeId in self.nodes}
    
class HITS(RandomSurfer):
    def __init__(self, index, seeds, prevNumber):
        super().__init(index, seeds, prevNumber)
    