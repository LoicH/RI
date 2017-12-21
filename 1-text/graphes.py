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
            self.graph = dok_matrix(len(self.index.getDocsID()))
            for docTitle in self.seeds:
                self.addDoc(docTitle, addNew=True)
                for prevTitle in np.random.choice(self.index.getPrevNodes(docTitle), 
                                                  size=self.prevNumber):
                    self.addDoc(prevTitle, addNew=False)
        return self.graph
        


class PageRank(RandomSurfer):
    def __init__(self, index, seeds, prevNumber):
        super().__init(index, seeds, prevNumber)
    
    
class HITS(RandomSurfer):
    def __init__(self, index, seeds, prevNumber):
        super().__init(index, seeds, prevNumber)
    