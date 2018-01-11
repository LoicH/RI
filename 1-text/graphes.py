# -*- coding: utf-8 -*-

""" Graph module
- PageRank
- HITS """

from scipy.sparse import dok_matrix
import numpy as np

class RandomWalker():
    def __init__(self, index, seeds, prevNeighbours):
        """
        :param seeds: List of docs title (strings)
        :param prevNeighbours: Number of previous neighbours to take
        """
        self.index = index
        self.seeds = seeds
        self.prevNeighbours = prevNeighbours
        self.nodes = set(seeds) # sets of node id
        self.nodeList = []
        self.graph = None #self.graph[i,j] = 1 if i→j
        self.idxMap = {}
        
        # Add all desired nodes:
        for seedTitle in self.seeds:
            # Add all childs of seedTitle in nodes: 
            self.nodes |= set(self.index.getSuccNodes(seedTitle))
            # Add some parents of seedTitle:
            self.nodes |= set(np.random.choice(self.index.getPrevNodes(seedTitle), 
                                               size=self.prevNeighbours))
        # Assign an index to each node:
        i = 0
        for node in self.nodes:
            self.idxMap[node] = i
            self.nodeList.append(node)
            i += 1
        # Construct the adjacency matrix:
        self.graph = dok_matrix((i,i))
        for node in self.nodes:
            nodeIdx = self.idxMap[node]
            # Get children, keep only the ones in self.nodes
            children = set(self.index.getSuccNodes(node)) & self.nodes
            childrenIdx = [self.idxMap[c] for c in children]
            self.graph[nodeIdx, childrenIdx] = 1


class PageRank(RandomWalker):
    def __init__(self, index, seeds, prevNeighbours):
        super().__init__(index, seeds, prevNeighbours)
    
    def getScores(self, nIter=100, teleportProba=0.1):
        """ Return the PageRank score of every node
        :param nIter: Number of iterations
        :return: Dictionary of {node ID (str) : score}
        """
        # Set every element in the graph matrix to either 1/N or 1/l_j
        graph = self.graph.toarray()
        N = len(self.nodes)
        print("%d nodes" % N)
        for i in range(N):
            if np.alltrue(graph[i] == 0):
                graph[i] = np.ones(N)
            s = (graph[i].sum())
            graph[i] /= s
        print("Graph:\n", graph)
        print("Nodes:", self.nodeList)
        # Do the power method to find the scores:
        scores = np.ones(N)/N
        for i in range(nIter):
            scores = (1-teleportProba) * graph.dot(scores)
            scores += teleportProba/N * np.ones(N)
        
        return {self.nodeList[i] : scores[i] for i in range(N)}
                        
    
class HITS(RandomWalker):
    def __init__(self, index, seeds, prevNeighbours):
        super().__init(index, seeds, prevNeighbours)
    