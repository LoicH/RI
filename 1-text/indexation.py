# -*- coding: utf-8 -*-

""" Created on 28th sep. 2017
@author: Loïc Herbelot
"""

import ParserCACM
import TextRepresenter
import os

class Index(object):
    """ Stores word frequencies among documents """


    def __init__(self, name, out_dir):
        """ Constructor 
        :param name: the name of the index. The index will be 
            stored under [directory]/[name]_{index, inverted}
        :param out_dir: Where to save the index.
        """
        # The name of the index
        self.name = name
        # The path to the index file
        self.indexPath = os.path.join(out_dir, self.name + "_index.txt")
        # The path to the inverted index file
        self.invertedPath = os.path.join(out_dir, self.name + "_inverted.txt")
        # Dictionary {doc: (position in index, len of representation)} 
        self.docs = {}
        # Dict {stem: {"pos":position in inverted idx, "len":len of repr.}}
        # self.stems["foo"]["pos"] gives the position of "foo"
        self.stems = {}
        # Dict {int(doc): string("source path;position in source;text length")}
        self.docFrom = {}
        
        
    def indexation(self, corpus, parser, txtRepr):
        """ Create the indexes.
        :param corpus: the path to the files that contains 
            all document informations
        :param parser: The Parser object, must implement nextDocument() 
            method
        :param txtRepr: The TextRepresenter object, must implement
            getTextRepresentation(str) method.
        :return: None
        """ 
        print("Performing the indexation...")
        self.parser = parser
        self.textRepresenter = txtRepr
        
        self.parser.initFile(corpus)
        
        # Build the index of documents and compute the length of each
        # stem representation
        print("1st pass: build the index...")
        doc = self.parser.nextDocument()
        with open(self.indexPath, "w") as index:
            while doc is not None:
                title = doc.getId()
                self.docFrom[title] = doc.get("from")
                #~ print("Parsing doc n°" + title)
                stems = (self.textRepresenter
                             .getTextRepresentation(doc.getText()))
                #~ print("Doc",title,"stems:",stems)
                # Add all the stems in the vocabulary
                for stem, freq in stems.items():
                    self.addStem(stem, title, freq)
                    
                docRepr = [w + ":" + str(freq) for (w, freq) in stems.items()]
                # the string that will be written:
                toWrite = title + '{'
                toWrite += ','.join(docRepr) + '}\n'
                self.docs[title] = (index.tell(), len(toWrite))
                
                index.write(toWrite)
                doc = self.parser.nextDocument()
        
        
        # Build the inverted index
        print("2nd pass: build the inverted index...")
        self.parser.initFile(corpus) # Reset the parser 
        doc = self.parser.nextDocument()
        with open(self.invertedPath, "w+") as invIndex:
            while doc is not None:
                title = doc.getId()
                #~ print("Parsing doc n°" + title)
                stems = (self.textRepresenter
                            .getTextRepresentation(doc.getText()))
                for stem, freq in stems.items():
                    self.writeStem(stem, title, freq, invIndex)
                doc = self.parser.nextDocument()
        
        print("Finished.")
        
        
    def writeStem(self, stem, docId, freq, indexFile):
        """Write the stem in the inverted index (used in 2nd pass)
        
        :param stem: The word to write
        :param docId: The identifier for the document
        :param freq: The frequency of the stem in the document
        :param indexFile: The file handler. Should be an open file
        
        :return: None"""
        if self.stems[stem]["pos"] >= 0:  # We already encoutered this stem
            cursor = self.stems[stem]["pos"]
            indexFile.seek(cursor)
            repres = indexFile.read(self.stems[stem]["len"])
             # Write the docId after the first ',,'
            idx_empty = repres.find(',,')
            # Relative seek:
            indexFile.seek(cursor + idx_empty + 1)
            indexFile.write(docId + ':' + str(freq))
            indexFile.seek(cursor)
            repres = indexFile.read(self.stems[stem]["len"])

        else: # New stem
            indexFile.seek(0, os.SEEK_END)
            self.stems[stem]["pos"] = indexFile.tell()
            startRepr = stem + '{' + docId + ":" + str(freq)
            endRepr = '}\n'
            midRepr = (self.stems[stem]["len"] - len(startRepr) - len(endRepr)) * ','
            repres = startRepr + midRepr + endRepr
            indexFile.write(repres)
        
    def addStem(self, stem, docId, freq):
        """ Add the stem in self.stems (used in 1st pass)
        
        :param stem: The word to add to the vocabulary
        :param docId: string, the identifier for the document
        :param freq: The frequency of stem in doc.
        
        :return: None
        """
        if stem in self.stems:        # Not a new stem
            addRepr = docId + ':' + str(freq) + ','
            lenToAdd = len(addRepr) 
            self.stems[stem]["len"] += lenToAdd
            
        else:  # Or add the new stem:
            addRepr = stem + '{' + docId + ':' + str(freq) + '}\n'
            lenToAdd = len(addRepr)
            self.stems[stem] = {"pos": -1,
                                "len": lenToAdd}
        
    def getTfsForDoc(self, docId):
        """ Return the stems found inside a document, with their
        frequency.
        
        :param docId: The identifier for the wanted document
        
        :return: a dictionary of {string(stem): int(frequency)}"""
        
        (pos, length) = self.docs[str(docId)]
        tfs = None
        with open(self.indexPath, 'r') as index:
            index.seek(pos)
            descr = index.read(length)
            start = descr.find('{')
            stop = descr.find('}')
            descr = descr[start+1:stop]
            freqs = [stem.split(':') for stem in descr.split(',')]
            tfs = {word:int(freq) for (word, freq) in freqs}
        
        return tfs
        
    def getTfsForStem(self, stem):
        """Return the doc frequencies of a given stem
        :param stem: The wanted word
        :return: A dictionary {int(docId}: int(frequency)}"""
        if stem not in self.stems:
            return {}
        with open(self.invertedPath, "r") as invIndex:
            invIndex.seek(self.stems[stem]["pos"])
            repres = invIndex.read(self.stems[stem]["len"])             
            # 'repres' should look like: "[stem]{[doc]:[freq], ...}\n"
            if not repres.startswith(stem):
                print("Error with the representation. Exit")
                exit()
                
        # Parsing the representation:
        # List of strings of the format "[docId]:[freq]":
        docFreq = repres.strip(stem)[1:-2].split(',')
        # List of lists of the format ['[docId]', '[freq]']:
        docFreq = [s.split(':') for s in docFreq]
        # Constructing the final dictionary:
        docFreq = {int(docId):int(freq) for (docId, freq) in docFreq}
        return docFreq
        
    def getStrDoc(self, doc):
        """ Return the string from where a document came in the 
        source file""" 
        srcPath, pos, length = self.docFrom[int(doc)]
        with open(srcPath, 'r') as f:
            f.seek(pos)
            txt = f.read(length)
        return txt

    def getDocsID(self):
        return list(self.docs.keys())


