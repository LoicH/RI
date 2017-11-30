# -*- coding: utf-8 -*-

"""
@author: Loïc Herbelot
"""

import io
import re


class Query():
    def __init__(self, queryID, text,relevants=None):
        """
        :param queryID:
        :param text:
        :param relevants: Dictionary {id: {"subtheme":int, "score":float}}
        """
        self.id = queryID
        self.text = text
        self.relevants = relevants

    def getID(self):
        return self.id

    def getText(self):
        return self.text

    def getRelevants(self):
        return self.relevants

    def setRelevants(self, rel):
        self.relevants = rel

    def __str__(self):
        return ("Query {id="+self.id+", txt='"+self.text+
                "', relevances="+str(list(self.relevants.keys()))+"}")


class QueryParser:
    """ Assumptions for the QueryParser:
    1) All of the query data (id, question) in the query file is between two tags.
    2) All of the relevances data (queryID, docID, score) is sorted in the
       relevance file.

    A subclass must implement the following methods:
    1) parseRelevData(self, data (string)) -> QueryObject
    2) parseQueryData(self, data (string)) -> dictionary of docs scores

    """
    def __init__(self, qry, rel, begin, end=None):
        """ Constructor
        """
        self.begin = begin
        self.end = end

        # open the files and get their lengths
        self.qryFile = open(qry, "r")
        self.qryLen = self.qryFile.seek(0, io.SEEK_END)
        self.qryFile.seek(0)

        self.relFile = open(rel, "r")
        self.relLen = self.relFile.seek(0, io.SEEK_END)
        self.relFile.seek(0)

    def __del__(self):
        if (self.qryFile is not None):
            self.qryFile.close()
        if (self.relFile is not None):
            self.relFile.close()

    def extractQueryData(self):
        """ Return the data extracted and reposition the cursor in the file
            that describes the query.
        :return: text data
        """
        # Search the begin tag
        line = self.qryFile.readline()
        while (self.qryFile.tell() < self.qryLen and          # not at the end of the file
               self.begin not in line and                     # not a begin tag in the line
               (self.end is not None or self.end not in line)) : # not an end tag in the line
#            print(">", line, end='')
            line = self.qryFile.readline()

        if self.qryFile.tell() == self.qryLen: # We reached the end of the file
            return None
#        print("Found the start:", line)
        queryData = line

        # Read line by line until the end tag or a new start tag.
#        print("Reading the data query")
        cursor = self.qryFile.tell()
        line = self.qryFile.readline()
        while (self.qryFile.tell() < self.qryLen and
               self.begin not in line):
#            print(">", line, end='')
            queryData += line
            cursor = self.qryFile.tell()
            line = self.qryFile.readline()

        # Reposition the cursor before the new start tag
        self.qryFile.seek(cursor)

        return queryData

    def extractRelevData(self, qryId):
        """ Extract the data from the .rel file"""

        def startingNumber(s):
            """Returns whether s starts with n (string)"""
            return int(re.match("^0*(\d*)", s).group(1))

        # print("Extract relevance data for query #" + qryId)
        line = self.relFile.readline()
        while (self.relFile.tell() < self.relLen and
               startingNumber(line) < int(qryId)):
            # print(">", line, end='')
            line = self.relFile.readline()

        # print("self.relFile.tell() = %d, self.relLen = %d" % (self.relFile.tell(), self.relLen))
        if self.relFile.tell() == self.relLen:
            if startingNumber(line) == int(qryId): # the last line of the file contained data
                return line
            else:
                return None
        # print("Found the start:", line)
        queryData = line

        # print("Reading the relevances")
        cursor = self.relFile.tell()
        line = self.relFile.readline()
        while (self.relFile.tell() < self.relLen and
               startingNumber(line) == int(qryId)) :
            queryData += line
            # print(">", line, end='')
            cursor = self.relFile.tell()
            line = self.relFile.readline()

        # Reposition the cursor before the new start tag
        self.relFile.seek(cursor)
        return queryData


    def parseQueryData(self, data):
        raise NotImplementedError("Absract method")

    def parseRelevData(self, data):
        raise NotImplementedError("Absract method")

    def nextQuery(self):
        # Extract data between start and end tags
        queryData = self.extractQueryData()
        if queryData is None:
            return None
        # Call the abstract method "parseQueryData" to construct a Query object
        query = self.parseQueryData(queryData)
        # Call the method "extractRelevance(queryId)"
        relevData = self.extractRelevData(query.getID())
        # Call the abstract method "parseRelevance" to get a dictionary
        # Add this dictionary to the Query object
        query.setRelevants(self.parseRelevData(relevData))
        return query



class QueryParserCACM(QueryParser):
    def __init__(self, qryFile, relFile):
        """ Constructor"""
        super().__init__(qryFile, relFile, ".I")


    def parseRelevData(self, data):
        """ Parse the raw relevances in text to a dictionary
        :return: A dictionary {id(int): {"subtheme":int, "score":float"}}
        """
        dic = {}
        for line in data.split('\n'):
            if len(line) >1:
                search = re.search("^\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)\s*$", line)
                qryId, docId, subtheme, score = search.groups()
                dic[int(docId)] = {"subtheme":int(subtheme),
                                   "score":float(score)}
        return dic

    def parseQueryData(self, data):
        """ Parse the raw query data into a Query object
        .I [id]
        .W
         [text]
        :return: a Query object"""
        idSearch = re.search(".I (\d+)", data)
        if idSearch is None:
            print("Error, can't retrieve the ID of the query in '"+data+"'")
        queryId = idSearch.group(1)
        questionSearch = re.search(".W\s*([\s\S]*?)\.[A-Z]\s", data)
        if questionSearch is None:
            print("Error, can't retrieve the question of the query in '"+data+"'")
        question = questionSearch.group(1)

        return Query(queryId, question)


if __name__ == "__main__":
    qp = QueryParserCACM("cacm_sample/cacm.qry", "cacm_sample/cacm.rel")
    query = qp.nextQuery()
    while query is not None:
        print(query)
        print(20*'-')
        query = qp.nextQuery()
