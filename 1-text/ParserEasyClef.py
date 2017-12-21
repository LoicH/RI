# -*- coding: utf-8 -*-
import Parser
import Document as Doc
import re

class ParserEasyClef(Parser.Parser):
    def __init__(self):
        Parser.Parser.__init__(self,".I")
    
    def getDocument(self, text):
        docId = re.search("\.I\s(.*)", text).group(1)
        docTitle = re.search("\.T\s(.*)", text).group(1)
        docText = re.search("\.W\s(.*)", text).group(1)
        docDate = re.search("\.B\s(.*)", text).group(1)
        doc = Doc.Document(docId, docTitle+"\n"+docText+"\n"+docDate,
                           others=dict())
        return doc
