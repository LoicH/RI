'''
Created on 5 sept. 2016

@author: SL
'''


class Document(object):
    """ Document class """


    def __init__(self,identifier,text="",others=None):
        """ Constructor 
        :param identifier: unique identifier
        :param text: Usually 'title+author+kw+texte'
        :param other: Other informations, such as where to find this doc.
            In other["from"] you can find the source doc."""
        
        self.identifier=identifier
        self.text=text
        # dictionary of info
        self.others=others;
        
    def getId(self):
        return self.identifier
    
    def getText(self):
        return self.text
       
    def get(self,key):
        """get("from") gives the original source file"""
        return self.others[key]  
    
    def set(self,key,value):
        self.others[key]=value

    def __str__(self):
        return "id="+self.identifier+"\ntxt="+self.text+"\nothers="+str(self.others)




