# These are the trained models for keyword extraction
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

##from sklearn
from minerva.proc.nlp_functions import removeStopwords

class BaseKeywordExtractor(object):
    """
        Base class for the keyword extractors
    """
    def __init__(self):
        """
        """
        pass

    def train(self, train_set):
        """
        """

    def extract(self, doc, cit, params={}):
        """
        """
        pass

class TFIDFKeywordExtractor(BaseKeywordExtractor):
    """
        Simple tfidf keyword extractor
    """

    def train(self, train_set, params):
        """
        """

##        all_kws={x[0]:x[1] for x in kw_data["best_kws"]}
##        for sent in docfrom.allsentences:
##            for token in sent["token_features"]:
##                if token["text"] in all_kws:
##                    token["extract"]=True
##                    token["weight"]=all_kws[token["text"]]

    def extract(self, doc, cit, params={}):
        """
        """
        doctext=doc.getFullDocumentText(True, False)



def main():
    pass

if __name__ == '__main__':
    main()
