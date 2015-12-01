#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      dd
#
# Created:     11/11/2014
# Copyright:   (c) dd 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import nltk.collocations
import cPickle
from az_features import *
from nlp_functions import *


class AZannotator:
    def __init__(self,filename=None):
        """
            Takes optional filename to load prebuilt classifier
        """
        self.classifier=None
        if filename:
            self.loadClasifier(filename)
        pass

    def saveClasifier(self,clasifier,filename):
        cPickle.dump(classifier,open(filename,"w"))

    def loadClasifier(self,filename):
        self.classifier=cPickle.load(open(filename,"r"))

    def annotateDoc(self,doc):
        """
            Annotates each sentence with its AZ class
        """
        prebuildAZFeaturesForDoc(doc)

        for sentence in doc.allsentences:
            features={}
            for feature in AZ_all_features:
                feature(features,sentence,doc, None)

            for feature in AZ_precomputed_features:
                features["F_"+feature]=sentence[feature]

            formPat.extractFeatures(sentence["text"],features) # formulaic patterns
            formPat.extractFeatures(sentence["text"],features,True) # agent patterns
    ##            print features
            az=self.classifier.classify(features)
            sentence["az"]=az

        removePrebuiltAZFeatures(doc)

class CFCannotator:
    def __init__(self,filename=None):
        """
            Takes optional filename to load prebuilt classifier
        """
        self.classifier=None
        if filename:
            self.loadClasifier(filename)
        pass

    def saveClasifier(self,clasifier,filename):
        cPickle.dump(classifier,open(filename,"w"))

    def loadClasifier(self,filename):
        self.classifier=cPickle.load(open(filename,"r"))

    def annotateDoc(self,doc):
        """
            Annotates each citation with its citation function "cfunc"
        """
        prebuildAZFeaturesForDoc(doc)

        for sentence in doc.allsentences:
            cfc_citations=[doc.citation_by_id[citation] for citation in sentence["citations"]]
            cfc_citations.extend(loadRefAuthorsFromSentence(sentence))

            for citation in cfc_citations:
                features={}
                for feature in CFC_all_features:
                    feature(features,sentence,doc, None)

                for feature in CFC_precomputed_features:
                    features["F_"+feature]=sentence[feature]

                formPat.extractFeatures(sentence["text"],features) # formulaic patterns
                formPat.extractFeatures(sentence["text"],features,True) # agent patterns

                citation["cfunc"]=self.classifier.classify(features)
##                print citation["cfunc"]

        removePrebuiltAZFeatures(doc)


def main():

    pass

if __name__ == '__main__':
    main()
