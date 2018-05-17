# Own implementation of AZ annotation
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import

import inspect
import os
import six.moves.cPickle

from .az_features import (prebuildAZFeaturesForDoc, AZ_all_features,
                          AZ_precomputed_features, formPat, removePrebuiltAZFeatures, loadRefAuthorsFromSentence,
                          CFC_all_features, CFC_precomputed_features)


class AZannotator:
    def __init__(self, filename=None):
        """
            Takes optional filename to load prebuilt classifier
        """
        self.classifier = None
        self.filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        if filename:
            self.loadClassifier(filename)
        pass

    def saveClassifier(self, classifier, filename):
        six.moves.cPickle.dump(classifier, open(filename, "w"))

    def loadClassifier(self, filename):
        self.classifier = six.moves.cPickle.load(open(os.path.join(self.filepath, filename), "rb"))

    def annotateDoc(self, doc):
        """
            Annotates each sentence with its AZ class
        """
        prebuildAZFeaturesForDoc(doc)

        for sentence in doc.allsentences:
            features = {}
            for feature in AZ_all_features:
                feature(features, sentence, doc, None)

            for feature in AZ_precomputed_features:
                if feature in sentence:
                    features["F_" + feature] = sentence[feature]

            formPat.extractFeatures(sentence["text"], features)  # formulaic patterns
            formPat.extractFeatures(sentence["text"], features, True)  # agent patterns
            ##            print features
            az = self.classifier.classify(features)
            sentence["az"] = az

        removePrebuiltAZFeatures(doc)


class CFCannotator:
    def __init__(self, filename=None):
        """
            Takes optional filename to load prebuilt classifier
        """
        self.classifier = None
        self.filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        if filename:
            self.loadClasifier(filename)
        pass

    def saveClasifier(self, classifier, filename):
        six.moves.cPickle.dump(classifier, open(filename, "w"))

    def loadClasifier(self, filename):
        self.classifier = six.moves.cPickle.load(open(os.path.join(self.filepath, filename), "r"))

    def annotateDoc(self, doc):
        """
            Annotates each citation with its citation function "cfunc"
        """
        prebuildAZFeaturesForDoc(doc)

        for sentence in doc.allsentences:
            cfc_citations = [doc.citation_by_id[citation] for citation in sentence.get("citations", [])]
            cfc_citations.extend(loadRefAuthorsFromSentence(sentence))

            for citation in cfc_citations:
                features = {}
                for feature in CFC_all_features:
                    feature(features, sentence, doc, None)

                for feature in CFC_precomputed_features:
                    features["F_" + feature] = sentence[feature]

                formPat.extractFeatures(sentence["text"], features)  # formulaic patterns
                formPat.extractFeatures(sentence["text"], features, True)  # agent patterns

                citation["cfunc"] = self.classifier.classify(features)
        ##                print citation["cfunc"]

        removePrebuiltAZFeatures(doc)


def main():
    pass


if __name__ == '__main__':
    main()
