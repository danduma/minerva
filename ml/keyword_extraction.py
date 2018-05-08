# These are the trained models for keyword extraction
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

from __future__ import absolute_import
# For license information, see LICENSE.TXT
from __future__ import print_function

import os

import six.moves.cPickle
from ml.keyword_support import filterFeatures, unPadTokens, getTokenListFromContexts
from six.moves import range
from six.moves import zip
from sklearn import ensemble, svm, neural_network
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer


class BaseKeywordExtractor(object):
    """
        Base class for the keyword extractors
    """

    def __init__(self, subclass, params={}, fold=0):
        """
        """
        self.filepath = ""

    @classmethod
    def processReader(self, reader):
        return reader

    def learnFeatures(self, all_token_features):
        """
            Prepares the vectorizer, teaches it all of the features. Needed
            as each possible lemma becomes an individual feature
        """
        pass

    def train(self, train_set, params={}):
        """
        """

    def test(self, test_set, params={}):
        """
        """

    def extract(self, doc, cit, params={}):
        """
        """
        pass

    def saveClasifier(self, classifier, filename):
        six.moves.cPickle.dump(classifier, open(filename, "w"))

    def loadClasifier(self, filename):
        self.classifier = six.moves.cPickle.load(open(os.path.join(self.filepath, filename), "r"))


class TFIDFKeywordExtractor(BaseKeywordExtractor):
    """
        Simple tfidf keyword extractor. Picks the top terms (individually) by
        TFIDF score. Score is already annotated on each token.
    """

    def __init__(self, subclass, params={}, fold=0):
        """
        """

    @classmethod
    def processReader(self, reader):
        return reader

    def train(self, train_set, params={}):
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
        doctext = doc.formatTextForExtraction(doc.getFullDocumentText(True, False))


class SKLearnExtractor(BaseKeywordExtractor):
    """
        Simple SVM extractor
    """

    def __init__(self, subclass, params={}, fold=0, exp={"exp_dir": ".", "name": "exp"}, interactive=False):
        """
        """
        self.subclass_name = subclass
        self.subclass = None
        self.params = params
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = self._get_classifier()
        self.fold = fold
        self.exp = exp
        self.interactive = interactive

    def _get_classifier(self):
        """
        Creates an instance of the classifier specified in the

        :param params:
        :return:
        """
        for module in [ensemble, svm, neural_network]:
            if hasattr(module, self.subclass_name):
                self.subclass = getattr(module, self.subclass_name)
                break

        assert self.subclass, "Extractor class not found " + str(self.subclass_name)

        return self.subclass(**self.params)

    @classmethod
    def processReader(self, reader):
        return unPadTokens(getTokenListFromContexts(reader))

    def learnFeatures(self, all_token_features, ignore_features=[]):
        """
            Prepares the vectorizer, teaches it all of the features. Needed
            as each possible lemma becomes an individual feature
        """

        features, labels, weights = list(zip(*all_token_features))

        self.vectorizer.fit([filterFeatures(f, ignore_features) for f in features])

    def train(self, train_set, params={}):
        """
        """
        features, labels, weights = list(zip(*train_set))
        features = self.vectorizer.transform(features)
        self.classifier.fit(features, labels)
        self.informativeFeatures(features)

    def informativeFeatures(self, features):
        """
        Make a plot of the most informative features as reported by the classifier

        :param features:
        :return:
        """
        import numpy as np
        import matplotlib.pyplot as plt

        num_features = min(features.shape[1], 30)

        try:
            importances = self.classifier.feature_importances_
        except:
            print("Classifier does not support listing feature importance")
            return

        std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_],
                     axis=0)

        indices = np.argsort(importances)[::-1]
        indices = indices[:num_features]

        # Print the feature ranking
        print("Feature ranking:")

        feature_names = self.vectorizer.get_feature_names()

        for f in range(num_features):
            print("%d. %s - %d (%f)" % (f, feature_names[indices[f]], indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(list(range(num_features)), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(list(range(num_features)), [feature_names[index] for index in indices], rotation=90)
        plt.xlim([-1, num_features])

        if self.interactive:
            plt.show()
        else:
            filename = os.path.join(self.exp["exp_dir"], "feature_importance_%s_%d.png" % (self.exp["name"], self.fold))
            plt.savefig(filename)

    def test(self, test_set, params={}):
        """
            Test the trained extractor on a test set
        """
        features, labels, weights = list(zip(*test_set))

        true_ones = []
        for index, extract in enumerate(labels):
            if extract:
                true_ones.append(features[index])

        terms_to_extract = [f["text"] for f in true_ones]
        from collections import Counter
        counter = Counter(terms_to_extract)
        ##        print("Terms to extract:",sorted(counter.items(), key=lambda x:x[1],reverse=True))

        ##        print("True in labels:",(True in labels))
        features = self.vectorizer.transform(features)
        ##        print(features)
        predicted = self.classifier.predict(features)
        ##        print("True in predicted:",(True in predicted))

        print("Classification report for classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(labels, predicted)))
        print("AUC:\n%f" % metrics.roc_auc_score(labels, predicted, average="weighted"))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels, predicted))


def main():
    pass


if __name__ == '__main__':
    main()
