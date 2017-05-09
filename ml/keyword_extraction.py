# These are the trained models for keyword extraction
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from minerva.proc.nlp_functions import removeStopwords
from minerva.proc.results_logging import ProgressIndicator
import os, cPickle

SENTENCE_FEATURES_TO_COPY=["az","csc_type"]

def buildFeatureSetForContext(all_token_features,all_keywords):
    """
        Returns a list of (token_features_dict,{True,False}) tuples. For each
        token, based on its features, the token is annotated as to-be-extracted or not
    """
    res=[]
    for token in all_token_features:
        extract=(token["text"] in all_keywords.keys())
        weight=token.get("weight",0.0)
        try:
            del token["weight"]
            del token["extract"]
        except:
            pass
        res.append((token,extract,weight))
    return res


def prepareFeatureData(precomputed_contexts):
    """
        Extracts just the features, prepares the data in a format ready for training
        classifiers, just one very long list of (token_features_dict, {True,False})
    """


    all_token_features=[]

    progress=ProgressIndicator(True,len(precomputed_contexts),True)
    for context in precomputed_contexts:
        all_keywords={t[0]:t[1] for t in context["best_kws"]}
        for sentence in context["context"]:
            for feature in SENTENCE_FEATURES_TO_COPY:
                for token_feature in sentence["token_features"]:
                    token_feature[feature]=sentence.get(feature,"")
##            for token_feature in sentence["token_features"]:
##                for key in token_feature:
##                    if key.startswith("dist_cit_"):
        all_token_features.extend(buildFeatureSetForContext(sentence["token_features"],all_keywords))
        progress.showProgressReport("Preparing feature data")
    return all_token_features


class BaseKeywordExtractor(object):
    """
        Base class for the keyword extractors
    """
    def __init__(self, params={}, fold=0):
        """
        """
        self.filepath=""

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

    def saveClasifier(self,classifier,filename):
        cPickle.dump(classifier,open(filename,"w"))

    def loadClasifier(self,filename):
        self.classifier=cPickle.load(open(os.path.join(self.filepath,filename),"r"))

class TFIDFKeywordExtractor(BaseKeywordExtractor):
    """
        Simple tfidf keyword extractor. Picks the top terms (individually) by
        TFIDF score. Score is already annotated on each token.
    """

    def __init__(self, params={}, fold=0):
        """
        """

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
        doctext=doc.getFullDocumentText(True, False)


class SVMKeywordExtractor(BaseKeywordExtractor):
    """
        Simple SVM extractor
    """
    def __init__(self, params={}, fold=0, exp={"exp_dir":".","name":"exp"}, interactive=False):
        """
        """
        from sklearn import svm
        from sklearn import ensemble
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier=svm.SVC(gamma="auto", max_iter=-1, verbose=True)
##        self.classifier=ensemble.RandomForestClassifier(verbose=True)
        self.fold=fold
        self.exp=exp
        self.interactive=interactive

    def learnFeatures(self, all_token_features):
        """
            Prepares the vectorizer, teaches it all of the features. Needed
            as each possible lemma becomes an individual feature
        """
        features, labels, weights=zip(*all_token_features)
        self.vectorizer.fit(features)

    def train(self, train_set, params={}):
        """
        """
        features, labels, weights=zip(*train_set)
        features=self.vectorizer.transform(features)
        self.classifier.fit(features,labels)
        self.informativeFeatures(features)

    def informativeFeatures(self, features):
        import numpy as np
        import matplotlib.pyplot as plt

        num_features=min(features.shape[1],30)

        try:
            importances = self.classifier.feature_importances_
        except:
            print("Classifier does not support listing feature importance")
            return

        std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_],
                     axis=0)

        indices = np.argsort(importances)[::-1]
        indices=indices[:num_features]

        # Print the feature ranking
        print("Feature ranking:")

        feature_names=self.vectorizer.get_feature_names()

        for f in range(num_features):
            print("%d. %s - %d (%f)" % (f, feature_names[indices[f]], indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(num_features), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(num_features), [feature_names[index] for index in indices], rotation=90)
        plt.xlim([-1, num_features])

        if self.interactive:
            plt.show()
        else:
            filename=os.path.join(self.exp["exp_dir"],"feature_importance_%s_%d.png" % (self.exp["name"],self.fold))
            plt.savefig(filename)

    def test(self, test_set, params={}):
        """
        """
        features, labels, weights=zip(*test_set)

        true_ones=[]
        for index, extract in enumerate(labels):
            if extract:
                true_ones.append(features[index])

        terms_to_extract=[f["text"] for f in true_ones]
        from collections import Counter
        counter=Counter(terms_to_extract)
##        print("Terms to extract:",sorted(counter.items(), key=lambda x:x[1],reverse=True))

##        print("True in labels:",(True in labels))
        features=self.vectorizer.transform(features)
##        print(features)
        predicted=self.classifier.predict(features)
##        print("True in predicted:",(True in predicted))

        print("Classification report for classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(labels, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels, predicted))

def main():
    pass

if __name__ == '__main__':
    main()
