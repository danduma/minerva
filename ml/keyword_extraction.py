# These are the trained models for keyword extraction
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from sklearn.feature_extraction import DictVectorizer

from minerva.proc.nlp_functions import removeStopwords

SENTENCE_FEATURES_TO_COPY=["az"]

def buildFeatureSetForContext(all_token_features,all_keywords):
    """
        Returns a list of (token_features_dict,{True,False}) tuples. For each
        token, based on its features, the token is annotated as to-be-extracted or not
    """
    res=[]
    for token in all_token_features:
        extract=(token["text"] in all_keywords.keys())
        res.append((token,extract))
    return res

def prepareFeatureData(precomputed_contexts):
    """
        Extracts just the features, prepares the data in a format ready for training
        classifiers, just one very long list of (token_features_dict, {True,False})
    """
    from minerva.proc.results_logging import ProgressIndicator
    res=[]
    all_token_features=[]

    progress=ProgressIndicator(True,len(precomputed_contexts),True)
    for context in precomputed_contexts:
        all_keywords={t[0]:t[1] for t in context["best_kws"]}
        for sentence in context["context"]:
            for feature in SENTENCE_FEATURES_TO_COPY:
                for token_feature in sentence["token_features"]:
                    token_feature[feature]=sentence[feature]
            all_token_features.extend(sentence["token_features"])
        progress.showProgressReport("Preparing feature data")

    return buildFeatureSetForContext(all_token_features,all_keywords)


class BaseKeywordExtractor(object):
    """
        Base class for the keyword extractors
    """
    def __init__(self, params={}):
        """
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

    def saveClasifier(self,clasifier,filename):
        cPickle.dump(classifier,open(filename,"w"))

    def loadClasifier(self,filename):
        self.classifier=cPickle.load(open(os.path.join(self.filepath,filename),"r"))

class TFIDFKeywordExtractor(BaseKeywordExtractor):
    """
        Simple tfidf keyword extractor. Picks the top terms (individually) by
        TFIDF score. Score is already annotated on each token.
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
    def __init__(self, params={}):
        """
        """
        from sklearn import svm, metrics
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier=svm.SVC(gamma=0.001,
                                verbose=True)

    def train(self, train_set, params={}):
        """
        """
        features, labels=zip(*train_set)
        features=self.vectorizer.fit_transform(features)

        self.classifier.fit(features,labels)

    def test(self, test_set, params={}):
        """
        """
        features, labels=zip(*test_set)

        features=np.array(result.items(), dtype=test_set)

        predicted=self.classifier.predict(features)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(labels, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels, predicted))

def main():
    pass

if __name__ == '__main__':
    main()
