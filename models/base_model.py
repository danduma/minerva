import os
import sys

import numpy as np

from models.keyword_features import statsOnPredictions

np.warnings.filterwarnings('ignore')

from .keyword_features import FeaturesReader, normaliseFeatures, filterOutFeatures, normaliseWeights
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from proc.stopword_proc import getStopwords
from tqdm import tqdm
import json

CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)


class BaseModel(object):
    def __init__(self, exp_dir,
                 params={},
                 train_data_filename="feature_data.json.gz",
                 test_data_filename="feature_data_test.json.gz"):

        self.exp_dir = exp_dir
        self.features_data_filename = os.path.join(self.exp_dir, train_data_filename)
        self.test_data_filename = os.path.join(self.exp_dir, test_data_filename)
        self.dtype = np.float32
        self.dict_vectorizer = DictVectorizer(dtype=self.dtype, sparse=False)
        self.all_context_features = []
        self.regression = params.get("regression", False)
        self.stopwords = getStopwords(self.exp_dir)
        self.features_to_filter_out = ["lemma", "dep", "pos", "text", "lemma_"]
        self.corpus_label = ""

        if "pmc" in self.exp_dir:
            self.corpus_label = "PMC"
            self.corpus = "pmc"
        elif "aac" in self.exp_dir:
            self.corpus = "aac"
            self.corpus_label = "AAC"

    def augmentSingleContext(self, context):
        pass

    def processFeatures(self, contexts):
        for context in tqdm(contexts, desc="Adding context features"):
            self.augmentSingleContext(context)

        self.filterStopWordsInData(self.contexts)
        normaliseFeatures(contexts)
        normaliseWeights(contexts)

    def loadData(self):
        print("Loading data...")
        self.reader = FeaturesReader(self.features_data_filename)
        self.contexts = [c for c in self.reader]
        if len(self.contexts) == 0:
            raise ValueError("No data to train with!")

        self.preProcessLoadedData()
        self.processFeatures(self.contexts)
        self.learnFeatures()
        self.postProcessLoadedData()

    def preProcessLoadedData(self):
        pass

    def postProcessLoadedData(self):
        pass

    def filterOutSingleContextFeatures(self, context):
        """
        This filters out features that we want to keep in the context
        but do not want to vectorize and use for the model

        :param context:
        :return:
        """
        return filterOutFeatures(context, self.features_to_filter_out, self.corpus)

    def filterStopWordsInPredicted(self, tokens, predicted):
        self.extra_stopwords_removed = []
        for index, t in enumerate(tokens):
            if t["text"].lower() in self.stopwords:
                if predicted[index] > 0:
                    self.extra_stopwords_removed.append(t["text"].lower())
                    if self.regression:
                        predicted[index] = 0
                    else:
                        predicted[index] = False

        return predicted

    def filterStopWordsInData(self, contexts):
        self.extra_stopwords_removed = []
        for ctx_index, context in enumerate(contexts):
            for index, t in enumerate(context["tokens"]):
                if t["text"].lower() in self.stopwords or len(t["text"]) < 3:
                    if context["extract_mask"][index] > 0 or context["weight_mask"][index] > 0:
                        self.extra_stopwords_removed.append(t["text"].lower())
                        context["weight_mask"][index] = 0
                        context["extract_mask"][index] = False

        return contexts

    def learnFeatures(self):
        self.all_context_features = []
        for context in self.contexts:
            self.all_context_features.extend(self.filterOutSingleContextFeatures(context)["tokens"])

        self.dict_vectorizer.fit(self.all_context_features)
        self.num_extra_features = len(self.dict_vectorizer.get_feature_names())
        print("Number of token features", self.num_extra_features)
        print(self.dict_vectorizer.get_feature_names())

    def trainModel(self):
        pass

    def defineModel(self):
        pass

    def testModel(self):
        pass

    def externalTestModel(self):
        # import subprocess
        this_file = os.path.abspath(sys.argv[0])
        command = "python3 " + os.path.join(os.path.dirname(os.path.dirname(this_file)),
                                            "kw_evaluation_runs",
                                            "evaluate_kw_selection.py")
        if "aac" in self.exp_dir:
            corpus = "aac"
        elif "pmc" in self.exp_dir:
            corpus = "pmc"
        else:
            corpus = "aac"

        if self.exp_dir.endswith(os.sep):
            self.exp_dir = self.exp_dir[:-1]

        exp_name = os.path.basename(self.exp_dir)
        command += " " + corpus + " " + exp_name
        print("Running:\n", command)
        os.system(command)

    def printClassificationResults(self, y_test, predicted):
        print("Classification report for classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(y_test, predicted)))
        print("AUC:\n%f" % metrics.roc_auc_score(y_test, predicted, average="weighted"))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))

    def savePredictions(self, result_contexts):
        with open(os.path.join(self.exp_dir, "predictions.json"), "w") as f:
            json.dump(result_contexts, f)

        statsOnPredictions(result_contexts, self.stopwords)

    def run(self, external_test=False):
        self.loadData()
        self.defineModel()
        self.trainModel()
        self.testModel()
        if external_test:
            self.externalTestModel()
