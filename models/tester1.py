# Keyword trainer. Uses extractors from ml.keyword_extraction
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function

import os
import json
from collections import defaultdict, Counter

# from sklearn import ensemble, svm, neural_network
from sklearn import metrics

from models.keyword_features import FeaturesReader, getTrainTestData, filterOutFeatures, flattenList, \
    makeEvaluationQuery
from models.base_model import BaseModel
# from sklearn import model_selection
# import matplotlib.pyplot as plt

GLOBAL_FILE_COUNTER = 0


def listAllKeywordsToExtractFromReader(reader):
    """
        Lists all keywords that are marked as extract:true in a list or reader object
    """
    to_extract = []

    for kw_data in reader:
        if isinstance(kw_data, dict):
            best_kws = {t[0]: t[1] for t in kw_data["best_kws"]}
            for sent in kw_data["context"]:
                for token in sent["token_features"]:
                    if token["text"].lower() in best_kws:
                        to_extract.append(token["text"].lower())
        elif isinstance(kw_data, tuple):
            if kw_data[1]:
                to_extract.append(kw_data[0]["text"])

    return Counter(to_extract)


class TesterModel(BaseModel):
    """
        Just extract the best scoring KWs for testing
    """

    def __init__(self, exp_dir, params={},
                 train_data_filename="feature_data.json.gz",
                 test_data_filename="feature_data_test.json.gz"):
        """
        """
        super(TesterModel).__init__(exp_dir, params, train_data_filename, test_data_filename)
        self.classifier = None
        self.use_weights = False

    def loadData(self):
        pass

    def postProcessLoadedData(self):
        pass

    def testModel(self, params={}):
        """
            Test the trained extractor on a test set
        """
        self.reader = FeaturesReader(self.test_data_filename)
        self.testing_contexts = [c for c in self.reader]

        context_tokens = []

        for context in self.testing_contexts:
            context_tokens.append([t["text"].lower() for t in context["tokens"]])

        # self.testing_contexts = filterAndAddFeatures(self.testing_contexts)

        self.X_test, self.y_test = getTrainTestData(self.testing_contexts, return_weight=self.use_weights)
        self.y_test = flattenList(self.y_test)

        all_y_predicted = []

        result_contexts = []

        for ctx_index, context in enumerate(self.testing_contexts):

            terms_to_extract = {t[0]: t[1] for t in context["best_kws"]}

            # from collections import Counter
            # counter = Counter(terms_to_extract)
            # print("Terms to extract:",sorted(counter.items(), key=lambda x:x[1],reverse=True))
            #
            # print("True in labels:",(True in labels))
            # print(features)
            predicted = []

            for token_text in context_tokens[ctx_index]:
                if token_text in terms_to_extract:
                    if self.use_weights:
                        predicted.append(terms_to_extract[token_text])
                    else:
                        predicted.append(True)
                else:
                    if self.use_weights:
                        predicted.append(0)
                    else:
                        predicted.append(False)

            all_y_predicted.append(predicted)

            result_context = makeEvaluationQuery(context,
                                                 context_tokens[ctx_index],
                                                 predicted,
                                                 use_weight=self.use_weights)
            result_contexts.append(result_context)

        all_y_predicted = flattenList(all_y_predicted)
        if not self.use_weights:
            self.printClassificationResults(self.y_test, all_y_predicted)

        self.savePredictions(result_contexts)

    def printClassificationResults(self, y_test, predicted):
        print("Classification report for classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(y_test, predicted)))
        print("AUC:\n%f" % metrics.roc_auc_score(y_test, predicted, average="weighted"))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))

    def informativeFeatures(self, features, fold=0):
        pass

    def defineModel(self):
        pass

    def trainModel(self):
        """
            Train an extractor
        """
        pass

    def trainSingleModel(self):
        pass

    def savePredictions(self, result_contexts):
        with open(os.path.join(self.exp_dir, "predictions.json"), "w") as f:
            json.dump(result_contexts, f)
            # for context in result_contexts:
            #     f.write(json.dumps(context))


def main():
    params = {
        # "class_weight": "balanced_subsample",
        "class_weight": {True: 1000,
                         False: 0.001},
    }
    model = TesterModel("/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace")
    model.use_weights = True
    model.run(external_test=True)

    pass


if __name__ == '__main__':
    main()
