# Keyword trainer. Uses extractors from ml.keyword_extraction
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pickle

np.warnings.filterwarnings('ignore')

import os
import json
from collections import Counter
import re

# import autosklearn.classification

from sklearn import ensemble, svm, neural_network, linear_model
from sklearn import metrics

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from models.keyword_features import FeaturesReader, getTrainTestData, flattenList, \
    makeEvaluationQuery, normaliseFeatures, getRootDir, plotInformativeFeatures
from models.base_model import BaseModel

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


def get_token(token_list, index, prop):
    if index < 0 or index > (len(token_list) - 1):
        return None
    else:
        if prop not in token_list[index]:
            raise ValueError(prop + " not in token")
        return token_list[index][prop]


def addNGramFeatures(tokens, features, ngram_len=3):
    ngram_len = max(1, ngram_len)

    for index, token in enumerate(tokens):
        for feature in features:
            prevs = [get_token(tokens, index - ng_index, feature) for ng_index in range(1, ngram_len + 1, 1)]
            to_add = {}
            for ng_index in range(1, ngram_len + 1, 1):
                to_add["prev_" + feature + str(ng_index)] = " ".join([str(x) for x in prevs[-ng_index:]])

            token.update(to_add)

    return tokens


def addSurfaceFeatures(tokens):
    for index, token in enumerate(tokens):
        token["has_num"] = (re.search("\d", token["text"]) is not None)
        token["caps_num"] = len(re.findall("[A-Z]", token["text"]))
        token["caps_ratio"] = token["caps_num"] / float(len(token["text"]))
    return tokens


class SKLearnModel(BaseModel):
    """
        This class encapsulates the training and testing of a keyword extractor,
        using k-fold cross-validation
    """

    def __init__(self, exp_dir, params={},
                 train_data_filename="feature_data.json.gz",
                 test_data_filename="feature_data_test.json.gz"):
        """
        """
        super(SKLearnModel, self).__init__(exp_dir, params, train_data_filename, test_data_filename)

        self.subclass_name = params.get("model", "RandomForestClassifier")
        self.subclass = None

        self.interactive = params.get("interactive", False)
        self.params = params.get("sklearn_params", {})
        self.add_ngram_features = params.get("augment_features", False)

        self.model = self._get_model()

        # self.features_to_filter_out = ["lemma", "dep", "pos", "text", "lemma_", "dist_cit_norm", "ent_type",
        #                                "token_type"]
        #
        self.features_to_filter_out = ["lemma", "dep", "pos", "text", "lemma_", "dist_cit_norm", "ent_type",
                                       "token_type", "az", "csc_type"]

    def _get_model(self):
        """
        Creates an instance of the classifier specified in the

        :param params:
        :return:
        """
        for module in [ensemble, svm, neural_network, linear_model]:
            if hasattr(module, self.subclass_name):
                self.subclass = getattr(module, self.subclass_name)
                break

        assert self.subclass, "Extractor class not found " + str(self.subclass_name)

        return self.subclass(**self.params)

    def augmentSingleContext(self, context):
        if self.add_ngram_features:
            addNGramFeatures(context["tokens"], ["pos_", "dep_"], 2)

        addSurfaceFeatures(context["tokens"])
        self.addAffixFeatures(context["tokens"])

    def addAffixFeatures(self, tokens):
        for index, token in enumerate(tokens):
            token["suffix_2"] = ""
            token["suffix_3"] = ""
            token["suffix_4"] = ""

            if token["token_type"] in ["p", "c"] or len(token["text"]) < 4 or token["text"].lower() in self.stopwords:
                continue

            text = token["text"].lower()
            if len(text) > 3:
                suffix2 = text[-2:]
                if suffix2 in self.top2suffixes:
                    token["suffix_2"] = suffix2
            if len(token["text"]) > 4:
                suffix3 = text[-3:]
                if suffix3 in self.top3suffixes:
                    token["suffix_3"] = suffix3
            if len(token["text"]) > 5:
                suffix4 = text[-4:]
                if suffix4 in self.top4suffixes:
                    token["suffix_4"] = suffix4
        return tokens

    def preProcessLoadedData(self):
        suffixes2 = []
        suffixes3 = []
        suffixes4 = []
        for context in self.contexts:
            for token in context["tokens"]:
                if token["token_type"] in ["p", "c"] or len(token["text"]) < 4 or token[
                    "text"].lower() in self.stopwords:
                    continue
                text = token["text"].lower()
                if len(text) > 3:
                    suffixes2.append(token["text"].lower()[-2:])
                if len(text) > 4:
                    suffixes3.append(token["text"].lower()[-3:])
                if len(text) > 5:
                    suffixes4.append(token["text"].lower()[-4:])

        self.suffixes2_counter = Counter(suffixes2)
        self.suffixes3_counter = Counter(suffixes3)
        self.suffixes4_counter = Counter(suffixes4)

        self.top2suffixes = set([x[0] for x in self.suffixes2_counter.most_common(50)])
        self.top3suffixes = set([x[0] for x in self.suffixes3_counter.most_common(50)])
        self.top4suffixes = set([x[0] for x in self.suffixes4_counter.most_common(50)])

        print("Top 2 suffixes", self.top2suffixes)
        print("Top 3 suffixes", self.top3suffixes)
        print("Top 4 suffixes", self.top4suffixes)

    def postProcessLoadedData(self):
        train_val_cutoff = int(.80 * len(self.contexts))

        self.training_contexts = self.contexts[:train_val_cutoff]
        self.validation_contexts = self.contexts[train_val_cutoff:]

        self.X_train, self.y_train = getTrainTestData(self.training_contexts,
                                                      self.regression)
        self.X_val, self.y_val = getTrainTestData(self.validation_contexts,
                                                  self.regression)

        self.X_train = flattenList(self.X_train)
        self.X_val = flattenList(self.X_val)

        self.training_tokens = [t for t in self.X_train]
        self.validation_tokens = [t for t in self.X_val]

        self.y_train = flattenList(self.y_train)
        self.y_val = flattenList(self.y_val)

        print("Vectorizing features...")
        self.X_train = self.dict_vectorizer.transform(self.X_train)
        self.X_val = self.dict_vectorizer.transform(self.X_val)
        print(len(self.X_train[0]), "total features")

        self.MAX_CONTEXT_LEN = max([len(x) for x in self.X_train])

    def testModel(self, params={}):
        """
            Test the trained extractor on a test set
        """
        self.reader = FeaturesReader(self.test_data_filename)
        self.testing_contexts = [c for c in self.reader]
        self.filterStopWordsInData(self.testing_contexts)

        self.processFeatures(self.testing_contexts)
        context_tokens = []

        for context in self.testing_contexts:
            context_tokens.append([t["text"].lower() for t in context["tokens"]])
            self.augmentSingleContext(context)
            new_mask = []
            for k in context["weight_mask"]:
                if k is not None:
                    new_mask.append(k)
                else:
                    new_mask.append(0.0)
            context["weight_mask"] = new_mask

        self.testing_contexts = normaliseFeatures(self.testing_contexts)
        # self.testing_contexts=self.filterOutSingleContextFeatures()

        self.X_test, self.y_test = getTrainTestData(self.testing_contexts,
                                                    self.regression)
        self.y_test = flattenList(self.y_test)

        all_y_predicted = []

        result_contexts = []

        for index, context in enumerate(self.testing_contexts):
            X_test = self.dict_vectorizer.transform(context["tokens"])
            predicted = self.model.predict(X_test)

            predicted = self.filterStopWordsInPredicted(context["tokens"], predicted)
            # print("Extra stopwords removed", self.extra_stopwords_removed)

            all_y_predicted.append(predicted)

            result_context = makeEvaluationQuery(context,
                                                 context_tokens[index],
                                                 predicted,
                                                 use_weight=self.regression)
            result_contexts.append(result_context)

        all_y_predicted = flattenList(all_y_predicted)
        if self.regression:
            self.printRegressionResults(self.y_test, all_y_predicted)
        else:
            self.printClassificationResults(self.y_test, all_y_predicted)

        self.savePredictions(result_contexts)

    def printClassificationResults(self, y_test, predicted):
        print("Classification report for model %s:\n%s\n"
              % (self.model, metrics.classification_report(y_test, predicted)))
        print("AUC:\n%f" % metrics.roc_auc_score(y_test, predicted, average="weighted"))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))

    def printRegressionResults(self, y_test, predicted):
        print("Regression report for model %s:\n%s\n%s\n"
              % (self.model,
                 metrics.mean_squared_error(y_test, predicted),
                 metrics.r2_score(y_test, predicted),
                 ))

    def informativeFeatures(self, features, fold=0):
        """
        Make a plot of the most informative features as reported by the classifier

        :param features:
        :return:
        """
        self.fold = fold
        num_features = min(features.shape[1], 40)

        try:
            importances = self.model.feature_importances_
        except:
            print("Classifier does not support listing feature importance")
            return

        std = np.std([tree.feature_importances_ for tree in self.model.estimators_],
                     axis=0)

        indices = np.argsort(importances)[::-1]
        indices = indices[:num_features]

        # Print the feature ranking
        print("Feature ranking:")

        feature_names = self.dict_vectorizer.get_feature_names()

        to_render = {"importances": importances, "num_features": num_features, "std": std,
                     "feature_names": feature_names, "indices": indices}
        pickle.dump(to_render, open(os.path.join(self.exp_dir, "feature_importances.pickle"), "wb"))

        for f in range(num_features):
            print("%d. %s - %d (%f)" % (f, feature_names[indices[f]], indices[f], importances[indices[f]]))

        plotInformativeFeatures(importances, num_features, std, feature_names, indices, self.corpus_label, self.exp_dir)

    def defineModel(self):
        pass

    def trainModel(self):
        print("Training...")
        self.model.fit(self.X_train, self.y_train)
        self.informativeFeatures(self.X_train)

        predicted = self.model.predict(self.X_val)
        self.filterStopWordsInPredicted(self.validation_tokens, predicted)

        if self.regression:
            print("Validation set:\n\n", metrics.mean_squared_error(self.y_val, predicted))
        else:
            print("Validation set:\n\n", metrics.classification_report(self.y_val, predicted))

        # self.printTestResults(self.y_val, predicted)


def main():
    # Linear regression
    params_lin_reg = {
        "augment_features": False,
        "regression": True,
        "model": "LinearRegression",
        "sklearn_params": {
        },
    }

    # # RandomForestRegressor
    params_random_forest = {
        "augment_features": False,
        "regression": True,
        # "regression": False,
        # "model": "RandomForestRegressor",
        "model": "ExtraTreesRegressor",
        # "model": "LinearRegression",
        # "classifier": "RandomForestClassifier",
        "sklearn_params": {
            "n_estimators": 10,
            # "class_weight": "balanced_subsample",
            # "class_weight": {True: 1000,
            #                  False: 0.1},
            "n_jobs": -1,
            "verbose": 3,
        },
    }

    # MLPRegressor
    params_mlp = {
        "augment_features": False,
        "regression": True,
        "model": "MLPRegressor",
        "sklearn_params": {
            "verbose": 3,
            "hidden_layer_sizes": (300, 200, 100),
            # "activation": "logistic",
            "activation": "relu",
            "learning_rate": "adaptive",
            # "solver": "lbfgs",
            "tol": 1e-5
        },
    }

    # params = params_mlp
    # params = params_lin_reg
    params = params_random_forest

    exp_dir = os.path.join(getRootDir("aac"), "experiments", "aac_generate_kw_trace")
    model = SKLearnModel(exp_dir, params=params,
                         train_data_filename="feature_data_at_w_min1.json.gz",
                         test_data_filename="feature_data_test_at_w.json.gz"
                         )

    # exp_dir = os.path.join(getRootDir("pmc_coresc"), "experiments", "pmc_generate_kw_trace")
    # model = SKLearnModel(exp_dir, params=params,
    #                      train_data_filename="feature_data_at_w_min1.json.gz",
    #                      test_data_filename="feature_data_test_at_w.json.gz",
    #                      )

    model.run(external_test=True)

    pass


if __name__ == '__main__':
    main()
