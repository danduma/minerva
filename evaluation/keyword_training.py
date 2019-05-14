# Keyword trainer. Uses extractors from ml.keyword_extraction
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function

import gc
import json
import os
from collections import defaultdict, Counter

import ml.keyword_extraction as extractors
from db.result_store import OfflineResultReader, ResultIncrementalReader
from evaluation.pipeline_functions import getDictOfTestingMethods
from ml.keyword_support import saveFeatureData, saveAllKeywordSelectionTrace, getMatrixFromTokenList, saveMatrix, loadMatrix, loadFeatureData, saveOfflineKWSelectionTraceToCSV
from proc.results_logging import ResultsLogger
from six.moves import range
from sklearn import model_selection

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
                    if token["text"] in best_kws:
                        to_extract.append(token["text"])
        elif isinstance(kw_data, tuple):
            if kw_data[1]:
                to_extract.append(kw_data[0]["text"])

    return Counter(to_extract)


class KeywordTrainer(object):
    """
        This class encapsulates the training and testing of a keyword extractor,
        using k-fold cross-validation
    """

    def __init__(self, exp, options):
        """
        """
        self.exp = exp
        self.options = options
        self.all_doc_methods = {}
        self.extractor_class = getattr(extractors, self.exp["keyword_extractor_class"])

        features_filename = os.path.join(self.exp["exp_dir"], "feature_data.json.gz")
        features_numpy_filename = os.path.join(self.exp["exp_dir"], "feature_data.npy.gz")

        if self.options.get("run_package_features", False) or not os.path.isfile(features_filename):
            print("Prepackaging features...")

            saveOfflineKWSelectionTraceToCSV("kw_data", os.path.join(self.exp["exp_dir"], "cache"), self.exp["exp_dir"])

            self.reader = saveFeatureData(self.reader, features_filename)
            self.matrix = getMatrixFromTokenList(self.reader)
            saveMatrix(features_numpy_filename, self.matrix)
        ##            print("\n After saving feature data:",listAllKeywordsToExtractFromReader(self.reader))
        else:
            print("Loading prepackaged features...")
            self.reader = loadFeatureData(features_filename)
            self.matrix = loadMatrix(features_numpy_filename)

        max_data_points = self.exp.get("max_data_points", len(self.reader))
        if max_data_points > len(self.reader):
            self.reader = self.reader[:max_data_points]
            self.matrix.resize((max_data_points, self.matrix.shape[1]))
        self.scores = []

    def trainExtractor(self, current_split_fold):
        """
            Train an extractor for the given fold
        """
        ##        all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])
        ##        annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"] in ["annotated_boost"]]

        numfolds = self.exp.get("cross_validation_folds", 4)

        if len(self.reader) == 0:
            raise ValueError("No precomputed formulas")

        if len(self.reader) < numfolds:
            raise ValueError("Number of results is smaller than number of folds")

        kf = model_selection.KFold(n_splits=numfolds,
                                   shuffle=False,
                                   random_state=None)  # indices=True, k=None
        train_folds = []
        test_folds = []
        # for train_index, test_index in kf.split(range(len(self.reader))):
        for train_index, test_index in kf.split(self.reader):
            train_folds.append(train_index)
            test_folds.append(test_index)

        traincv = train_folds[current_split_fold]
        testcv = test_folds[current_split_fold]

        if isinstance(self.reader, ResultIncrementalReader):
            train_set = self.reader.subset(traincv)
            test_set = self.reader.subset(testcv)
        elif isinstance(self.reader, list):
            train_set = [self.reader[i] for i in traincv]
            test_set = [self.reader[i] for i in testcv]
        else:
            raise ValueError("Unkown class of results")
        ##        print("train_set: ",listAllKeywordsToExtractFromReader(train_set))
        ##        print("test_set: ",listAllKeywordsToExtractFromReader(test_set))

        if len(train_set) == 0:
            print("Training set len is 0!")
            return defaultdict(lambda: 1)

        print("Training for %d/%d tokens " % (len(train_set), len(self.reader)))

        # what to do with the runtime_parameters?
        ##            all_doc_methods[method]["runtime_parameters"]=weights
        trained_model = self.extractor_class(self.exp.get("keyword_extractor_subclass", None),
                                             self.exp.get("keyword_extractor_parameters", {}),
                                             fold=current_split_fold,
                                             exp=self.exp)
        trained_model.learnFeatures(self.reader, self.exp.get("keyword_ignore_features", []))
        trained_model.train(train_set)
        self.scores.append(trained_model.test(test_set))

        return trained_model

    def trainExtractors(self):
        """
            Run the final stage of the training pipeline
        """
        gc.collect()
        options = self.options
        self.all_doc_methods = getDictOfTestingMethods(self.exp["doc_methods"])

        trained_extractors = {}
        if options.get("override_folds", None):
            self.exp["cross_validation_folds"] = options["override_folds"]

        if options.get("override_metric", None):
            self.exp["metric"] = options["override_metric"]

        self.reader = self.extractor_class.processReader(self.reader)

        numfolds = self.exp.get("cross_validation_folds", 4)

        # First we train a keyword extractor from each fold's training set
        for split_fold in range(numfolds):
            print("\nFold #" + str(split_fold))
            trained_extractors[split_fold] = self.trainExtractor(split_fold)

        # TODO: actually test the kws selected by trained extractors in retrieval; see how much it actually helps the task
        #


##        print("Now applying and testing keywords...\n")
##        self.measureScoresOfExtractors(trained_extractors)


def main():
    logger = ResultsLogger(False, False)
    ##    logger.addResolutionResultDict
    pass


if __name__ == '__main__':
    main()
