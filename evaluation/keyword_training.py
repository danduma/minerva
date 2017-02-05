# Keyword trainer
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import gc, random, os, sys, json
from copy import deepcopy
from collections import defaultdict
from sklearn import cross_validation
import pandas as pd

import minerva.db.corpora as cp
from minerva.db.result_store import OfflineResultReader, ResultIncrementalReader
from minerva.evaluation.base_pipeline import getDictOfTestingMethods
from minerva.evaluation.precompute_functions import precomputeFormulas
from minerva.evaluation.weight_functions import runPrecomputedQuery
import minerva.ml.keyword_extraction as extractors
from minerva.proc.results_logging import ResultsLogger

##from minerva.proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11

GLOBAL_FILE_COUNTER=0

def baseline_score(results):
    """
        Takes some retrieval results and computes the baseline score for them

        TODO store this score from the retrieval?
    """
    pass


def saveFeatureData(precomputed_contexts,path):
    """
        Calls prepareFeatureData() and dumps prepared data to json file
    """
    feature_data=extractors.prepareFeatureData(precomputed_contexts)
    json.dump(feature_data,file(path,"w"))
    return feature_data

class KeywordTrainer(object):
    """
        This class encapsulates the training and testing of a keyword extractor,
        using k-fold cross-validation
    """
    def __init__(self, exp, options):
        """
        """
        self.exp=exp
        self.options=options
        self.all_doc_methods={}
        self.extractor_class=getattr(extractors,self.exp["keyword_extractor_class"])

        features_filename=os.path.join(self.exp["exp_dir"],"feature_data.json")

        if self.options.get("run_package_features", False) or not os.path.isfile(features_filename):
            print("Prepackaging features...")
            self.reader=OfflineResultReader("kw_data", os.path.join(self.exp["exp_dir"], "cache"))
            self.reader=saveFeatureData(self.reader, features_filename)
        else:
            self.reader=json.load(file(features_filename,"r"))
        self.scores=[]

    def trainExtractor(self, split_fold):
        """
            Train an extractor for the given fold
        """
        all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])
        annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"] in ["annotated_boost"]]

        numfolds=self.exp.get("cross_validation_folds",2)

        if len(self.reader) == 0:
            raise ValueError("No precomputed formulas")

        if len(self.reader) < numfolds:
            raise ValueError("Number of results is smaller than number of folds")

        cv = cross_validation.KFold(len(self.reader), n_folds=numfolds, shuffle=False, random_state=None) # indices=True, k=None
        cv=[k for k in cv]

        traincv, testcv=cv[split_fold]
        if isinstance(self.reader, ResultIncrementalReader):
            train_set=self.reader.subset(traincv)
            test_set=self.reader.subset(testcv)
        elif isinstance(self.reader, list):
            train_set=[self.reader[i] for i in traincv]
            test_set=[self.reader[i] for i in testcv]
        else:
            raise ValueError("Unkown class of results")
##            train_set=self.reader.subset(traincv)
##            train_set=[self.reader[i] for i in traincv]
        if len(train_set) == 0:
            print("Training set len is 0!")
            return defaultdict(lambda:1)

        print("Training for %d/%d tokens " % (len(train_set),len(self.reader)))
        trained_models={}

        res={}
        # what to do with the runtime_parameters?
##            all_doc_methods[method]["runtime_parameters"]=weights
        trained_model=self.extractor_class(self.exp.get("keyword_extractor_parameters",{}))
        trained_model.train(train_set)
        self.scores.append(trained_model.test(test_set))

        return trained_model

##    def generateRetrievalResults(self):
##        """
##            Returns the equivalent of loadPrecomputedFormulas but generates those
##            formulas on the fly
##        """
##        for precomputed_query in self.precomputed_queries:
##            retrieval_result=deepcopy(precomputed_query)
##            retrieval_result["doc_method"]=doc_method
##
##            del retrieval_result["query_text"]
##
##            formulas=precomputeFormulas(retrieval_model, precomputed_query, doc_list)
##            retrieval_result["formulas"]=formulas
##
##            for remove_key in ["dsl_query", "lucene_query"]:
##                if remove_key in retrieval_result:
##                    del retrieval_result[remove_key]
##
##            retrieval_result["experiment_id"]=self.exp["experiment_id"]
##
##            yield retrieval_result

    def trainKeywords(self):
        """
            Run the final stage of the training pipeline
        """
        gc.collect()
        options=self.options
        self.all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])

        trained_extractors={}
        if options.get("override_folds",None):
            self.exp["cross_validation_folds"]=options["override_folds"]

        if options.get("override_metric",None):
            self.exp["metric"]=options["override_metric"]

        numfolds=self.exp.get("cross_validation_folds",2)

        # First we train a keyword extractor from each fold's training set
        for split_fold in range(numfolds):
            print("\nFold #"+str(split_fold))
            trained_extractors[split_fold]=self.trainExtractor(split_fold)

##        # Then we actually test them in retrieval
##        print("Now applying and testing keywords...\n")
##        self.measureScoresOfExtractors(trained_extractors)


def main():
    logger=ResultsLogger(False,False)
##    logger.addResolutionResultDict
    pass

if __name__ == '__main__':
    main()
