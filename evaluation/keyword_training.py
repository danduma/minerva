# Keyword trainer. Uses extractors from ml.keyword_extraction
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import gc, os, json
from collections import defaultdict, Counter
from sklearn import cross_validation
import pandas as pd

from minerva.db.result_store import OfflineResultReader, ResultIncrementalReader
from minerva.evaluation.base_pipeline import getDictOfTestingMethods
##from minerva.evaluation.precompute_functions import precomputeFormulas
import minerva.ml.keyword_extraction as extractors
from minerva.proc.results_logging import ResultsLogger

##from minerva.proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11

GLOBAL_FILE_COUNTER=0



def saveFeatureData(precomputed_contexts, path):
    """
        Calls prepareFeatureData() and dumps prepared data to json file
    """
    feature_data=extractors.prepareFeatureData(precomputed_contexts)
    json.dump(feature_data,file(path,"w"))
    return feature_data


def saveKeywordSelectionScores(reader, exp_dir):
    """
        Saves a CSV to measure the performance of keyword selection
    """
    def getScoreDataLine(kw_data):
        """
            Returns a single dict/line for writing to a CSV
        """
        return {
                 "precision_score":kw_data["precision_score"],
                 "mrr_score":kw_data["mrr_score"],
                 "rank":kw_data["rank"],
                 "ndcg_score":kw_data["ndcg_score"],

                 "precision_score_kw":kw_data["kw_selection_scores"]["precision_score"],
                 "mrr_score_kw":kw_data["kw_selection_scores"]["mrr_score"],
                 "rank_kw":kw_data["kw_selection_scores"]["rank"],
                 "ndcg_score_kw":kw_data["kw_selection_scores"]["ndcg_score"],

                 "precision_score_kw_weight":kw_data["kw_selection_weight_scores"]["precision_score"],
                 "mrr_score_kw_weight":kw_data["kw_selection_weight_scores"]["mrr_score"],
                 "rank_kw_weight":kw_data["kw_selection_weight_scores"]["rank"],
                 "ndcg_score_kw_weight":kw_data["kw_selection_weight_scores"]["ndcg_score"],
               }


    lines=[]
    for kw_data in reader:
        lines.append(getScoreDataLine(kw_data))

    data=pd.DataFrame(lines)
    data.to_csv(os.path.join(exp_dir,"kw_selection_scores.csv"))


def saveAllKeywordSelectionTrace(reader, exp_dir):
    """
        Saves a CSV to trace everything that happened, inspect the dataset
    """
    def getScoreDataLine(kw_data):
        """
            Returns a single dict/line for writing to a CSV
        """
        context=u" ".join([s["text"] for s in kw_data["context"]])
        return {
                 "cit_id": kw_data["cit_id"],
                 "context": context,
                 "file_guid":kw_data.get("file_guid",""),
                 "match_guid":kw_data["match_guid"],
                 "best_kws": [kw[0] for kw in kw_data["best_kws"]],

                 "precision_score":kw_data["precision_score"],
                 "mrr_score":kw_data["mrr_score"],
                 "rank":kw_data["rank"],
                 "ndcg_score":kw_data["ndcg_score"],

                 "precision_score_kw":kw_data["kw_selection_scores"]["precision_score"],
                 "mrr_score_kw":kw_data["kw_selection_scores"]["mrr_score"],
                 "rank_kw":kw_data["kw_selection_scores"]["rank"],
                 "ndcg_score_kw":kw_data["kw_selection_scores"]["ndcg_score"],

                 "precision_score_kw_weight":kw_data["kw_selection_weight_scores"]["precision_score"],
                 "mrr_score_kw_weight":kw_data["kw_selection_weight_scores"]["mrr_score"],
                 "rank_kw_weight":kw_data["kw_selection_weight_scores"]["rank"],
                 "ndcg_score_kw_weight":kw_data["kw_selection_weight_scores"]["ndcg_score"],
               }


    lines=[]
    for kw_data in reader:
        lines.append(getScoreDataLine(kw_data))

    data=pd.DataFrame(lines)
    data.to_csv(os.path.join(exp_dir,"kw_selection_trace.csv"), encoding="utf-8")


def listAllKeywordsToExtractFromReader(reader):
    """
        Lists all keywords that are marked as extract:true in a list or reader object
    """
    to_extract=[]

    for kw_data in reader:
        if isinstance(kw_data,dict):
            best_kws={t[0]:t[1] for t in kw_data["best_kws"]}
            for sent in kw_data["context"]:
                for token in sent["token_features"]:
                    if token["text"] in best_kws:
                        to_extract.append(token["text"])
        elif isinstance(kw_data,tuple):
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
        self.exp=exp
        self.options=options
        self.all_doc_methods={}
        self.extractor_class=getattr(extractors,self.exp["keyword_extractor_class"])

        features_filename=os.path.join(self.exp["exp_dir"],"feature_data.json")

        if self.options.get("run_package_features", False) or not os.path.isfile(features_filename):
            print("Prepackaging features...")
            self.reader=OfflineResultReader("kw_data", os.path.join(self.exp["exp_dir"], "cache"))
##            print("\n From OfflineResultReader:",listAllKeywordsToExtractFromReader(self.reader))
##            saveKeywordSelectionScores(self.reader, self.exp["exp_dir"])
            saveAllKeywordSelectionTrace(self.reader, self.exp["exp_dir"])
##            print("\n After saving kw selection data:",listAllKeywordsToExtractFromReader(self.reader))
            self.reader=saveFeatureData(self.reader, features_filename)
##            print("\n After saving feature data:",listAllKeywordsToExtractFromReader(self.reader))
        else:
            print("Loading prepackaged features...")
            self.reader=json.load(file(features_filename,"r"))

        self.reader=self.reader[:self.exp.get("max_data_points",len(self.reader))]
        self.scores=[]

    def trainExtractor(self, split_fold):
        """
            Train an extractor for the given fold
        """
##        all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])
##        annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"] in ["annotated_boost"]]

        numfolds=self.exp.get("cross_validation_folds",4)

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

##        print("train_set: ",listAllKeywordsToExtractFromReader(train_set))
##        print("test_set: ",listAllKeywordsToExtractFromReader(test_set))

        if len(train_set) == 0:
            print("Training set len is 0!")
            return defaultdict(lambda:1)

        print("Training for %d/%d tokens " % (len(train_set),len(self.reader)))

        # what to do with the runtime_parameters?
##            all_doc_methods[method]["runtime_parameters"]=weights
        trained_model=self.extractor_class(self.exp.get("keyword_extractor_parameters",{}), fold=split_fold, exp=self.exp)
        trained_model.learnFeatures(self.reader)
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

    def trainExtractors(self):
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

        numfolds=self.exp.get("cross_validation_folds",4)

        # First we train a keyword extractor from each fold's training set
        for split_fold in range(numfolds):
            print("\nFold #"+str(split_fold))
            trained_extractors[split_fold]=self.trainExtractor(split_fold)

        # TODO: actually test trained extractors in retrieval
##        print("Now applying and testing keywords...\n")
##        self.measureScoresOfExtractors(trained_extractors)


def main():
    logger=ResultsLogger(False,False)
##    logger.addResolutionResultDict
    pass

if __name__ == '__main__':
    main()
