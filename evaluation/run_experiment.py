# main entry point for the running of the experiments
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from minerva.db.corpora import Corpus
from prebuild import prebuildBOWsForTests,prebuildLuceneIndexes,buildGeneralLuceneIndex
from testing_functions import *
from results_analysis import saveGraphForResults, makeAllGraphsForExperiment

import context_extract

import gc,sys,json

class Experiment:
    """
        Encapsulates an experiment's parameters:
    """
    def __init__(self,doc_methods,qmethods):
        self.doc_methods=doc_methods
        self.qmethods=qmethods


def runExperiment(exp_name,options):
    """
        Loads a JSON describing an experiment and runs it all
    """
    if not Corpus.experimentExists(exp_name):
        raise IOError("Experiment file does not exist for "+exp_name)
        return

    exp=Corpus.loadExperiment(exp_name)
    ensureDirExists(exp["exp_dir"])

    for option in exp["prebuild_bows"]:
        exp["prebuild_bows"][option]["function"]=getattr(context_extract, exp["prebuild_bows"][option]["function"])

    for option in exp["qmethods"]:
        exp["qmethods"][option]["function"]=getattr(context_extract, exp["qmethods"][option]["function"])

    if len(exp["test_files"]) > 0:
        Corpus.TEST_FILES=exp["test_files"]
    else:
        if exp["test_files_condition"] == "":
            raise ValueError("No test_files specified or test_files_condition")
            return
        Corpus.TEST_FILES=Corpus.listPapers(exp["test_files_condition"])
        exp["test_files"]=Corpus.TEST_FILES

    if exp["full_corpus"]:
        prebuild_list=Corpus.listAllPapers()
    else:
        prebuild_list=Corpus.listIncollectionReferencesOfList(Corpus.TEST_FILES)
        prebuild_list.extend(Corpus.TEST_FILES)

    if options["run_prebuild_bows"] and len(exp["prebuild_bows"]) > 0:
        prebuildBOWsForTests(exp["prebuild_bows"],FILE_LIST=prebuild_list, force_prebuild=options["force_prebuild"])

    if not exp["full_corpus"]:
        if options["run_prebuild_indexes"] and len(exp["prebuild_indexes"]) > 0:
            prebuildLuceneIndexes(Corpus.TEST_FILES, exp["prebuild_indexes"])
    else:
        if options["run_prebuild_indexes"] and len(exp["prebuild_general_indexes"]) > 0:
            buildGeneralLuceneIndex(Corpus.TEST_FILES,exp["prebuild_general_indexes"])


    gc.collect()
    if options["run_precompute_queries"] or not exists(exp["exp_dir"]+"queries_by_"+exp["queries_classification"]+".json"):
        precomputeQueries(exp)

    # testing pipeline
    if exp["type"] == "compute_once":
        runPrecomputedCitationResolutionLucene(exp)
    elif exp["type"] == "train_weights":
        gc.collect()

        if options["run_precompute_retrieval"] or not exists(exp["exp_dir"]+"prr_"+exp["queries_classification"]+"_"+exp["train_weights_for"][0]+".json"):
            precomputeExplainQueries(exp)

        best_weights={}
        if options.get("override_folds",None):
            exp["cross_validation_folds"]=options["override_folds"]

        if options.get("override_metric",None):
            exp["metric"]=options["override_metric"]

        numfolds=exp.get("cross_validation_folds",2)

        for split_fold in range(numfolds):
            print
            print "Fold #"+str(split_fold)
            best_weights[split_fold]=dynamicWeightValues(exp,split_fold)

        print "Now applying and testing weights..."
        print
        measureScores(exp, best_weights)

##        makeAllGraphsForExperiment(exp["exp_dir"])

experiment={
    "name":"exp1",
    "description":"",
    "prebuild_bows":{},
    "prebuild_indexes":{},
    "prebuild_general_indexes":{},
    "doc_methods":{},
    "qmethods":{},
    "test_files":[],
    "test_files_condition":"num_in_collection_references >= 8",
    "use_default_similarity":False,
    "weight_values":[1,3,5],
    "split_set":None,
    "full_corpus":False,
    "type":"compute_once",
    "numchunks":20,
    "output_filename":"results.csv",
    "pivot_table":"",
    "queries_to_process":"ALL",
    "queries_classification":"CSC",
    "train_weights_for":CORESC_LIST,
    "precomputed_queries_filename":"precomputed_queries.json",
    "files_dict_filename":"files_dict.json",
    "cross_validation_folds":2,
    "metric":"avg_mrr",
}

options={
    "run_prebuild_bows":False,
    "force_prebuild":False,
    "run_prebuild_indexes":False,
    "run_precompute_queries":False,
    "run_precompute_retrieval":False,
    "override_folds":4,
    "override_metric":"avg_mrr",
}

##options={
##    "run_prebuild_bows":False,
##    "force_prebuild":False,
##    "run_prebuild_indexes":False,
##    "run_precompute_queries":False,
##    "run_precompute_retrieval":False,
##    "override_folds":4,
##    "override_metric":"avg_mrr",
##}

def main():

    if len(sys.argv) > 1:
        runExperiment(sys.argv[1], options)
        return

##    runExperiment("w20_csc_csc_fa_w0135",options)

##    runExperiment("w_all_all_methods_fa",options)
##    runExperiment("w_all_all_methods_stdsim",options)

##    runExperiment("w20_ilcpar_csc_fa",options)
##    runExperiment("w20_ilcpar_az_az_fa",options)
##    runExperiment("w20_ilcpar_az_az_fa",options)

    runExperiment("w20_az_az_fa",options)
##    runExperiment("w20_az_csc_fa",options)
##    runExperiment("w20_csc_az_fa",options)

    pass

if __name__ == '__main__':
    main()
