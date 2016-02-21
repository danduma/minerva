# main entry point for the running of the experiments
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from minerva.evaluation.experiment import Experiment

def runExperiment(experiment,options):
    """
    """


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
    "train_weights_for":"",
    "precomputed_queries_filename":"precomputed_queries.json",
    "files_dict_filename":"files_dict.json",
    "cross_validation_folds":2,
    "metric":"avg_mrr",
}

options={
    "run_prebuild_bows":False,
    "force_prebuild":False,
    "rebuild_indexes":False,
    "recompute_queries":False,
    "run_precompute_retrieval":False,
    "override_folds":4,
    "override_metric":"avg_mrr",
}

##options={
##    "run_prebuild_bows":False,
##    "force_prebuild":False,
##    "rebuild_indexes":False,
##    "recompute_queries":False,
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

##    runExperiment("w20_az_az_fa",options)
##    runExperiment("w20_az_csc_fa",options)
##    runExperiment("w20_csc_az_fa",options)

    pass

if __name__ == '__main__':
    main()
