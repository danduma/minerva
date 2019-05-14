# Experiments with the ACL corpus like the ones for LREC'16
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function
from __future__ import absolute_import
# from proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST

from evaluation.experiment import Experiment
from db.ez_connect import ez_connect
from kw_evaluation_runs.thesis_settings import CONTEXT_EXTRACTION_C6
import re

# BOW files to prebuild for generating document representation.
prebuild_bows = {
    "full_text": {"function": "getDocBOWfull", "parameters": [1]},

}

# bow_name is just about the name of the file containing the BOWs
prebuild_indeces = {
}

# the name of the index is important, should be unique
prebuild_general_indexes = {
    "az_ilc_az_annotated_pmc_2013": {"type": "ilc_mashup",
                                     "bow_name": "ilc_annotated",  # bow to load
                                     "ilc_method": "ilc_annotated",  # bow to load
                                     "mashup_method": "az_annotated",
                                     "ilc_parameters": ["paragraph"],
                                     # parameter has to match a parameter of a prebuilt bow
                                     "parameters": [1],  # parameter has to match a parameter of a prebuilt bow
                                     "max_year": 2013  # cut-off point for adding files to index
                                     },
}

doc_methods = {
    # "_full_text": {"type": "standard_multi", "index": "az_ilc_az_annotated_pmc_2013_1", "parameters": ["paragraph"],
    #               "runtime_parameters": {"_full_text": 1}}
    "_all_text": {"type": "standard_multi", "index": "az_ilc_az_annotated_pmc_2013_1", "parameters": ["paragraph"],
                  "runtime_parameters": {"_all_text": 1}},
}

# this is the dict of query extraction methods
qmethods = {
    "sentence": {"parameters": [
        CONTEXT_EXTRACTION_C6
    ],
        "method": "Sentences",
    },
}

experiment = {
    "name": "pmc_generate_kw_trace",
    "description":
        "Full-text indexing of pmc to test kw extraction",
    # dict of bag-of-word document representations to prebuild
    "prebuild_bows": prebuild_bows,
    # dict of per-file indexes to prebuild
    "prebuild_indeces": prebuild_indeces,
    # dict of general indexes to prebuild
    "prebuild_general_indexes": prebuild_general_indexes,
    # dictionary of document representation methods to test
    "doc_methods": doc_methods,
    # dictionary of query generation methods to test
    "qmethods": qmethods,
    # list of files in the test set
    "test_files": [],
    # SQL condition to automatically generate the list above
    "test_files_condition": "metadata.num_in_collection_references:>0 AND metadata.year:>2013",
    # how to sort test files
    "test_files_sort": "metadata.num_in_collection_references:desc",
    # This lets us pick just the first N files from which to generate queries
    "max_test_files": 500000,  #
    # "max_test_files": 1000, # for testing
    # Use Lucene DefaultSimilarity? As opposed to FieldAgnosticSimilarity
    "use_default_similarity": True,

    # Annotate sentences with AZ/CoreSC/etc?
    "rhetorical_annotations": [],
    # Run annotators? If False, it is assumed the sentences are already annotated
    "run_rhetorical_annotators": False,
    # Separate queries by AZ/CSC, etc?
    "use_rhetorical_annotation": False,
    ##    "weight_values":[],
    # use full-collection retrival? If False, it runs "citation resolution"
    "full_corpus": True,
    # "compute_once", "train_weights", "test_selectors", "extract_kw", "test_kw_selection"
    "type": "extract_kw",
    # If full_corpus, this is the cut-off year for including documents in the general index.
    # In this way we can separate test files and retrieval files.
    "index_max_year": 2013,
    # how many chunks to split each file for statistics on where the citation occurs
    "numchunks": 10,
    # name of CSV file to save results in
    "output_filename": "results.csv",
    "pivot_table": "",
    "max_results_recall": 200,
    # should queries be classified based on some rhetorical class of the sentence: "az", "csc_type", "" or None
    "queries_classification": "",
    # do not process more than this number of queries of the same type (type on line above)
    "max_per_class_results": 1000,
    # do not generate queries for more than this number of citations

    # "max_queries_generated": 10000000, # for training
    # "max_queries_generated": 1000, # for testing
    # "max_queries_generated": 20, # for PRUEBA

    # of all precomputed queries, which classes should be processed/evaluated?
    "queries_to_process": ["ALL"],
    # what "zones" to try to train weights for
    "train_weights_for": [],  # ["Bac"], ["Hyp","Mot","Bac","Goa","Obj","Met","Exp","Mod","Obs","Res","Con"]
    # add another doc_method showing the score based on analytical random chance?
    "add_random_control_result": False,

    # "precomputed_queries_filename": "precomputed_queries.json",

    # resolvable citations should be at least this level of multi
    # "resolvable_cit_min_multi": 2,

    # "files_dict_filename": "files_dict.json",
    # "files_dict_filename": "files_dict_new1k.json",
    "files_dict_filename": "files_dict_training.json",
    # "files_dict_filename": "files_dict_PRUEBA.json",

    "feature_data_filename": "feature_data.json.gz",

    # what to extract as a citation's context
    "context_extraction": "sentence",
    "context_extraction_parameter": CONTEXT_EXTRACTION_C6,

    # exact name of index to use for extracting idf scores etc. for document feature annotation
    # "features_index_name": "idx_full_text_pmc_2013_1",
    "features_index_name": "idx_az_ilc_az_annotated_pmc_2013_1_paragraph",
    "features_field_name": "_all_text",

    # parameters to keyword selection method
    "keyword_selector": "MultiMaximalSetSelector",
    # "keyword_selector": "AllSelector",
    "keyword_selection_parameters": {},
    # how many folds to use for training/evaluation
    "cross_validation_folds": 4,
    # an upper limit on the number of data points to use
    ##    "max_data_points": 50000,

    "filter_options_resolvable": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # How many authors can the citing and cited paper maximally overlap on?
        # "max_year": 2020,
        # should only match papers in same collection
        "limit_to_same_collection": True,
    },

    "filter_options_ilc": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # How many authors can the citing and cited paper maximally overlap on?
        # What's the max year for considering a citation? Should match index_max_year above
        "max_year": 2013,
        # should only match papers in same collection
        "limit_to_same_collection": True,
    }

}

options = {
    "run_prebuild_bows": 0,  # should the whole BOW building process run?
    "overwrite_existing_bows": 0,  # if a BOW exists already, should we overwrite it?
    "build_indexes": 0,  # rebuild indices?

    "generate_queries": 0,  # precompute the queries?
    "force_regenerate_resolvable_citations": 0,  # find again the resolvable citations in a file?
    "overwrite_existing_queries": 0,  # force rebuilding of queries too?

    "clear_existing_prr_results": 1,  # delete previous precomputed results? i.e. start from scratch
    "run_precompute_retrieval": 1,
    # only applies if type == "train_weights" or "extract_kw". This is necessary for run_feature_annotation! And this is because each pipeline may do different annotation
    "run_feature_annotation": 0,
    # annotate scidocs with document-wide features for keyword extraction? By default, False
    "refresh_results_cache": 1,  # should we clean the offline reader cache and redownload it all from elastic?
    # "run_experiment": 1,
    "run_package_features": 1,
    # should we read the cache and repackage all feature information, or use it if it exists already?
    # "run_query_start_at": 2211,
    "max_queries_to_process": 10000,
    "list_missing_files": 1,
}


def main():
    global options, experiment
    corpus = ez_connect("PMC_CSC", "koko")

    # to_generate = "training_min2"
    to_generate = "training_min1"
    # to_generate = "testing"
    # to_generate = "PRUEBA"
    # to_generate = "export"

    # experiment["keyword_selector"] = "MultiMaximalSetSelector"
    # suffix = "mms"

    experiment["keyword_selector"] = "AllSelector"
    suffix="w"

    if to_generate.startswith("training"):
        experiment["max_queries_generated"] = 10000
        experiment["max_test_files"] = 500000

        # experiment["test_files_condition"] = "metadata.num_in_collection_references:>0 "
        # experiment["features_field_name"] = "_full_text"

        experiment["test_files_condition"] = "metadata.num_in_collection_references:>0 AND metadata.year:>2013"
        experiment["features_field_name"] = "_all_text"

        # experiment["test_guids_to_ignore_file"] = "test_guids_pmc_c3_1k.txt"
        experiment["load_preselected_test_files_list"] = "train_guids_pmc_c6.txt"

        match = re.search(r"_min(\d)", to_generate)
        if match:
            min_multi = int(match.group(1))
            experiment["resolvable_cit_min_multi"] = min_multi
            experiment["precomputed_queries_filename"] = "precomputed_queries_training_min%d.json" % min_multi
            experiment["files_dict_filename"] = "files_dict_training_min%d.json" % min_multi
            experiment["feature_data_filename"] = "feature_data_at_%s_min%d.json.gz" % (suffix, min_multi)
        else:
            experiment["resolvable_cit_min_multi"] = 1
            experiment["precomputed_queries_filename"] = "precomputed_queries_training.json"
            experiment["files_dict_filename"] = "files_dict_training.json"
            experiment["feature_data_filename"] = "feature_data_at_%s.json.gz" % suffix

    elif to_generate == "testing":
        experiment["name"] = "pmc_generate_kw_trace_TEST"
        experiment["max_queries_generated"] = 1000
        # experiment["load_preselected_test_files_list"] = "test_guids.txt"
        experiment["load_preselected_test_files_list"] = "test_guids_pmc_c3_1k.txt"
        experiment["precomputed_queries_filename"] = "precomputed_queries_test.json"
        experiment["files_dict_filename"] = "files_dict_test.json"
        experiment["feature_data_filename"] = "feature_data_test_at_%s.json.gz" % suffix
        # experiment["resolvable_cit_min_multi"]= 1
        experiment["max_test_files"] = 1000
        experiment["test_files_condition"] = "metadata.num_in_collection_references:>0 AND metadata.year:>2013"

    elif to_generate == "export":
        experiment["precomputed_queries_filename"] = "precomputed_queries_training_min1.json"
        experiment["feature_data_filename"] = "feature_data_at_%s_min1.json.gz" % suffix
        options["clear_existing_prr_results"] = 0
        options = {
            "run_prebuild_bows": 0,  # should the whole BOW building process run?
            "overwrite_existing_bows": 0,  # if a BOW exists already, should we overwrite it?
            "build_indexes": 0,  # rebuild indices?
            "generate_queries": 0,  # precompute the queries?
            "force_regenerate_resolvable_citations": 0,  # find again the resolvable citations in a file?
            "overwrite_existing_queries": 0,  # force rebuilding of queries too?
            # delete previous precomputed results? i.e. start from scratch
            "run_precompute_retrieval": 0,
            "run_feature_annotation": 0,
            "refresh_results_cache": 1,  # should we clean the offline reader cache and redownload it all from elastic?
            "run_package_features": 1,
            "list_missing_files": 0,
        }

    exp = Experiment(experiment, options, False)
    exp.run()


if __name__ == '__main__':
    main()
