# Evaluate the generated predictions
#
# Copyright:   (c) Daniel Duma 2016-18
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import

from evaluation.experiment import Experiment
from db.ez_connect import ez_connect

from proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST

# BOW files to prebuild for generating document representation.
prebuild_bows = {
}

# bow_name is just about the name of the file containing the BOWs
prebuild_indeces = {
}

prebuild_general_indexes = {
}

doc_methods_aac = {
    "ilc_az_annotated": {"type": "annotated_boost",
                         "index": "az_ilc_az_annotated_aac_2010_1",
                         "parameters": ["paragraph"],
                         "runtime_parameters": {
                             # "ALL": ["ilc_AZ_" + zone for zone in AZ_ZONES_LIST]
                             "ALL": ["_all_text"]
                             # "ALL": ["_full_text"]
                         },
                         # "preset_runtime_weights_files": ["aac_weights_ilc_all1.csv"],
                         # "preset_runtime_weights_files": ["aac_weights_ilc.csv"],
                         # "multi_match_type": "cross_fields"
                         },
}

doc_methods_pmc = {
    "az_annotated": {"type": "annotated_boost",
                     "index": "az_ilc_az_annotated_pmc_2013_1",
                     "parameters": ["paragraph"],
                     "runtime_parameters": {
                         # "ALL": CORESC_LIST,
                         "ALL": ["_all_text"]
                         # "ALL": ["_full_text"]
                     },
                     "preset_runtime_weights_files": ["pmc_weights_full_text_all1.csv"],
                     # "preset_runtime_weights_files": ["pmc_weights_full_text.csv"],
                     # "multi_match_type": "cross_fields"
                     },
}

# this is the dict of query extraction methods
qmethods = {
}

experiment = {
    "name": None,
    "description":
        """ """,
    # dict of bag-of-word document representations to prebuild
    "prebuild_bows": prebuild_bows,
    # dict of per-file indexes to prebuild
    "prebuild_indeces": prebuild_indeces,
    # dict of general indexes to prebuild
    "prebuild_general_indexes": prebuild_general_indexes,
    # dictionary of document representation methods to test
    "doc_methods": {},
    # dictionary of query generation methods to test
    "qmethods": qmethods,
    # list of files in the test set
    "test_files": [],
    # SQL condition to automatically generate the list above
    "test_files_condition": "",

    "test_files_sort": "metadata.num_in_collection_references:desc",
    # This lets us pick just the first N files from which to generate queries
    "max_test_files": 10000,
    # Use Lucene DefaultSimilarity? As opposed to FieldAgnosticSimilarity
    "use_default_similarity": True,
    # Annotate sentences with AZ/CoreSC/etc?
    "rhetorical_annotations": [],
    "weight_values": [],
    # ?
    "split_set": None,
    # use full-collection retrival? If False, it runs "citation resolution"
    "full_corpus": True,
    # "compute_once","train_weights"
    "type": "compute_once",
    # If full_corpus, this is the cut-off year for including documents in the general index.
    # In this way we can separate test files and retrieval files.
    "index_max_year": 2010,
    "numchunks": 10,
    # name of CSV file to save results in
    "output_filename": "results.csv",
    "pivot_table": "",
    "max_results_recall": 200,
    # should queries be classified based on some rhetorical class of the sentence: "az", "csc", "query_method"
    # "queries_classification": "query_method",
    # do not process more than this number of queries of the same type (type on line above)
    "max_per_class_results": 1000,
    # do not generate queries for more than this number of citations
    "max_queries_generated": 1000,
    # ignore all GUIDs contained in this file
    # "test_guids_to_ignore_file": "ignore_guids_aac_c3.json",
    # of all precomputed queries, which classes should be processed/evaluated?
    "queries_to_process": ["ALL"],
    # what "zones" to try to train weights for
    "train_weights_for": None,
    # add another doc_method showing the score based on analytical random chance?
    "add_random_control_result": False,

    "files_dict_filename": "files_dict_new1k.json",
    # how many folds to use for training/evaluation
    "cross_validation_folds": 4,

    "filter_options_resolvable": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # None == no limit
        # What's the max year for considering a citation? Should match index_max_year above
        # "max_year": 2010,
    },

    "filter_options_ilc": {
        # Should ILC_BOWS exclude papers that have the same first author as the TARGET document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # How many authors can the citing and cited paper maximally overlap on?
        # What's the max year for considering a citation? Should match index_max_year above
        "max_year": 2010,
    },

}

options = {
    "run_prebuild_bows": 0,  # should the whole BOW building process run?
    "overwrite_existing_bows": 0,  # if a BOW exists already, should we overwrite it?
    "build_indexes": 0,  # rebuild indices?
    "force_recreate_indexes": 0,  # force rebuild indices?
    "generate_queries": 0,  # precompute the queries?
    "force_regenerate_resolvable_citations": 0,  # find again the resolvable citations in a file?
    "overwrite_existing_queries": 0,  # force rebuilding of queries too?

    "clear_existing_prr_results": 0,  # delete previous precomputed results? i.e. start from scratch
    "run_precompute_retrieval": 0,
    # only applies if type == "train_weights" or "extract_kw". This is necessary for run_feature_annotation! And this is because each pipeline may do different annotation
    "run_feature_annotation": 0,
    # annotate scidocs with document-wide features for keyword extraction? By default, False
    "refresh_results_cache": 0,  # should we clean the offline reader cache and redownload it all from elastic?
    "run_experiment": 1,  # must be set to 1 for "run_package_features" to work
    "run_package_features": 0,
    # should we read the cache and repackage all feature information, or use it if it exists already?
    "run_query_start_at": 0,
    "list_missing_files": 0,

    # "max_files_to_process": 100,  # FOR QUICK TESTING
}

SERVER="aws-server"


def main(corpus_label, exp_name, precomputed_queries_filename="predictions.json"):
    experiment["precomputed_queries_filename"] = precomputed_queries_filename

    if corpus_label.lower() == "aac":
        corpus = ez_connect("AAC", SERVER)
        experiment["name"] = exp_name
        experiment["load_preselected_test_files_list"] = "test_guids_c3_aac.txt"
        experiment["test_files_condition"] = "metadata.num_in_collection_references:>0 AND metadata.year:>2010"
        experiment["doc_methods"] = doc_methods_aac

    elif corpus_label.lower() == "pmc":
        corpus = ez_connect("PMC_CSC", SERVER)

        experiment["files_dict_filename"] = "files_dict_test.json"
        # experiment["files_dict_filename"] = "precomputed_queries_test.json"
        experiment["name"] = exp_name
        experiment["load_preselected_test_files_list"] = "test_guids_pmc_c3_1k.txt"
        experiment["test_files_condition"] = "metadata.num_in_collection_references:>0 AND metadata.year:>2013"
        experiment["doc_methods"] = doc_methods_pmc

    # corpus.closeIndex("idx_*")

    exp = Experiment(experiment, options, use_celery=False, process_command_line=False)
    print("exp_dir: ", exp.exp["exp_dir"])
    print("Query file: ", precomputed_queries_filename)
    exp.run()
    pass


if __name__ == '__main__':
    import plac

    plac.call(main)
