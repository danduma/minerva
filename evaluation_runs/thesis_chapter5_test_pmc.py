# Reimplementation of the equivalent ACL14 experiments for Chapter 3 of thesis
#
# Copyright:   (c) Daniel Duma 2016-18
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


from __future__ import print_function

from __future__ import absolute_import

from evaluation.experiment import Experiment
from db.ez_connect import ez_connect

# BOW files to prebuild for generating document representation.
from proc.nlp_functions import CORESC_LIST

prebuild_bows = {
    # "full_text": {"function": "getDocBOWfull", "parameters": [1]},
    # "title_abstract": {"function": "getDocBOWTitleAbstract", "parameters": [1]},
    # "passage":{"function":"getDocBOWpassagesMulti", "parameters":[150,175,200,250,300,350,400,450]},
    # "inlink_context": {"function": "getDocBOWInlinkContextCache",
    #                    "parameters": [
    #                        30, 40, 50, 100,
    #                        "1only",
    #                        "paragraph",
    #                        "1up",
    #                        "0up_1down",
    #                        "1up_1down",
    #                        "2up_2down"
    #                    ]},
    ##"ilc_AZ":{"function":"generateDocBOW_ILC_Annotated", "parameters":["paragraph","1up_1down","1up","1only"] },

    # "ilc_annotated": {"function": "generateDocBOW_ILC_Annotated",
    #                   "parameters":
    #                       [
    #                         # "2up_2down",
    #                        # "1up_1down",
    #                        "paragraph"]},

    "az_annotated": {"function": "getDocBOWannotated", "parameters": [1]},
    ##"section_annotated":{"function":"getDocBOWannotatedSections", "parameters":[1]},
}

# bow_name is just about the name of the file containing the BOWs
prebuild_indeces = {

}

prebuild_general_indexes = {
    # "az_annotated_pmc_2013": {"type": "standard_multi",
    #                           "bow_name": "az_annotated",  # bow to load
    #                           "parameters": [1],  # parameter has to match a parameter of a prebuilt bow
    #                           "max_year": 2013  # cut-off point for adding files to index
    #                           },

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
    "az_annotated": {"type": "annotated_boost",
                     "index": "az_ilc_az_annotated_pmc_2013_1",
                     "parameters": ["paragraph"],
                     "runtime_parameters": {
                         "ALL": CORESC_LIST
                     },
                     "preset_runtime_weights_files": ["pmc_weights_full_text_all1.csv"],
                     # "preset_runtime_weights_files": ["pmc_weights_full_text.csv"],
                     # "multi_match_type": "cross_fields"
                     },

    # "ilc_az_annotated": {"type": "annotated_boost",
    #                          "index": "az_ilc_az_annotated_pmc_2013_1",
    #                          "parameters": ["paragraph"],
    #                          "runtime_parameters": {
    #                              "ALL": ["ilc_CSC_"+ zone for zone in CORESC_LIST]
    #                          },
    #                          # "preset_runtime_weights_files": ["pmc_weights_ilc_all1.csv"],
    #                          "preset_runtime_weights_files": ["pmc_weights_ilc.csv"],
    #                          # "multi_match_type": "cross_fields"
    #                          },

    # "ilc_az_annotated_combined": {"type": "annotated_boost",
    #                               "index": "az_ilc_az_annotated_pmc_2013_1",
    #                               "parameters": ["paragraph"],
    #                               "runtime_parameters": {
    #                                   "ALL": CORESC_LIST + ["ilc_AZ_" + zone for zone in CORESC_LIST]
    #                               },
    #                               # "preset_runtime_weights_files": ["pmc_weights_ilc_all1.csv"],
    #                               "preset_runtime_weights_files": ["pmc_weights_ilc.csv",
    #                                                                "pmc_weights_full_text.csv"],
    #                               # "multi_match_type": "cross_fields"
    #                               },
}

# this is the dict of query extraction methods
qmethods = {
    "sentence": {"parameters": [
        ##                "1only",
        ##                "paragraph",
        ##                "1up",
        ##                "0up_1down",
        "1up_1down",
        ##                "2up_2down"
    ],
        "method": "Sentences",
    },

}

experiment = {
    "name": "thesis_chapter5_test_pmc",
    "description":
        """Reimplement the LREC & WOSP experiments to test weights for PMC""",
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
    # "test_files_condition": "metadata:\"metadata.num_in_collection_references:>0\" AND metadata:\"metadata.year:>2013\"",
    # "test_files_condition": "metadata:\"metadata.num_in_collection_references:>\\\"0\\\"\" AND metadata:\"metadata.year:>\\\"2013\\\"\"",
    # "test_files_condition": [{"range": {"metadata.year": {"gt": 2013}}},
    #                          {"range": {"metadata.num_in_collection_references": {"gt": 0}}}],
    # how to sort test files
    "test_files_sort": "metadata.num_in_collection_references:desc",
    # This lets us pick just the first N files from which to generate queries
    "max_test_files": 1000,
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
    "type": "test_weights",
    # If full_corpus, this is the cut-off year for including documents in the general index.
    # In this way we can separate test files and retrieval files.
    "index_max_year": 2013,
    "numchunks": 10,
    # name of CSV file to save results in
    "output_filename": "results.csv",
    "pivot_table": "",
    "max_results_recall": 200,
    # should queries be classified based on some rhetorical class of the sentence: "az", "csc", "query_method"
    "queries_classification": "query_method",
    # do not process more than this number of queries of the same type (type on line above)
    "max_per_class_results": 1000,
    # do not generate queries for more than this number of citations
    "max_queries_generated": 1000,
    # of all precomputed queries, which classes should be processed/evaluated?
    "queries_to_process": ["ALL"],
    # what "zones" to try to train weights for
    "train_weights_for": None,
    # add another doc_method showing the score based on analytical random chance?
    "add_random_control_result": False,
    "precomputed_queries_filename": "precomputed_queries.json",
    "files_dict_filename": "files_dict.json",
    # how many folds to use for training/evaluation
    "cross_validation_folds": 4,

    # "fixed_runtime_parameters": {"_full_text": 1},
    # "retrieval_class": "ElasticRetrievalBoostTweaked8",
    "retrieval_class": "ElasticRetrievalBoost",

    "filter_options_resolvable": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # None == no limit
        # What's the max year for considering a citation? Should match index_max_year above
        # "max_year": 2013,
    },

    "filter_options_ilc": {
        # Should ILC_BOWS exclude papers that have the same first author as the TARGET document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # How many authors can the citing and cited paper maximally overlap on?
        # What's the max year for considering a citation? Should match index_max_year above
        "max_year": 2013,
    }

}

options = {
    "run_prebuild_bows": 1,  # should the whole BOW building process run?
    "overwrite_existing_bows": 1,  # if a BOW exists already, should we overwrite it?
    "build_indexes": 1,  # build indices?
    "force_recreate_indexes": 0,  # force rebuild indices?
    "generate_queries": 0,  # precompute the queries?
    "force_regenerate_resolvable_citations": 0,  # find again the resolvable citations in a file?
    "overwrite_existing_queries": 0,  # force rebuilding of queries too?

    "clear_existing_prr_results": 1,  # delete previous precomputed results? i.e. start from scratch
    "run_precompute_retrieval": 0,
    # only applies if type == "train_weights" or "extract_kw". This is necessary for run_feature_annotation! And this is because each pipeline may do different annotation
    "run_feature_annotation": 0,
    # annotate scidocs with document-wide features for keyword extraction? By default, False
    "refresh_results_cache": 1,  # should we clean the offline reader cache and redownload it all from elastic?
    "run_experiment": 0,  # must be set to 1 for "run_package_features" to work
    "run_package_features": 0,
    # should we read the cache and repackage all feature information, or use it if it exists already?
    "run_query_start_at": 0,
    "bow_building_start_at": 10273,
    "index_start_at": 0,
    "list_missing_files": 1,

    # "max_files_to_process": 330000,  # FOR QUICK TESTING
}


def main():
    corpus = ez_connect("PMC_CSC", "aws-server")
    # corpus.closeIndex("idx_*")
    exp = Experiment(experiment, options, use_celery=False)
    exp.run()
    pass


if __name__ == '__main__':
    main()
