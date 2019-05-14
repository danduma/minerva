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

# BOW files to prebuild for generating document representation.
prebuild_bows = {
    "full_text": {"function": "getDocBOWfull", "parameters": [1]},

}

# bow_name is just about the name of the file containing the BOWs
prebuild_indeces = {
}

# the name of the index is important, should be unique
prebuild_general_indexes = {
    # "full_text_aac_2010": {"type": "standard_multi", "bow_name": "full_text", "parameters": [1]},

    "az_ilc_az_annotated_aac_2010":
        {"type": "ilc_mashup",
         "ilc_method": "ilc_annotated",  # method that generates the bow
         "mashup_method": "az_annotated",
         # parameter has to match a parameter of a prebuilt bow
         "parameters": [1],
         "ilc_parameters": ["paragraph"],
         "max_year": 2010  # cut-off point for adding files to index
         },
}

doc_methods = {
    # "_full_text": {"type": "standard_multi", "index": "az_ilc_az_annotated_aac_2010_1", "parameters": ["paragraph"],
    #               "runtime_parameters": {"_full_text": 1}}
    "_all_text": {"type": "standard_multi", "index": "az_ilc_az_annotated_aac_2010_1", "parameters": ["paragraph"],
                  "runtime_parameters": {"_all_text": 1}},
    # "mixed": {"type": "standard_multi",
    #           "index": "az_ilc_az_annotated_aac_2010_1",
    #           "parameters": ["paragraph"],
    #           "runtime_parameters": {"_all_text": 1}},
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
    "name": "aac_full_text_kw_selection",
    "description":
        "Full-text indexing of AAC to test kw selection",
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
    "test_files_condition": "metadata.num_in_collection_references:>0 AND metadata.year:>2010",
    # how to sort test files
    "test_files_sort": "metadata.num_in_collection_references:desc",
    # This lets us pick just the first N files from which to generate queries
    "max_test_files": 1000,
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
    "type": "test_kw_selection",
    # If full_corpus, this is the cut-off year for including documents in the general index.
    # In this way we can separate test files and retrieval files.
    "index_max_year": 2010,
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
    "max_queries_generated": 1000,
    # of all precomputed queries, which classes should be processed/evaluated?
    "queries_to_process": ["ALL"],
    # what "zones" to try to train weights for
    "train_weights_for": [],  # ["Bac"], ["Hyp","Mot","Bac","Goa","Obj","Met","Exp","Mod","Obs","Res","Con"]
    # add another doc_method showing the score based on analytical random chance?
    "add_random_control_result": False,
    # "precomputed_queries_filename": "precomputed_queries.json",
    "precomputed_queries_filename": "precomputed_queries_new1k.json",
    # "files_dict_filename": "files_dict.json",
    "files_dict_filename": "files_dict_new1k.json",

    # what to extract as a citation's context
    "context_extraction": "sentence",
    "context_extraction_parameter": CONTEXT_EXTRACTION_C6,

    # exact name of index to use for extracting idf scores etc. for document feature annotation
    # "features_index_name": "idx_full_text_aac_2010_1",
    "features_index_name": "idx_az_ilc_az_annotated_aac_2010_1_paragraph",
    "features_field_name": "_all_text",

    # parameters to keyword selection method
    "keyword_selection_parameters": {
        # "NBestSelector10": {"class": "NBestSelector",
        #                     "parameters": {"N": 10}
        #                     },
        # "NBestSelector20": {"class": "NBestSelector",
        #                     "parameters": {"N": 20}
        #                     },
        # "NBestSelector5": {"class": "NBestSelector",
        #                    "parameters": {"N": 5}
        #                    },

        # "MinimalSetSelector": {"class": "MinimalSetSelector",
        #                        "parameters": {
        #                            "filter_stopwords": False
        #                        }
        #                        },
        # "MultiMaximalSetSelector": {"class": "MultiMaximalSetSelector",
        #                             "parameters": {
        #                                 "filter_stopwords": False
        #                             }
        #                             },
        # "AllSelector": {"class": "AllSelector",
        #                 "parameters": {
        #                     "filter_stopwords": False
        #                 }
        #                 },

        # "KPtester_nokp": {"class": "KPtester",
        #                   "parameters": {"use_kps": False,
        #                                  "use_c3_stopword_list": True,
        #                                  "filter_stopwords": False,
        #                                  }
        #                   },
        #
        # "KPtester_kp_add": {"class": "KPtester",
        #                     "parameters": {"use_kps": True,
        #                                    "kp_method": "add",
        #                                    "use_c3_stopword_list": True,
        #                                    "filter_stopwords": False,
        #                                    }
        #                     },
        #
        # "KPtester_kp_add_avg": {"class": "KPtester",
        #                         "parameters": {"use_kps": True,
        #                                        "kp_method": "add",
        #                                        "use_c3_stopword_list": True,
        #                                        "filter_stopwords": False,
        #                                        "kp_score_compute": "avg"}
        #                         },
        # "KPtester_kp_sub": {"class": "KPtester",
        #                     "parameters": {"use_kps": True,
        #                                    "kp_method": "sub",
        #                                    "use_c3_stopword_list": True,
        #                                    "filter_stopwords": False,
        #                                    }
        #                     },
        # "KPtester_kp_sub_avg": {"class": "KPtester",
        #                         "parameters": {"use_kps": True,
        #                                        "kp_method": "sub",
        #                                        "use_c3_stopword_list": True,
        #                                        "filter_stopwords": False,
        #                                        "kp_score_compute": "avg"}
        #                         },
        #
        "StopwordTester": {"class": "StopwordTester",
                           "parameters": {
                           }
                           },

        # "QueryTester_sw_c6": {"class": "QueryTester",
        #                       "parameters": {
        #                           "filter_stopwords": True,
        #                           "use_weights": True,
        #                       }
        #                       },
        # "QueryTester_sw_c3": {"class": "QueryTester",
        #                       "parameters": {
        #                           "filter_stopwords": False,
        #                           "use_weights": True,
        #                           "use_c3_stopword_list": True
        #                       }
        #                       },
        # "QueryTester_noswfilter": {"class": "QueryTester",
        #                            "parameters": {
        #                                "filter_stopwords": False,
        #                                "use_weights": True,
        #                                "use_all_original_text": True
        #                            }
        #                            },

    },
    # how many folds to use for training/evaluation
    "cross_validation_folds": 4,
    # an upper limit on the number of data points to use
    ##    "max_data_points": 50000,

    "filter_options_resolvable": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # How many authors can the citing and cited paper maximally overlap on?
        # What's the max year for considering a citation? Should match index_max_year above
        # "max_year": 2010,
    },

    "filter_options_ilc": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # How many authors can the citing and cited paper maximally overlap on?
        # What's the max year for considering a citation? Should match index_max_year above
        "max_year": 2010,
    },

    # "expand_match_guids": True,
    # "match_guid_expansion_threshold": 0.2,
    # "match_guid_expansion_max_add": 5,

}

options = {
    "run_prebuild_bows": 0,  # should the whole BOW building process run?
    "overwrite_existing_bows": 0,  # if a BOW exists already, should we overwrite it?
    "build_indexes": 0,  # rebuild indices?

    "generate_queries": 0,  # precompute the queries?
    "overwrite_existing_queries": 0,  # force rebuilding of queries too?

    "force_regenerate_resolvable_citations": 0,  # find again the resolvable citations in a file?

    "clear_existing_prr_results": 1,  # delete previous precomputed results? i.e. start from scratch
    "run_precompute_retrieval": 1,

    # only applies if type == "train_weights" or "extract_kw". This is necessary for run_feature_annotation! And this is because each pipeline may do different annotation
    "run_feature_annotation": 0,
    # annotate scidocs with document-wide features for keyword extraction? By default, False
    "refresh_results_cache": 1,  # should we clean the offline reader cache and redownload it all from elastic?
    "run_experiment": 1,  # must be set to 1 for "run_package_features" to work
    "run_package_features": 1,
    # should we read the cache and repackage all feature information, or use it if it exists already?
    "run_query_start_at": 0,
    "list_missing_files": 0,

    # "max_queries_to_process": 10,
}


def main():
    corpus = ez_connect("AAC", "aws-server")
    exp = Experiment(experiment, options, False)
    exp.run()


if __name__ == '__main__':
    main()
