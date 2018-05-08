# Reimplementation of the equivalent ACL14 experiments
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

from __future__ import absolute_import
import os, json

from az.az_cfc_classification import AZ_ZONES_LIST, CORESC_LIST

import db.corpora as cp

from evaluation.experiment import Experiment
from evaluation.query_generation import QueryGenerator
from db.ez_connect import ez_connect

# BOW files to prebuild for generating document representation.
prebuild_bows = {
    # "full_text": {"function": "getDocBOWfull", "parameters": [1]},
    ##"title_abstract":{"function":"getDocBOWTitleAbstract", "parameters":[1]},
    ##"passage":{"function":"getDocBOWpassagesMulti", "parameters":[150,175,200,250,300,350,400,450]},
    "inlink_context": {"function": "getDocBOWInlinkContextCache",
                       "parameters": [30, 40, 50, 100, "1only",
                                      "paragraph",
                                      "1up",
                                      "0up_1down",
                                      "1up_1down",
                                      "2up_2down"]},
    # "inlink_context": {"function": "getDocBOWInlinkContextCache", "parameters": [100]},
    ##"ilc_AZ":{"function":"generateDocBOW_ILC_Annotated", "parameters":["paragraph","1up_1down","1up","1only"] },
    ##"az_annotated":{"function":"getDocBOWannotated", "parameters":[1]},
    ##"section_annotated":{"function":"getDocBOWannotatedSections", "parameters":[1]},
}

# bow_name is just about the name of the file containing the BOWs
prebuild_indeces = {
    # "inlink_context": {"type": "standard_multi", "bow_name": "inlink_context", "parameters": [30, 40, 50, 100]},
}

prebuild_general_indexes = {
    # "full_text_aac_2010": {"type": "standard_multi", "bow_name": "full_text", "parameters": [1]},
    "inlink_context_aac_2010": {"type": "standard_multi",
                                "bow_name": "inlink_context",
                                "parameters": [30, 40, 50, 100, "1only",
                                               "paragraph",
                                               "1up",
                                               "0up_1down",
                                               "1up_1down",
                                               "2up_2down"]},
    "ilc_full_text_aac_2010": {"type": "ilc_mashup",
                               "ilc_method": "inlink_context",
                               "mashup_method": "full_text",
                               "ilc_parameters": [30, 40, 50, 100, "1only",
                                                  "paragraph",
                                                  "1up",
                                                  "0up_1down",
                                                  "1up_1down",
                                                  "2up_2down"],
                               "parameters": [1]},
}

doc_methods = {
    # "full_text": {"type": "standard_multi", "index": "full_text_aac_2010", "parameters": [1],
    #               "runtime_parameters": ["text"]},
    ##    "title_abstract":{"type":"standard_multi", "index":"title_abstract_aac_2010", "parameters":[1], "runtime_parameters":{"text":"1"}},
    ##    "passage":{"type":"standard_multi", "index":"passage_aac_2010", "parameters":[250,350,400], "runtime_parameters":{"text":"1"}},
    ##
    # "inlink_context": {"type": "standard_multi",
    #                    "index": "inlink_context_aac_2010",
    #                    "parameters": [30, 40, 50, 100],
    #                    "runtime_parameters": {"text": "1"}},

    "ilc_full_text": {"type": "ilc_mashup",
                      "index": "ilc_full_text_aac_2010",
                      "ilc_parameters": [30, 40, 50, 100, "1only",
                                         "paragraph",
                                         "1up",
                                         "0up_1down",
                                         "1up_1down",
                                         "2up_2down"],
                      "parameters": ["1"],
                      "runtime_parameters": {"text": "1"}},

    ##    "ilc_passage":{"type":"ilc_mashup",  "index":"ilc_passage", "mashup_method":"passage","ilc_parameters":[10, 20, 30, 40, 50],
    ##        "parameters":[250,350], "runtime_parameters":{"text":"1","inlink_context":"1"}},

    # "ilc_annotated": {"function": "generateDocBOW_ILC_Annotated",
    #                   "parameters":
    #                       ["2up_2down",
    #                        "1up_1down",
    #                        "paragraph"]},

}

# this is the dict of query extraction methods
qmethods = {
    "window": {"parameters": [
        ##                (3,3),
        ##                (5,5),
        ##                (10,10),
        ##                (5,10),
        ##                (10,5),
        ##                (20,20),
        (20, 10),
        (10, 20),
        (30, 30),
        (50, 50),
        (100, 100),
        ##                (500,500),
    ],
        "method": "Window",
    },

    "sentence": {"parameters": [
        "1only",
        "paragraph",
        "1up",
        "0up_1down",
        "1up_1down",
        "2up_2down"
    ],
        "method": "Sentences",
    },

}

experiment = {
    "name": "thesis_chapter3_experiment_aac",
    "description":
        """Reimplement all of the ACL14 experiments with the full AAC collection""",
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
    "type": "compute_once",
    # If full_corpus, this is the cut-off year for including documents in the general index.
    # In this way we can separate test files and retrieval files.
    "index_max_year": 2010,
    "numchunks": 10,
    # name of CSV file to save results in
    "output_filename": "results.csv",
    "pivot_table": "",
    "max_results_recall": 200,
    # should queries be classified based on some rhetorical class of the sentence: "az", "csc"
    "queries_classification": None,
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

    "filter_options_resolvable": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_self_citation": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # None == no limit
        # What's the max year for considering a citation? Should match index_max_year above
        "max_year": 2010,
    },

    "filter_options_ilc": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_self_citation": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # How many authors can the citing and cited paper maximally overlap on?
        # What's the max year for considering a citation? Should match index_max_year above
        "max_year": 2010,
    }

}

options = {
    "run_prebuild_bows": 0,  # should the whole BOW building process run?
    "overwrite_existing_bows": 0,  # if a BOW exists already, should we overwrite it?
    "rebuild_indexes": 1,  # rebuild indices?
    "compute_queries": 1,  # precompute the queries?
    "force_regenerate_resolvable_citations": 1,  # find again the resolvable citations in a file?
    "overwrite_existing_queries": 1,  # force rebuilding of queries too?

    "clear_existing_prr_results": 1,  # delete previous precomputed results? i.e. start from scratch
    "run_precompute_retrieval": 0,
    # only applies if type == "train_weights" or "extract_kw". This is necessary for run_feature_annotation! And this is because each pipeline may do different annotation
    "run_feature_annotation": 0,
    # annotate scidocs with document-wide features for keyword extraction? By default, False
    "refresh_results_cache": 1,  # should we clean the offline reader cache and redownload it all from elastic?
    "run_experiment": 1,  # must be set to 1 for "run_package_features" to work
    "run_package_features": 0,
    # should we read the cache and repackage all feature information, or use it if it exists already?
    "start_at": 0,
    "list_missing_files": 1,

    # "max_files_to_process": 10,  # FOR QUICK TESTING
}


def main():
    corpus = ez_connect("AAC")
    # experiment["test_files"] = cp.Corpus.listPapers("year:>=2011")

    exp = Experiment(experiment, options)
    exp.run()
    pass


if __name__ == '__main__':
    main()
