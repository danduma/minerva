# Experiments with the ACL corpus like the ones for LREC'16
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function
from __future__ import absolute_import
# from proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST

import db.corpora as cp

from evaluation.experiment import Experiment
from proc.general_utils import getRootDir

# BOW files to prebuild for generating document representation.
prebuild_bows = {
    "full_text": {"function": "getDocBOWfull", "parameters": [1]},

}

# bow_name is just about the name of the file containing the BOWs
prebuild_indeces = {
}

# the name of the index is important, should be unique
prebuild_general_indexes = {
    "full_text_aac_2010": {"type": "standard_multi", "bow_name": "full_text", "parameters": [1]},
}

doc_methods = {
    "full_text": {"type": "standard_multi", "index": "full_text_aac_2010", "parameters": [1],
                  "runtime_parameters": {"text": 1}},
}

# this is the dict of query extraction methods
qmethods = {

    ##        "window":{"parameters":[
    ##                (3,3),
    ##                (5,5),
    ##                (10,10),
    ##                (5,10),
    ##                (10,5),
    ##                (20,20),
    ##                (20,10),
    ##                (10,20),
    ##                (30,30),
    ##                (50,50),
    ##                (100,100),
    ##                (500,500),
    ##                ],
    ##                "method":"Window",
    ##                },

    "sentence": {"parameters": [
        ##                "1only",
        ##                "paragraph",
        ##                "1up",
        ##                "0up_1down",
        ##                "1up_1down",
        ##                "2up_2down",
        "2up_2down"
    ],
        "method": "Sentences",
    },

    ##            "annotated_sentence":{"parameters":[
    ##                "pno",
    ##                "po",
    ##                "no",
    ##                "n",
    ##                "p",
    ##                ],
    ##                "method":"SelectedSentences",
    ##                },

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
    # of all precomputed queries, which classes should be processed/evaluated?
    "queries_to_process": ["ALL"],
    # what "zones" to try to train weights for
    "train_weights_for": [],  # ["Bac"], ["Hyp","Mot","Bac","Goa","Obj","Met","Exp","Mod","Obs","Res","Con"]
    # add another doc_method showing the score based on analytical random chance?
    "add_random_control_result": False,
    "precomputed_queries_filename": "precomputed_queries.json",
    "files_dict_filename": "files_dict.json",

    # what to extract as a citation's context
    "context_extraction": "sentence",
    "context_extraction_parameter": "2up_2down",

    # exact name of index to use for extracting idf scores etc. for document feature annotation
    "features_index_name": "idx_full_text_aac_2010_1",

    # parameters to keyword selection method
    "keyword_selection_parameters": {"NBestSelector10": {"class": "NBestSelector",
                                                         "parameters": {"N": 10}
                                                         },
                                     "NBestSelector20": {"class": "NBestSelector",
                                                         "parameters": {"N": 20}
                                                         },
                                     "NBestSelector5": {"class": "NBestSelector",
                                                         "parameters": {"N": 5}
                                                         },
                                     "MinimalSetSelector": {"class": "MinimalSetSelector",
                                                            "parameters": {}
                                                            },
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
        "max_year": 2010,
    },

    "filter_options_ilc": {
        # Should resolvable citations exclude those that have the same first author as the test document?
        "exclude_same_first_author": True,
        # How many authors can the citing and cited paper maximally overlap on?
        "max_overlapping_authors": None,  # How many authors can the citing and cited paper maximally overlap on?
        # What's the max year for considering a citation? Should match index_max_year above
        "max_year": 2010,
    }
}

options = {
    "run_prebuild_bows": 0,  # should the whole BOW building process run?
    "overwrite_existing_bows": 0,  # if a BOW exists already, should we overwrite it?
    "rebuild_indexes": 0,  # rebuild indices?
    "generate_queries": 1,  # precompute the queries?
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
    "start_at": 0,
    "list_missing_files": 1,
}


def main():
    from multi.config import MINERVA_ELASTICSEARCH_ENDPOINT
    cp.useElasticCorpus()
    root_dir = getRootDir("aac")

    cp.Corpus.connectCorpus(root_dir, endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
    cp.Corpus.setCorpusFilter("AAC")
    ##    experiment["test_files"]=["456f8c80-9807-46a9-8455-cd4a7e346f9d"]

    exp = Experiment(experiment, options, False)
    exp.run()
    # from ml.keyword_support import saveOfflineKWSelectionTraceToCSV
    # from evaluation.kw_selection_pipeline import WRITER_NAME
    # import os
    # saveOfflineKWSelectionTraceToCSV(WRITER_NAME, os.path.join(exp.exp["exp_dir"], "cache"), exp.exp["exp_dir"])

if __name__ == '__main__':
    main()
