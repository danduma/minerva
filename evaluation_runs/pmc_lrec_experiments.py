# Experiments with the PMC corpus
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

from minerva.az.az_cfc_classification import AZ_ZONES_LIST, CORESC_LIST

import minerva.db.corpora as cp

from minerva.evaluation.experiment import Experiment

# BOW files to prebuild for generating document representation.
prebuild_bows={
##"full_text":{"function":"getDocBOWfull", "parameters":[1]},
##"title_abstract":{"function":"getDocBOWTitleAbstract", "parameters":[1]},
##"passage":{"function":"getDocBOWpassagesMulti", "parameters":[150,175,200,250,300,350,400,450]},
##"inlink_context":{"function":"generateDocBOWInlinkContext", "parameters":[200] },
##"ilc_AZ":{"function":"generateDocBOW_ILC_Annotated", "parameters":["paragraph","1up_1down","1up","1only"] },
"az_annotated":{"function":"getDocBOWannotated", "parameters":[1]},
##"section_annotated":{"function":"getDocBOWannotatedSections", "parameters":[1]},
}

# bow_name is just about the name of the file containing the BOWs
prebuild_indexes={
##    "full_text":{"type":"standard_multi", "bow_name":"full_text", "parameters":[1]},
##    "title_abstract":{"type":"standard_multi", "bow_name":"title_abstract", "parameters":[1]},
##    "passage":{"type":"standard_multi", "bow_name":"passage", "parameters":[150,175,200,250,300,350,400,450]},
##    "inlink_context":{"type":"standard_multi", "bow_name":"inlink_context", "parameters":[5, 10, 15, 20, 30, 40, 50]},
##    "inlink_context_year":{"type":"standard_multi", "bow_name":"inlink_context", "parameters":[5, 10, 15, 20, 30, 40, 50], "options":{"max_year":True}},
    "az_annotated_pmc_2013":{"type":"standard_multi",
                             "bow_name":"az_annotated", # bow to load
                             "parameters":[1], # parameter has to match a parameter of a prebuilt bow
                             },
##    "section_annotated":{"type":"standard_multi", "bow_methods":[("section_annotated",[1])], "parameters":[1]},

##    # this is just ilc but split by AZ
##    "ilc_AZ":{"type":"standard_multi", "bow_name":"ilc_AZ", "parameters":["paragraph","1up_1down","1up","1only"]},

##    "ilc_full_text":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"full_text", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1]},
##    "ilc_year_full_text":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"full_text", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1], "options":{"max_year":True}},
##    "ilc_section_annotated":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"section_annotated", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1]},
##    "ilc_passage":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"passage","ilc_parameters":[5, 10, 20, 30, 40, 50], "parameters":[250,300,350]},

# this is just normal az_annotated + normal ilc
####    "ilc_az_annotated":{"type":"ilc_mashup", "ilc_method":"inlink_context",  "mashup_method":"az_annotated", "ilc_parameters":[5, 10,20,30, 40, 50], "parameters":[1]},
##
####    # this is az-annotated text + az-annotated ilc
####    "az_ilc_az_":{"type":"ilc_mashup", "ilc_method":"ilc_AZ", "mashup_method":"az_annotated", "ilc_parameters":["paragraph","1up_1down","1up","1only"], "parameters":[1]},
}

prebuild_general_indexes={
##    "full_text":{"type":"standard_multi", "bow_name":"full_text", "parameters":[1]},
##    "ilc_full_text":{"type":"standard_multi", "bow_name":"full_text", "parameters":[1]},
    "az_annotated_pmc_2013":{"type":"standard_multi",
                             "bow_name":"az_annotated", # bow to load
                             "parameters":[1], # parameter has to match a parameter of a prebuilt bow
                             "max_year":2013 # cut-off point for adding files to index
                             },
}


doc_methods={
##    "full_text":{"type":"standard_multi", "index":"full_text", "parameters":[1], "runtime_parameters":["text"]},
##    "title_abstract":{"type":"standard_multi", "index":"title_abstract", "parameters":[1], "runtime_parameters":{"text":"1"}},
##    "passage":{"type":"standard_multi", "index":"passage", "parameters":[250,350,400], "runtime_parameters":{"text":"1"}},
##
##    "inlink_context":{"type":"standard_multi", "index":"inlink_context",
##        "parameters": [10, 20, 30], "runtime_parameters":{"inlink_context":"1"}},
##
##    "inlink_context_year":{"type":"standard_multi", "index":"inlink_context_year",
##        "parameters": [10, 20, 30], "runtime_parameters":{"inlink_context":"1"}},
##
##    "ilc_passage":{"type":"ilc_mashup",  "index":"ilc_passage", "mashup_method":"passage","ilc_parameters":[10, 20, 30, 40, 50],
##        "parameters":[250,350], "runtime_parameters":{"text":"1","inlink_context":"1"}},

    "az_annotated":{"type":"annotated_boost", "index":"az_annotated_pmc_2013", "parameters":[1], "runtime_parameters":{
##        "AZ_ALL":AZ_ZONES_LIST,
        "CSC_ALL":CORESC_LIST,
        }},

##    "section":{"type":"annotated_boost", "index":"section_annotated", "parameters":[1], "runtime_parameters":
##        {
##        "title_abstract":{"title":"1","abstract":"1"},
##         "full_text":["title","abstract","text"],
##        }},
##
##    "ilc":{"type":"ilc_annotated_boost", "index":"ilc_section_annotated", "ilc_parameters":[10, 20, 30, 40, 50], "parameters":[1], "runtime_parameters":
##        {
##         "title_abstract":["title","abstract","inlink_context"],
##         "full_text":["title", "abstract","text","inlink_context"],
##        }},

    # this is normal ilc + az_annotated
##    "ilc_az_annotated":{"type":"ilc_annotated_boost", "index":"ilc_az_annotated", "parameters":[1], "ilc_parameters":[10, 20, 30, 40, 50], "runtime_parameters":
##        {"ALL":["AIM","BAS","BKG","CTR","OTH","OWN","TXT","inlink_context"],
##        }},

    # this is sentence-based ILC, annotated with AZ and CSC
##    "ilc_AZ":{"type":"annotated_boost", "index":"ilc_AZ", "parameters":["paragraph","1up_1down","1up","1only"], "runtime_parameters":
##        {
##        "ALL":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT"]
##        }},

##    "ilc_AZ":{"type":"annotated_boost", "index":"ilc_AZ", "parameters":["paragraph"], "runtime_parameters":
##        {
##        "AZ":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT"],
##        "CSC": ["ilc_CSC_"+zone for zone in CORESC_LIST],
##        }},

    # this is sentence-based AZ and AZ-annotated document contents
##    "az_ilc_az":{"type":"ilc_annotated_boost", "index":"az_ilc_az", "parameters":[],
##        "ilc_parameters":["1only","1up","1up1down","paragraph"],
##        "runtime_parameters":
##        {
##        "ALL":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT","ilc_AZ_AIM"],
######         "OTH":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"1","OWN":"0","TXT":"0","inlink_context":1},
######         "OWN":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"0","OWN":"1","TXT":"0","inlink_context":1},
##        }},
    }

    # this is the dict of query extraction methods
qmethods={

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

            "sentence":{"parameters":[
##                "1only",
##                "paragraph",
##                "1up",
##                "0up_1down",
##                "1up_1down",
                "2up_2down"
                ],
                "method":"Sentences",
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

experiment={
    "name":"pmc_lrec_experiments",
    "description":
        """Re-run the original LREC experiments, but this time with the whole Sapienta-annotated PMC""",
    # dict of bag-of-word document representations to prebuild
    "prebuild_bows":prebuild_bows,
    # dict of per-file indexes to prebuild
    "prebuild_indexes":prebuild_indexes,
    # dict of general indexes to prebuild
    "prebuild_general_indexes":prebuild_general_indexes,
    # dictionary of document representation methods to test
    "doc_methods":doc_methods,
    # dictionary of query generation methods to test
    "qmethods":qmethods,
    # list of files in the test set
    "test_files":[],
    # SQL condition to automatically generate the list above
    "test_files_condition":"metadata.num_in_collection_references:>0 AND metadata.year:>2013",
    # This lets us pick just the first N files
    "max_test_files":100,
    # Use Lucene DefaultSimilarity? As opposed to FieldAgnosticSimilarity
    "use_default_similarity":True,
    # Annotate sentences with AZ/CoreSC/etc?
    "rhetorical_annotations":[],
    # Run annotators? If False, it is assumed the sentences are already annotated
    "run_rhetorical_annotators":False,
    # Separate queries by AZ/CSC, etc?
    "use_rhetorical_annotation":True,
    "weight_values":[],
    # ?
    "split_set":None,
    # use full-collection retrival? If False, it runs "citation resolution"
    "full_corpus":True,
    # "compute_once","train_weights"
    "type":"train_weights",
    # If full_corpus, this is the cut-off year for including documents in the general index.
    # In this way we can separate test files and retrieval files.
    "index_max_year": 2013,
    # how many chunks to split each file for statistics on where the citation occurs
    "numchunks":10,
    # name of CSV file to save results in
    "output_filename":"results.csv",
    "pivot_table":"",
    "max_results_recall":200,
    # should queries be classified based on some rhetorical class of the sentence: "az", "csc_type"
    "queries_classification":"csc_type",
    # of all precomputed queries, which classes should be processed/evaluated?
    "queries_to_process":["ALL"],
    # what "zones" to try to train weights for
    "train_weights_for":CORESC_LIST,
    # add another doc_method showing the score based on analytical random chance?
    "add_random_control_result": False,
    "precomputed_queries_filename":"precomputed_queries.json",
    "files_dict_filename":"files_dict.json",
}


options={
    "run_prebuild_bows":0, # should the whole BOW building process run?
    "force_prebuild":0,   # if a BOW exists already, should we overwrite it?
    "rebuild_indexes":1,   # rebuild indices?
    "recompute_queries":0, # force rebuilding of queries too?
    "run_precompute_retrieval":1, # only applies if type == "train_weights"
    "override_folds":4,
    "override_metric":"avg_mrr",
}

def main():
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\pmc_coresc")

##    experiment["test_files"]=["456f8c80-9807-46a9-8455-cd4a7e346f9d"]

    experiment["test_files"]=[
                            "bdc9a118-cb76-4d26-9c4d-e886794428f5",
                            "65a4319e-4324-4529-96fa-66c52e392da0",
                            "0cc28fb0-b116-4990-b816-3dc066273c34",
                            "ef3e4284-c527-4e83-9b59-f8996b09df76",
                            "b3129460-d284-4f69-83a8-f87f588e7800",
                            "d8548dab-ff28-4f93-b2ae-16887e59e8ad",
                            "42efd8ec-4c06-4754-a527-3045eed87766",
                            "f4374057-7ab2-4567-b73b-aa5b72328d3e",
                            "cbf989c5-79f5-4317-8515-2192e2a3fe2a",
                            "37d1cc24-68a5-4a36-b55d-94acdfad08c1",]

    exp=Experiment(experiment, options, True)
    exp.run()

if __name__ == '__main__':
    main()
##    from minerva.proc.doc_representation import getDictOfLuceneIndeces
##    from minerva.evaluation.base_pipeline import getDictOfTestingMethods
##    print(getDictOfLuceneIndeces(prebuild_general_indexes))
##    print(getDictOfTestingMethods(doc_methods))
