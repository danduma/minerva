#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      dd
#
# Created:     06/04/2015
# Copyright:   (c) dd 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
# including AZ

import context_extract
from az_cfc_classification import AZ_ZONES_LIST, CORESC_LIST

prebuild_bows={
    "full_text":{"function":"getDocBOWfull", "parameters":[1]},
    "title_abstract":{"function":"getDocBOWTitleAbstract", "parameters":[1]},
    ##"passage":{"function":"getDocBOWpassagesMulti", "parameters":[150,175,200,250,300,350,400,450]},
    "inlink_context":{"function":"generateDocBOWInlinkContext", "parameters":[5, 10, 15, 20, 30, 40, 50] },
    "ilc_AZ":{"function":"generateDocBOW_ILC_Annotated", "parameters":["paragraph","1up_1down","1up","1only"] },
    "az_annotated":{"function":"getDocBOWannotated", "parameters":[1]},
    "section_annotated":{"function":"getDocBOWannotatedSections", "parameters":[1]},
}

# bow_name is just about the name of the file containing the BOWs
prebuild_indexes={
    "full_text":{"type":"standard_multi", "bow_name":"full_text", "parameters":[1]},
    "title_abstract":{"type":"standard_multi", "bow_name":"title_abstract", "parameters":[1]},
##    "passage":{"type":"standard_multi", "bow_name":"passage", "parameters":[150,175,200,250,300,350,400,450]},
    "inlink_context":{"type":"standard_multi", "bow_name":"inlink_context", "parameters":[5, 10, 15, 20, 30, 40, 50]},
##    "inlink_context_year":{"type":"standard_multi", "bow_name":"inlink_context", "parameters":[5, 10, 15, 20, 30, 40, 50], "options":{"max_year":True}},
    "az_annotated":{"type":"standard_multi", "bow_methods":[("az_annotated",[1])], "parameters":[1]},
##    "section_annotated":{"type":"standard_multi", "bow_methods":[("section_annotated",[1])], "parameters":[1]},

##    # this is just ilc but split by AZ
    "ilc_AZ":{"type":"standard_multi", "bow_name":"ilc_AZ", "parameters":["paragraph","1up_1down","1up","1only"]},

    "ilc_full_text":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"full_text", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1]},
##    "ilc_year_full_text":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"full_text", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1], "options":{"max_year":True}},
##    "ilc_section_annotated":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"section_annotated", "ilc_parameters":[10,20,30, 40, 50], "parameters":[1]},
##    "ilc_passage":{"type":"ilc_mashup", "ilc_method":"inlink_context", "mashup_method":"passage","ilc_parameters":[5, 10, 20, 30, 40, 50], "parameters":[250,300,350]},

# this is just normal az_annotated + normal ilc
    "ilc_az_annotated":{"type":"ilc_mashup", "ilc_method":"inlink_context",  "mashup_method":"az_annotated", "ilc_parameters":[5, 10,20,30, 40, 50], "parameters":[1]},
##
####    # this is az-annotated text + az-annotated ilc
    "az_ilc_az_":{"type":"ilc_mashup", "ilc_method":"ilc_AZ", "mashup_method":"az_annotated", "ilc_parameters":["paragraph","1up_1down","1up","1only"], "parameters":[1]},
}

prebuild_general_indexes={
##"full_text":{"type":"standard_multi", "bow_name":"full_text", "parameters":[1]},
}


testing_methods={
    "full_text":{"type":"standard_multi", "index":"full_text", "parameters":[1], "runtime_parameters":["text"]},
    "title_abstract":{"type":"standard_multi", "index":"title_abstract", "parameters":[1], "runtime_parameters":{"text":"1"}},
##    "passage":{"type":"standard_multi", "index":"passage", "parameters":[250,350,400], "runtime_parameters":{"text":"1"}},
##
    "inlink_context":{"type":"standard_multi", "index":"inlink_context",
        "parameters": [10, 20, 30, 50], "runtime_parameters":{"inlink_context":"1"}},

##    "inlink_context_year":{"type":"standard_multi", "index":"inlink_context_year",
##        "parameters": [10, 20, 30], "runtime_parameters":{"inlink_context":"1"}},
##
##    "ilc_passage":{"type":"ilc_mashup",  "index":"ilc_passage", "mashup_method":"passage","ilc_parameters":[10, 20, 30, 40, 50],
##        "parameters":[250,350], "runtime_parameters":{"text":"1","inlink_context":"1"}},

    "az_annotated":{"type":"annotated_boost", "index":"az_annotated", "parameters":[1], "runtime_parameters":
        {"ALL":["AIM","BAS","BKG","CTR","OTH","OWN","TXT"]
        }},

##    "section":{"type":"annotated_boost", "index":"section_annotated", "parameters":[1], "runtime_parameters":
##        {
##        "title_abstract":{"title":"1","abstract":"1"},
##         "full_text":["title","abstract","text"],
##        }},
##
    "ilc":{"type":"ilc_annotated_boost", "index":"ilc_section_annotated", "ilc_parameters":[10, 20, 30, 40, 50], "parameters":[1], "runtime_parameters":
        {
##         "title_abstract":["title","abstract","inlink_context"],
         "full_text":["title", "abstract","text","inlink_context"],
        }},

    # this is normal ilc + az_annotated
##    "ilc_az_annotated":{"type":"ilc_annotated_boost", "index":"ilc_az_annotated", "parameters":[1], "ilc_parameters":[10, 20, 30, 40, 50], "runtime_parameters":
##        {"ALL":["AIM","BAS","BKG","CTR","OTH","OWN","TXT","inlink_context"],
##        }},

    # this is sentence-based ILC, annotated with AZ and CSC
    "ilc_AZ":{"type":"annotated_boost", "index":"ilc_AZ", "parameters":["paragraph","1up_1down","1up","1only"], "runtime_parameters":
        {
        "ALL":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT"],
        "CSC": ["ilc_CSC_"+zone for zone in CORESC_LIST],
        }},

##    "ilc_AZ":{"type":"annotated_boost", "index":"ilc_AZ", "parameters":["paragraph"], "runtime_parameters":
##        {
##        "AZ":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT"],
##        "CSC": ["ilc_CSC_"+zone for zone in CORESC_LIST],
##        }},

    # this is sentence-based AZ and AZ-annotated document contents
    "az_ilc_az":{"type":"ilc_annotated_boost", "index":"az_ilc_az", "parameters":[],
        "ilc_parameters":["1only","1up","1up1down","paragraph"],
        "runtime_parameters":
        {
        "AZ":["ilc_AZ_"+zone for zone in AZ_ZONES_LIST]+AZ_ZONES_LIST,
        "CSC":["ilc_CSC_"+zone for zone in CORESC_LIST]+CORESC_LIST,
####         "OWN":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"0","OWN":"1","TXT":"0","inlink_context":1},
        }},

    }

    # this is the dict of query extraction methods
qmethods={"window":{"parameters":[
##                (3,3),
##                (5,5),
                (10,10),
##                (5,10),
##                (10,5),
                (20,20),
                (20,10),
                (10,20),
                (30,30)
                ],
                "function":"getOutlinkContextWindowAroundCitationMulti",
                "type":"window"},

            "sentence":{"parameters":[
                "1only",
                "paragraph",
                "1up",
                "1up_1down"
                ],
                "function":"getOutlinkContextSentences",
                "type":"sentence"}
                }

import json

def main():
    experiment={
        "name":"w_all_all_methods_stdsim",
        "description":"",
        "prebuild_bows":prebuild_bows,
        "prebuild_indexes":prebuild_indexes,
        "prebuild_general_indexes":prebuild_general_indexes,
        "doc_methods":testing_methods,
        "qmethods":qmethods,
        "test_files":[],
        "test_files_condition":"num_in_collection_references >= 8",
        "use_default_similarity":True,
        "weight_values":[],
        "split_set":None,
        "full_corpus":False,
        "type":"compute_once", #"compute_once","train_weights"
        "numchunks":20,
        "output_filename":"results.csv",
        "pivot_table":"",
        "queries_to_process":["ALL"],
        "queries_classification":"CSC",
        "train_weights_for":CORESC_LIST,
        "precomputed_queries_filename":"precomputed_queries.json",
        "files_dict_filename":"files_dict.json",
    }

    import os
    exp_dir=r"C:\NLP\PhD\bob\experiments"+os.sep
    json.dump(experiment,open(exp_dir+experiment["name"]+".json","w"))

    pass

if __name__ == '__main__':
    main()
