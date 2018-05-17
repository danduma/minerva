# Experiments with the PMC corpus
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

from __future__ import absolute_import
from proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST

import db.corpora as cp

from evaluation.experiment import Experiment

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
prebuild_indeces={
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
                "1up_1down",
##                "2up_2down"
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
    "name":"pmc_lrec_experiments2",
    "description":
        """Re-run the original LREC experiments, but this time with the whole Sapienta-annotated PMC""",
    # dict of bag-of-word document representations to prebuild
    "prebuild_bows":prebuild_bows,
    # dict of per-file indexes to prebuild
    "prebuild_indeces":prebuild_indeces,
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
##    "split_set":None,
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
    "train_weights_for": ["Mod"], #CORESC_LIST, #["Bac"], ["Hyp","Mot","Bac","Goa","Obj","Met","Exp","Mod","Obs","Res","Con"]
    # add another doc_method showing the score based on analytical random chance?
    "add_random_control_result": False,
    "precomputed_queries_filename":"precomputed_queries.json",
    "files_dict_filename":"files_dict.json",
}


options={
    "run_prebuild_bows":0, # should the whole BOW building process run?
    "overwrite_existing_bows":0,   # if a BOW exists already, should we overwrite it?
    "rebuild_indexes":0,   # rebuild indices?
    "recompute_queries":0,  # force rebuilding of queries too?
    "run_precompute_retrieval":0,  # only applies if type == "train_weights"
    "clear_existing_prr_results":False, # delete previous precomputed results? i.e. start from scratch
    "override_folds":4,
    "override_metric":"avg_ndcg",
}

def main():
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\pmc_coresc", endpoint={"host":"129.215.90.202", "port":9200})

##    experiment["test_files"]=["456f8c80-9807-46a9-8455-cd4a7e346f9d"]

    experiment["test_files"]=[
##                                "bdc9a118-cb76-4d26-9c4d-e886794428f5",
##                                "65a4319e-4324-4529-96fa-66c52e392da0",
##                                "0cc28fb0-b116-4990-b816-3dc066273c34",
##                                "ef3e4284-c527-4e83-9b59-f8996b09df76",
##                                "b3129460-d284-4f69-83a8-f87f588e7800",
##                                "d8548dab-ff28-4f93-b2ae-16887e59e8ad",
##                                "42efd8ec-4c06-4754-a527-3045eed87766",
##                                "f4374057-7ab2-4567-b73b-aa5b72328d3e",
##                                "cbf989c5-79f5-4317-8515-2192e2a3fe2a",
##                                "37d1cc24-68a5-4a36-b55d-94acdfad08c1",
##                                "2b5202ec-e71b-4d1a-8ef4-439c4f505342",
##                                "11ba9f31-13f8-4a40-8bfc-6c9c7725e7ba",
##                                "e047f55f-ff56-44a6-a07c-794887330752",
##                                "d39d353a-f9ca-4ce3-ab42-e9a16f5bd372",
##                                "a407716f-4516-4cba-9c52-d4e3b09bcda6",
##                                "680724b2-50e7-4809-a86f-e63326059f7e",
##                                "1ce857ab-7692-4a95-9ba0-f517179a940e",
##                                "e12b2e84-a91d-4170-88a6-6ba983ceab1b",
##                                "5a6c0a35-dbe0-486a-8edf-3c3d3638f06e",
##                                "c40d5876-208c-4eb4-b239-652ed14f8560",
##                                "9a764770-fd73-474e-8f38-cf0128371e2c",
##                                "54432fc8-c1c4-42f9-95b0-c5fad39f8317",
##                                "a7dab0f1-5891-4d83-92c2-d25069c49d27",
##                                "283ed90d-3ff9-4161-8c4d-4e55a555973e",
##                                "6478c6ca-e16c-473f-9f4c-060143b3cc8f",
##                                "666f2c58-3180-465b-877c-28d14cbcdf98",
##                                "f5dedb99-f2a1-4ae9-b4a0-3c23e33cbfc9",
##                                "e5ed924b-8b78-4c76-bb6c-54d9790c8a15",
##                                "b8ace4e7-8523-471f-847b-b45aee8ccfc1",
##                                "ff30447d-828e-4699-bbf7-ce586aae9764",
##                                "aec8d55c-43e0-42cb-b832-77f888c2325a",
##                                "067862a3-d8fd-4252-b831-f6f120af82a1",
##                                "64956609-5a4d-4e05-bad1-0445c3d1834d",
##                                "cd1cd1ec-ecc9-4e70-96b3-7f1447ec0df3",
##                                "d61b922b-622b-440c-b040-3db563fd6f0e",
##                                "51d71d97-5abb-4a4d-ba77-7d18a11343f0",
##                                "b4c2215a-0a38-4e44-a5ab-f0d0114d89fc",
##                                "d3265a02-86ba-47e9-879c-a15043ca5808",
##                                "3e53830f-33a5-4192-9159-bcd01a3e66d3",
##                                "be50acb5-e165-4afb-b259-eeb9f28d0f2e",
##                                "fb8e6675-46d9-41c8-8ba0-598842a63fe8",
##                                "34043f1e-3424-4c4a-b782-9489fc274db5",
##                                "e07f7715-d400-4958-a0ac-2e6dab3b1843",
##                                "44f52aea-cae3-4e1c-85ce-da0038cbcea1",
##                                "b99f5b2d-6edd-4787-a50d-fef7d030ff05",
##                                "55ef06b7-ffc6-4e43-9362-daf4b9f6735f",
##                                "89c63f73-988a-4ece-99fd-e6c91fc9f6fd",
##                                "83293e90-8f3e-45db-8dae-49a179568d3e",
##                                "7fbcd237-d40d-4d44-9c9f-7e2f462e547e",
##                                "5303a3a3-c1cd-458c-9bef-56df3080169d",
##
##                                "a3af153f-cf9d-40ea-b64d-c6e52e0a187b",
##                                "38a67959-7ad9-426d-8357-51ab376b7a4b",
##                                "5f71b848-2f22-49bf-b32a-c1cc441a6dbe",
##                                "e875b2dc-3757-4728-8e8b-47c6a1d8241c",
##                                "a1bd31a8-66bc-4be9-aa85-e2e821aa18f5",
##                                "ef030f01-cdcb-4aaf-8ec1-c1a8778095da",
##                                "5b4f7822-1127-4d45-84ca-6755e1debaab",
##                                "a11c9c92-e294-4dfc-8e73-f432ad460776",
##                                "6c647939-ef22-4d8b-b887-121272168829",
##                                "d4c97daa-790c-40fe-a17e-fffa8e7fbd36",
##                                "65ee7543-549d-4821-b51d-8fc27dbe85cb",
##                                "4fe8ae7f-47bf-41fd-95ef-14ee7831f37e",
##                                "45d1bde0-2bd5-413b-89e3-9151d5a73ffb",
##                                "7bedaa57-30ff-4569-8456-59236171a80f",
##                                "67a054e0-744e-477b-80a4-06a268064bc7",
##                                "51a51cbb-952b-450b-970e-f6a23ecf9ce6",
##                                "b14209b3-d868-41cd-b1d0-f1a1489220f3",
##                                "53230a94-3baf-4825-a039-e8125890e737",
##                                "bb576feb-658e-45ef-810a-617b586159e5",
##                                "7d5ad1b5-2f3d-4728-b583-ec1ebbc3dac6",
##                                "f27cf9db-0d2d-490b-9917-c076a5ebca2c",
##                                "3b6679e5-deae-43ee-a98b-cac7029e92f4",
##                                "0d44cbba-1989-4654-b250-1b41285359ea",
##                                "b0cceb78-5f66-4084-accb-171040521cda",
##                                "18bf7b21-2456-49da-882a-06032ec46bec",
##                                "588b99bd-c358-440b-b30c-e1f3dc10b96b",
##                                "c4c1d5c0-7f40-465a-bbdb-351b4c9948a8",
##                                "0efcb373-ecd3-4e10-9f2a-1bbd3a6cbf58",
##                                "9bb0db11-2821-4d55-8f34-9bfd5d58f444",
##                                "49ff3f83-b4d7-4979-800f-785460c95552",
##                                "58e0a5d1-6343-4e2e-b544-6f690bff023e",
##                                "5a84843d-d7b0-43b0-846c-d30d3196ee8a",
##                                "6f244f35-8f61-4eb9-9de0-dfbbca63532b",
##                                "9d3cf2ea-162b-4e78-a311-b7333ad65c3a",
##                                "75e80547-50f5-4a12-8db6-e799a2e5029b",
##                                "f6c70cbd-e6c3-4ea5-b99f-ac0c455d832a",
##                                "f72d2af0-1e8a-40f5-9acf-ddbe9ddc4a7b",
##                                "9a81337e-1280-4b11-9b00-7e516d298ea1",
##                                "c7a83006-6ed3-46ec-b476-c49770dc4979",
##                                "aa45f968-61a4-421d-a532-a036fb8336ef",
##                                "f38092bc-b1e2-4ba8-ad60-b964825e52ac",
##                                "a755f020-c04d-4640-8b43-fb63b560bd6e",
##                                "1f06edd3-09d6-4033-b65e-a96d6a78f748",
##                                "33273c27-bcb7-4a4d-b339-c8af16c97b91",
##                                "802d2b57-0425-410e-82b6-f0024bc6f0dd",
##                                "699e837a-a662-49a0-b4c5-b3ef113eff34",
##                                "5b019a09-e21f-4109-a757-2c8396c8f169",
##                                "d4a70f39-7c5c-4566-8b5c-72208f3929ea",
##                                "6cf4f22d-c77b-4f0a-9e3c-378d7803f62b",

##                                "5cc55656-d309-4906-9cbf-7e34e734c352",
##                                "4222902b-e5fc-4eef-a7a7-79ec85d8e7c0",
##                                "4e408d49-6e51-441a-9c1c-8720d0d7032a",
##                                "aba078e1-c385-45b2-9adf-1ab7901b373b",
##                                "772eaabf-8996-486f-9bba-355cbf0c15e1",
##                                "38048193-6565-45ea-9950-64c7e4c266a3",
##                                "4d36eeb9-9121-4510-847e-99b80c77473e",
##                                "7b9e39c9-18a3-4112-ba9f-36b70d60f60f",
##                                "e31fd474-e2a0-4b3c-9d36-b31003b3bbc6",
##                                "69b92870-e050-4277-bd00-08f79aa6d9e6",
##                                "a67be750-73dc-427b-ac6a-e46adcaf7430",
##                                "1f8a95c4-856a-4f39-9c45-52d309d8c075",
##                                "b3606482-e22f-4809-948a-385a4f1e47cb",
##                                "7a4a67fb-3f4c-4c26-a060-775e8a4b7480",
##                                "bc39625f-1bc1-49ce-8ea3-4f8debe90b01",
##                                "202ef49f-c3f9-4d3a-b971-2d8094c06242",
##                                "28967c8d-2584-4898-9b62-0bfc669e2490",
##                                "355d6857-06ee-4430-a511-aaf0e8eaf23d",
##                                "0ce7eebd-0815-4f5d-b1c9-fafb65584994",
##                                "870c4608-525d-44d1-960f-4eb73589618c",
##                                "2420a665-d848-459f-9f51-456275d42e8b",
##                                "deb5362b-af92-4973-970f-ebe3fec12ee9",
##                                "d244e4d9-808c-4abd-a627-02716e9609c9",
##                                "de3f08f8-12fa-41d1-82cc-30c2c43cf52e",
##                                "575e9d63-94d7-483d-a980-8c974afc0ad9",
##                                "31410622-7133-4472-8225-8cf6b1eb1683",
##                                "80f3bd59-c5d6-43a0-97e3-155c1af50275",
##                                "cf2e8b40-5fab-4b17-acb1-6b676d909aa6",
##                                "51375367-8a16-4070-bab7-5ebcca3427c4",
##                                "dc293831-b099-45e2-a9ef-87ec8ccb8722",
##                                "aad3943f-37f9-4774-aa77-f312650b699e",
##                                "cdf828fc-2fb5-4c6f-8b42-8ff7c8ff0ff0",
##                                "6569f681-77e3-4ceb-a956-04e7a751f2b3",
##                                "2edffd94-b1db-46b7-bc4b-e4da9dcf4f51",
##                                "45a922bc-814a-40bd-a76a-fcbeca77bc81",
##                                "be4f19f7-de07-4674-8f03-1fbec9c7dd04",
##                                "48d9f4cf-c081-4520-b350-6ca3142987a7",
##                                "f35243d1-a3e3-4402-99b3-e576a27cde0d",
##                                "e8f567f8-3179-4214-bcbc-79332c1cfd1d",
##                                "209e32f7-a3cd-4e86-afee-2935a1f25514",
##                                "1cd47a2c-58c1-4c89-a689-cbdc0dd1f6b7",
##                                "2c64d4d5-3883-4fee-8c2c-1c0afb3835cf",
##                                "1323e0b5-c986-4ca6-855a-0b147d938e50",
##                                "d293f62a-983f-4ddc-a227-84d82bb36af1",
##                                "5b6439c9-466d-4bc0-aff2-e85de8eb9337",
##                                "da2b0b43-26b1-458b-b57a-83279ceb314e",
##                                "c21e2afc-0f92-490d-aa1b-7f826a83221d",
##                                "c5e67372-cf98-45db-bb7a-e3f4e7662774",
##                                "c7f91884-cfc8-406b-919e-658008c21279",
##                                "753b9d9a-ce8d-4fba-ac74-106526416738",
##                                "799680bf-5150-4fb2-b9b6-91fd0edc2593",
                            ]

    exp=Experiment(experiment, options, True)
    exp.run()

if __name__ == '__main__':
    main()
##    from proc.doc_representation import getDictOfLuceneIndeces
##    from evaluation.base_pipeline import getDictOfTestingMethods
##    print(getDictOfLuceneIndeces(prebuild_general_indexes))
##    print(getDictOfTestingMethods(doc_methods))
