# functions for testing purposes
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import glob, math, os, re, sys, gc, random, json
from copy import deepcopy
from collections import defaultdict, namedtuple, OrderedDict

import pandas as pd
import lucene
from sklearn import cross_validation

import minerva.db.corpora as cp
from minerva.proc.results_logging import ResultsLogger
from minerva.proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11
from minerva.proc.doc_representation import findCitationInFullText
from testing_pipelines import getDictOfTestingMethods
import minerva.proc.doc_representation as doc_representation
from minerva.proc.general_utils import getSafeFilename, exists, ensureDirExists
from minerva.evaluation.results_analysis import drawWeights, drawScoreProgression

from minerva.evaluation.lucene_retrieval import LuceneRetrieval,LuceneRetrievalBoost,storedFormula,precomputedExplainRetrieval

GLOBAL_FILE_COUNTER=0


class QueryGenerator(object):
    """
        Loops over the testing files, generating queries for each citation context.

        The actual query generation happens in the QueryExtractor classes
    """

    def __init__(self):
        """
        """

    def saveAllQueries(self, files_dict):
        """
            Dumps all precomputed queries to disk.
        """
        precomputed_queries=self.precomputed_queries
        json.dump(precomputed_queries,open(os.path.join(self.exp["exp_dir"],"precomputed_queries.json"),"w"))
        queries_by_az={zone:[] for zone in AZ_ZONES_LIST}
        queries_by_cfc=defaultdict(lambda:[])
        queries_by_csc=defaultdict(lambda:[])

        if self.exp.get("random_zoning",False):
            queries_by_rz7=defaultdict(lambda:[])
            queries_by_rz11=defaultdict(lambda:[])
        if self.exp.get("use_rhetorical_annotation", False):
            for precomputed_query in precomputed_queries:
                queries_by_az[precomputed_query["az"]].append(precomputed_query)
                queries_by_cfc[precomputed_query["cfc"]].append(precomputed_query)
                if "csc_type" in precomputed_query:
                    queries_by_csc[precomputed_query["csc_type"]].append(precomputed_query)

                if self.exp.get("random_zoning",False):
                    queries_by_rz7[random.choice(RANDOM_ZONES_7)].append(precomputed_query)
                    queries_by_rz11[random.choice(RANDOM_ZONES_11)].append(precomputed_query)

        json.dump(files_dict,open(self.exp["exp_dir"]+"files_dict.json","w"))
        if self.exp.get("use_rhetorical_annotation", False):
            json.dump(queries_by_az,open(self.exp["exp_dir"]+"queries_by_AZ.json","w"))
            json.dump(queries_by_cfc,open(self.exp["exp_dir"]+"queries_by_CFC.json","w"))
            json.dump(queries_by_csc,open(self.exp["exp_dir"]+"queries_by_CSC.json","w"))

        if self.exp.get("random_zoning",False):
            json.dump(queries_by_rz7,open(self.exp["exp_dir"]+"queries_by_rz7.json","w"))
            json.dump(queries_by_rz11,open(self.exp["exp_dir"]+"queries_by_rz11.json","w"))

    def loadDocAndResolvableCitations(self, guid):
        """
            Deals with all the loading of the SciDoc.

            Throws ValueError if cannot load doc.

            :param guid: GUID of the doc to load
            :returns (doc, doctext, precomputed_file)
            :rtype: tuple
        """
        doc=cp.Corpus.loadSciDoc(guid) # load the SciDoc JSON from the cp.Corpus
        if not doc:
            raise ValueError("ERROR: Couldn't load pickled doc: %s" % guid)
            return None

        doctext=doc.getFullDocumentText() #  store a plain text representation

        # load the citations in the document that are resolvable, or generate if necessary
        citations_data=cp.Corpus.loadOrGenerateResolvableCitations(doc)
        return [doc,doctext,citations_data]

    def generateQueries(self, m, doc, doctext, precomputed_query):
        """
            Generate all queries for a resolvable citation.

            :param m: resolvable citation {cit, match_guid}
            :type m:dict
            :param doc: SciDoc
            :type doc:SciDoc
            :param doctext: full document text as rendered
            :type doctext:basestring
            :param precomputed_query: pre-filled dict
            :type precomputed_query:dict
        """
        queries={}
        generated_queries=[]

        match=findCitationInFullText(m["cit"],doctext)
        if not match:
            assert match
            print("Weird! can't find citation in text!!")
            return generated_queries

        # this is where we are in the document
        position=match.start()
        doc_position=math.floor((position/float(len(doctext)))*self.exp["numchunks"])+1

        # generate all the queries from the contexts
        for method_name in self.exp["qmethods"]:
            method=self.exp["qmethods"][method_name]

            params={
                "method_name":method_name,
                "match_start":match.start(),
                "match_end":match.end(),
                "doctext":doctext,
                "docfrom":doc,
                "cit":m["cit"],
                "dict_key":"text",
                "parameters":method["parameters"],
                "separate_by_tag":method.get("separate_by_tag",""),
                "options":{"jump_paragraphs":True}
                }

            all_queries=method["extractor"].extractMulti(params)
            for query in all_queries:
                queries[query["query_method_id"]]=query

        parent_s=doc.element_by_id[m["cit"]["parent_s"]]
        az=parent_s.get("az","")
        cfc=doc.citation_by_id[m["cit"]["id"]].get("cfunc","")
        csc_type=parent_s.get("csc_type","")

        # for every generated query for this context
        for qmethod in queries:
            this_query=deepcopy(precomputed_query)
            this_query["query_method"]=qmethod
            this_query["query_text"]=queries[qmethod].get("text","")
            this_query["structured_query"]=queries[qmethod]["structured_query"]
            this_query["doc_position"]=doc_position
            # TODO remove AZ/CSC classification if not necessary
            this_query["az"]=az
            this_query["cfc"]=cfc
            this_query["csc_type"]=csc_type

##            assert this_query["query_text"] != ""
            # for every method used for extracting BOWs
            generated_queries.append(this_query)

        return generated_queries

    def precomputeQueries(self,exp):
        """
            Precompute all queries for all annotated citation contexts

            :param exp: experiment dict with all options
            :type exp: dict
        """
        self.exp=exp
        logger=ResultsLogger(True, message_text="Precomputing queries") # init all the logging/counting
        logger.startCounting() # for timing the process, start now

        logger.setNumItems(len(exp["test_files"]))
        logger.numchunks=exp["numchunks"]

        cp.Corpus.loadAnnotators()

        # convert nested dict to flat dict where each method includes its parameters in the name
        all_doc_methods=getDictOfTestingMethods(exp["doc_methods"])
        # same as above
        tfidfmodels={}

        self.precomputed_queries=[]

        files_dict=OrderedDict()

        if exp["full_corpus"]:
            files_dict["ALL_FILES"]={}
            files_dict["ALL_FILES"]["doc_methods"]=all_doc_methods
            files_dict["ALL_FILES"]["tfidf_models"]=[]
            for method in all_doc_methods:
                actual_dir=cp.Corpus.getRetrievalIndexPath("ALL_FILES",all_doc_methods[method]["index_filename"],exp["full_corpus"])
                files_dict["ALL_FILES"]["tfidf_models"].append({"method":method,"actual_dir":actual_dir})

        #===================================
        # MAIN LOOP over all testing files
        #===================================
        for guid in exp["test_files"]:
            logger.showProgressReport(guid) # prints out info on how it's going

            try:
                doc,doctext,citations_data=self.loadDocAndResolvableCitations(guid)
            except ValueError:
                print("Can't load SciDoc ",guid)
                continue

            resolvable=citations_data["resolvable"] # list of resolvable citations
            in_collection_references=citations_data["outlinks"] # list of cited documents (refereces)

            # TODO do I really need all this information? Recomputed as well?
            num_in_collection_references=len(in_collection_references)
##            print ("Resolvable citations:",len(resolvable), "In-collection references:",num_in_collection_references)

            precomputed_file={"guid":guid,"in_collection_references":num_in_collection_references,
            "resolvable_citations":len(resolvable),}

            if not exp["full_corpus"]:
                precomputed_file["tfidf_models"]=[]
                for method in all_doc_methods:
                    # get the actual dir for each retrieval method, depending on whether full_corpus or not
                    actual_dir=cp.Corpus.getRetrievalIndexPath(guid,all_doc_methods[method]["index_filename"],exp["full_corpus"])
                    precomputed_file["tfidf_models"].append({"method":method,"actual_dir":actual_dir})

            files_dict[guid]=precomputed_file

            method_overlap_temp={}
            methods_overlap=0
            total_overlap_points=0

##            print("Precomputing and exporting citation contexts = queries...")
            for m in resolvable:
                precomputed_query={"file_guid":guid,
                                    "citation_id":m["cit"]["id"],
                                    "match_guid":m["match_guid"],
                                    "citation_multi": m["cit"].get("multi",1),
                                   }
                self.precomputed_queries.extend(self.generateQueries(m, doc, doctext, precomputed_query))

        self.saveAllQueries(files_dict)
        print("Precomputed citations saved.")


#===============================================================================
#  functions to measure score
#===============================================================================

##def analyticalRandomChanceMRR(numinlinks):
##    """
##        Returns the MRR score based on analytical random chance
##    """
##    res=0
##    for i in range(numinlinks):
##        res+=(1/float(numinlinks))*(1/float(i+1))
##    return res


##def addNewWindowQueryMethod(queries, name, method, match, doctext):
##    """
##        Runs a multi query generation function, adds all results with procedural
##        identifier to queries dict
##    """
##    all_queries= method["function"](match, doctext, method["parameters"], maxwords=20, options={"jump_paragraphs":True})
##    for cnt, p_size in enumerate(method["parameters"]):
##        method_name=name+str(p_size[0])+"_"+str(p_size[1])
##        queries[method_name]=all_queries[cnt]
##
##def addNewSentenceQueryMethod(queries, name, method, docfrom, cit, param):
##    """
##        Runs a multi query generation function, adds all results with procedural
##        identifier to queries dict
##    """
####    docfrom, cit, param, separate_by_tag=None, dict_key="text")
##
##    all_queries= method["function"](docfrom, cit, method["parameters"], maxwords=20, options={"jump_paragraphs":True})
##    for cnt, param in enumerate(method["parameters"]):
##        method_name=name+"_"+param
##        queries[method_name]=all_queries[cnt]["text"]


def testPrebuilt():
    """
        Tries to load a prebuilt bow to visualize its contents
    """
    bla=loadPrebuiltBOW("c92-2117.xml","inlink_context",50)
    print bla

def createExplainQueriesByAZ(retrieval_results_filename="prr_all_ilc_AZ_paragraph_ALL.json"):
    """
        Creates _by_az and _by_cfc files from precomputed_retrieval_results
    """
    retrieval_results=json.load(open(cp.Corpus.dir_prebuiltBOWs+retrieval_results_filename,"r"))
    retrieval_results_by_az={zone:[] for zone in AZ_ZONES_LIST}

    for retrieval_result in retrieval_results:
        retrieval_results_by_az[retrieval_result["az"]].append(retrieval_result)
##        retrieval_results_by_cfc[retrieval_result["query"]["cfc"]].append(retrieval_result)

    json.dump(retrieval_results_by_az,open(cp.Corpus.dir_prebuiltBOWs+"retrieval_results_by_az.json","w"))
##    json.dump(retrieval_results_by_cfc,open(cp.Corpus.dir_prebuiltBOWs+"retrieval_results_by_cfc.json","w"))


def fixEndOfFile():
    """
        Fixing bad JSON or dumping
    """
    exp={}
    exp["exp_dir"]=r"C:\NLP\PhD\bob\experiments\w20_csc_fa_w0135"+os.sep

    files={}
    AZ_LIST=[zone for zone in AZ_ZONES_LIST if zone != "OWN"]
    print AZ_LIST

    for div in AZ_LIST:
        files["AZ_"+div]=open(exp["exp_dir"]+"prr_AZ_"+div+".json","r+")

##    for div in CORESC_LIST:
##        files["CSC_"+div]=open(exp["exp_dir"]+"prr_CSC_"+div+".json","r+")

    files["ALL"]=open(exp["exp_dir"]+"prr_ALL.json","r+")

    for div in AZ_LIST:
        files["AZ_"+div].seek(-1,os.SEEK_END)
        files["AZ_"+div].write("]")

##    for div in CORESC_LIST:
##        files["CSC_"+div].seek(-1,os.SEEK_END)
##        files["CSC_"+div].write("]")

##    files["ALL"].seek(-1,os.SEEK_END)
##    files["ALL"].write("]")



def main():
# new trials
##    methods={
##    "full_text":{"type":"standard_multi", "parameters":[1]},
##    "passage":{"type":"standard_multi", "parameters":[150, 250, 350]},
##    "title_abstract":{"type":"standard_multi", "parameters":[1]},
##    "inlink_context":{"type":"standard_multi", "parameters": [5, 10, 20, 30]},
##    "ilc_title_abstract":{"type":"ilc_mashup", "mashup_method":"title_abstract", "ilc_parameters":[5, 10,20], "parameters":[1]},
##    "ilc_full_text":{"type":"ilc_mashup", "mashup_method":"full_text", "ilc_parameters":[5,10,20], "parameters":[1]},
##    "ilc_passage":{"type":"ilc_mashup", "mashup_method":"passage","ilc_parameters":[5, 10, 20], "parameters":[250,350]}
##    }
##    methods={
##    "inlink_context":{"type":"standard_multi", "parameters": [10, 20, 30]},
##    }

# AZ ones
    testing_methods={
##    "full_text":{"type":"standard_multi", "index":"full_text", "parameters":[1], "runtime_parameters":["text"]},
##    "title_abstract":{"type":"standard_multi", "index":"title_abstract", "parameters":[1], "runtime_parameters":{"text":"1"}},
##    "passage":{"type":"standard_multi", "index":"passage", "parameters":[250,350,400], "runtime_parameters":{"text":"1"}},

##    "inlink_context":{"type":"standard_multi", "index":"inlink_context",
##        "parameters": [10, 20, 30], "runtime_parameters":{"inlink_context":"1"}},

##    "inlink_context_year":{"type":"standard_multi", "index":"inlink_context_year",
##        "parameters": [10, 20, 30], "runtime_parameters":{"inlink_context":"1"}},

##    "ilc_passage":{"type":"ilc_mashup",  "index":"ilc_passage", "mashup_method":"passage","ilc_parameters":[10, 20, 30, 40, 50],
##        "parameters":[250,350], "runtime_parameters":{"text":"1","inlink_context":"1"}},

##    "az_annotated":{"type":"annotated_boost", "index":"az_annotated", "parameters":[1], "runtime_parameters":
##        {"ALL":["AIM","BAS","BKG","CTR","OTH","OWN","TXT"]
##        }},

##    "section":{"type":"annotated_boost", "index":"section_annotated", "parameters":[1], "runtime_parameters":
##        {
####        "title_abstract":{"title":"1","abstract":"1"},
##         "full_text":["title","abstract","text"],
##        }},

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
##        {"ALL":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT"]}
##        },

    "ilc_AZ":{"type":"annotated_boost", "index":"ilc_AZ", "parameters":["paragraph"], "runtime_parameters":
##        {"AZ":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT"]}
        {"CSC": ["ilc_CSC_"+zone for zone in CORESC_LIST]}
        },

    # this is sentence-based AZ and AZ-annotated document contents
##    "ilc_az_ilc_az":{"type":"ilc_annotated_boost", "index":"ilc_AZ", "parameters":[1],
##        "ilc_parameters":["1only","1up","1up1down","paragraph"],
##        "runtime_parameters":
##        {
##        "ALL":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT","ilc_AZ_AIM"],
##         "OTH":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"1","OWN":"0","TXT":"0","inlink_context":1},
##         "OWN":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"0","OWN":"1","TXT":"0","inlink_context":1},
##        }},

    }

    # this is the dict of query extraction methods
    qmethods={"window":{"parameters":[
##                (3,3),
##                (5,5),
##                (10,10),
##                (5,10),
##                (10,5),
                (20,20),
##                (20,10),
##                (10,20),
##                (30,30)
                ],
                "function":doc_representation.getOutlinkContextWindowAroundCitationMulti,
                "type":"window"},

##            "sentence":{"parameters":[
##                "1only",
##                "paragraph",
##                "1up",
##                "1up_1down"
##                ],
##                "function":doc_representation.getOutlinkContextSentences,
##                "type":"sentence"}
                }



    # 1. automatically get list of papers to test on with SQL query
    cp.Corpus.TEST_FILES=cp.Corpus.listPapers("num_in_collection_references >= 8")

    # 2a. uncomment this to generate the queries and files_dict for Citation Resolution
##    precomputeQueries(cp.Corpus.TEST_FILES,20,testing_methods, qmethods)

    # (x) this is just testing for year in ILC - tested, little difference
##    precomputeQueries(cp.Corpus.TEST_FILES,20,testing_methods, qmethods, files_dict_filename="ilc_year_test_file_dict.json",
##    filename="ilc_year_precomputed_queries.json")
##    precomputeQueries(cp.Corpus.TEST_FILES,20,testing_methods, qmethods, files_dict_filename="sentence_test_file_dict.json",
##    filename="sentence_precomputed_queries.json")

    # 2b. uncomment this to generate the queries and files_dict for Full cp.Corpus
##    precomputeQueries(cp.Corpus.TEST_FILES,20,testing_methods,qmethods, files_dict_filename="files_dict_full_corpus.json", full_corpus=True)

    # (x) this is the old Citation Resolution run, just for testing now
##    runPrecomputedCitationResolutionLucene("results_full_corpus_test1.csv", files_dict_filename="files_dict_full_corpus.json",
##    full_corpus=True, testing_methods=testing_methods)

##    runPrecomputedCitationResolutionLucene("overlap_bulkScorer_explain.csv", files_dict_filename="files_dict.json", full_corpus=False)

##    runPrecomputedCitationResolutionLucene("results_sentence_test2.csv", files_dict_filename="sentence_test_file_dict.json", full_corpus=False, testing_methods=testing_methods, precomputed_queries_filename="sentence_precomputed_queries.json")

    # 3. uncomment this to precompute the Lucene explanations
##    precomputeExplainQueries(testing_methods,retrieval_results_filename="CSC",use_default_similarity=False)
##    createExplainQueriesByAZ()

    # each parameter is a zone to process
    zones_to_process=["OWN"]
    if len(sys.argv) > 1:
        zones_to_process=sys.argv[1:]


    # 4a. uncomment to run all combinations of weights to find the best
##    autoTrainWeightValues_optimized(zones_to_process,testing_methods=testing_methods,filename_add="ilc_az_fa_1000_1",split_set=1)
##    autoTrainWeightValues_optimized(zones_to_process,testing_methods=testing_methods,filename_add="ilc_az_fa_1000_2",split_set=2)
##    autoTrainWeightValues_optimized(zones_to_process,filename_add="ilc_az_fa_second1000",split_set=2)
##    autoTrainWeightValues_optimized(zones_to_process,filename_add="test_optimized_fa_third1000",split_set=3)

    # 4b. this is to run without optimization
##    autoTrainWeightValues(zones_to_process, filename_add="_FA_BulkScorer_first650")
##    autoTrainWeightValues(zones_to_process, filename_add="_FA_BulkScorer_second650")

##    testWeightCounter()

    pass

if __name__ == '__main__':
    main()
