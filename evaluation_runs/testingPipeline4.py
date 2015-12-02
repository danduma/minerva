#-------------------------------------------------------------------------------
# Name:        exp
# Purpose:      experiments
#
# Author:      Daniel Duma
#
# Created:     08/12/2013
# Copyright:   (c) Daniel Duma 2013
#-------------------------------------------------------------------------------

import context_extract

import lucene

from pandas import *

import glob, math, os, datetime, re, copy, sys, gc
from collections import defaultdict, namedtuple, OrderedDict
from results_logging import progressIndicator,resultsLogger
##import heapq
from az_cfc_classification import AZ_ZONES_LIST
from general_utils import getSafeFilename

from lucene_retrieval import luceneRetrieval,luceneRetrievalBoost,storedFormula,precomputedExplainRetrieval
from corpora import Corpus

def formatReferenceAPA(reference):
    """
        Formats a reference/metadata in plain text APA
    """
    return ref.get("authors","")+". "+ref.get("year","")+". "+ref.get("title","")+ ". "+ref.get("publication","")


class weightCounterList:
    """
        Deals with iterating over combinations of weights for training
    """
    def __init__(self, values=[1,3,5]):
        """
            values = list of values each weight can take
        """
        self.values=values
        self.counters=[]
        self.MAX_COUNTER=len(values)-1

    def numCombinations(self):
        """
            Returns the number of total possible combinations of this counter
        """
        return pow(len(self.values),len(self.counters))

    def initWeights(self,parameters):
        """
            Creates counters and weights from input list of parameters to change
        """
        self.parameters=parameters
        # create a dict where every field gets a weight of x
        self.weights={x:self.values[0] for x in parameters}
        # make a counter for each field
        self.counters=[0 for x in range(len(self.weights))]

    def allOnes(self):
        """
            Return all ones
        """
        return {x:1 for x in self.parameters}

    def allCombinationsProcessed(self):
        """
            Returns false if there is some counter that is not at its max
        """
        for counter in self.counters:
            if counter < self.MAX_COUNTER:
                return False
        return True

    def nextCombination(self):
        """
            Increases the counters and adjusts weights as necessary
        """
        self.counters[0]+=1

        for index in range(len(self.counters)):
            if self.counters[index] > self.MAX_COUNTER:
                self.counters[index+1] += 1
                self.counters[index]=0
            self.weights[self.weights.keys()[index]]=self.values[self.counters[index]]

    def getPossibleValues(self):
        return self.values


#===============================================================================
#  functions to measure score
#===============================================================================

def analyticalRandomChanceMRR(numinlinks):
    """
    """
    res=0
    for i in range(numinlinks):
        res+=(1/float(numinlinks))*(1/float(i+1))
    return res


def addNewWindowQueryMethod(queries, name, method, match, doctext):
    """
        Runs a multi query generation function, adds all results with procedural
        identifier to queries dict
    """
    all_queries= method["function"](match, doctext, method["parameters"], maxwords=20, options={"jump_paragraphs":True})
    for cnt, p_size in enumerate(method["parameters"]):
        method_name=name+str(p_size[0])+"_"+str(p_size[1])
        queries[method_name]=all_queries[cnt]

def addNewSentenceQueryMethod(queries, name, method, docfrom, cit, param):
    """
        Runs a multi query generation function, adds all results with procedural
        identifier to queries dict
    """
##    docfrom, cit, param, separate_by_tag=None, dict_key="text")

    all_queries= method["function"](docfrom, cit, method["parameters"], maxwords=20, options={"jump_paragraphs":True})
    for cnt, param in enumerate(method["parameters"]):
        method_name=name+"_"+param
        queries[method_name]=all_queries[cnt]["text"]

def generateRetrievalModels(all_doc_methods, all_files, files_dict={}, full_corpus=False):
    """
        Generates the files_dict with the paths to the retrieval models
    """
    for guid in all_files:
        files_dict[guid]["tfidf_models"]=[]
        for method in all_doc_methods:
            actual_dir=Corpus.getLuceneIndexPath(guid,all_doc_methods[method]["index_filename"],full_corpus)
            files_dict[guid]["tfidf_models"].append({"method":method,"actual_dir":actual_dir})


#===============================================================================
#   TESTING PIPELINE
#===============================================================================

def runPrecomputedCitationResolutionLucene(output_filename="results.csv", precomputed_queries_filename="precomputed_queries.json",
files_dict_filename="files_dict.json", testing_methods=None, runtime_parameters=None, logging=True, full_corpus=False, compare_explain=False):
    """
        Using Lucene, run Citation Resolution
        Load everything from precomputed queries
    """
    logger=resultsLogger(False) # init all the logging/counting
    logger.startCounting() # for timing the process, start now

    lucene.initVM(maxheap="800m") # init Lucene VM
    logger.output_filename=Corpus.output_dir+output_filename

    precomputed_queries=json.load(open(Corpus.prebuiltBOWs_dir+precomputed_queries_filename,"r"))
    files_dict=json.load(open(Corpus.prebuiltBOWs_dir+files_dict_filename,"r"))

    logger.setNumItems(len(precomputed_queries))

    tfidfmodels={}
    all_doc_methods=None

    if testing_methods:
        all_doc_methods=context_extract.getDictOfTestingMethods(testing_methods)
        # essentially this overrides whatever is in files_dict, if testing_methods was passed as parameter

        if full_corpus:
            all_files=["ALL_FILES"]
        else:
            all_files=files_dict.keys()

        generateRetrievalModels(all_doc_methods,all_files,files_dict,full_corpus)
    else:
        all_doc_methods=files_dict["ALL_FILES"]["doc_methods"] # load from files_dict

    if full_corpus:
        for model in files_dict["ALL_FILES"]["tfidf_models"]:
            # create a Lucene search instance for each method
            tfidfmodels[model["method"]]=luceneRetrievalBoost(model["actual_dir"],model["method"],logger=None)

    main_all_doc_methods=copy.deepcopy(all_doc_methods)

    if compare_explain:
        for method in main_all_doc_methods:
            all_doc_methods[method+"_EXPLAIN"]=all_doc_methods[method]

    main_all_doc_methods=copy.deepcopy(all_doc_methods)

    # this is for counting overlaps only
    methods_overlap=0
    total_overlap_points=0
    rank_differences=[]
    rank_per_method=defaultdict(lambda:[])
    precision_per_method=defaultdict(lambda:[])
    previous_guid=""

    #===================================
    # MAIN LOOP over all testing files
    #===================================
    for precomputed_query in precomputed_queries:
        guid=precomputed_query["file_guid"]
        logger.total_citations+=files_dict[guid]["resolvable_citations"]

        all_doc_methods=copy.deepcopy(main_all_doc_methods)

        if (not full_corpus and guid != previous_guid):
            previous_guid=guid

            key="ALL_FILES" if full_corpus else guid

            for model in files_dict[key]["tfidf_models"]:
                # create a Lucene search instance for each method
                tfidfmodels[model["method"]]=luceneRetrievalBoost(model["actual_dir"],model["method"],logger=None)
                # this is to compare bulkScorer and .explain() on their overlap
                if compare_explain:
                    tfidfmodels[model["method"]+"_EXPLAIN"]=luceneRetrievalBoost(model["actual_dir"],model["method"],logger=None)
                    tfidfmodels[model["method"]+"_EXPLAIN"].useExplainQuery=True

        # create a dict where every field gets a weight of 1
        for method in main_all_doc_methods:
            all_doc_methods[method]["runtime_parameters"]={x:1 for x in main_all_doc_methods[method]["runtime_parameters"]}

        if runtime_parameters:
            for method in runtime_parameters:
                all_doc_methods[method]["runtime_parameters"]=runtime_parameters[method]

##        assert(logger.item_counter < 180)

        # for every method used for extracting BOWs
        for doc_method in all_doc_methods:
            # ACTUAL RETRIEVAL HAPPENING - run query
            logger.logReport("Citation: "+precomputed_query["citation_id"]+"\n Query method:"+precomputed_query["query_method"]+" \nDoc method: "+doc_method +"\n")
            logger.logReport(precomputed_query["query_text"]+"\n")

            retrieved=tfidfmodels[doc_method].runQuery(precomputed_query["query_text"],all_doc_methods[doc_method]["runtime_parameters"], guid)

            result_dict={"file_guid":guid,
            "citation_id":precomputed_query["citation_id"],
            "doc_position":precomputed_query["doc_position"],
            "query_method":precomputed_query["query_method"],
            "doc_method":doc_method ,
            "az":precomputed_query["az"],
            "cfc":precomputed_query["cfc"],
            "match_guid":precomputed_query["match_guid"]}

            if not retrieved:    # the query was empty or something
                score=0
                precision_score=0
##                        print "Error: ", doc_method , qmethod,tfidfmodels[method].indexDir
##                        logger.addResolutionResult(guid,m,doc_position,qmethod,doc_method ,0,0,0)
                result_dict["mrr_score"]=0
                result_dict["precision_score"]=0
                result_dict["ndcg_score"]=0
                result_dict["rank"]=0
                result_dict["first_result"]=""

                logger.addResolutionResultDict(result_dict)

            else:
                result=logger.measureScoreAndLog(retrieved, precomputed_query["citation_multi"], result_dict)
                rank_per_method[result["doc_method"]].append(result["rank"])
                precision_per_method[result["doc_method"]].append(result["precision_score"])

            # DO SOMETHING TO LOG TEXT AND LIST OF REFERENCES ETC
##                    pre_selection_text=doctext[queries[qmethod]["left_start"]-300:queries[qmethod]["left_start"]]
##                    draft_text=doctext[queries[qmethod]["left_start"]:queries[qmethod]["right_end"]]
##                    post_selection_text=doctext[queries[qmethod]["right_end"]:queries[qmethod]["left_start"]+300]
##                    draft_text=u"<span class=document_text>{}</span> <span class=selected_text>{}</span> <span class=document_text>{}</span>".format(pre_selection_text, draft_text, post_selection_text)
##                    print draft_text


        # extra method = random chance control
##        logger.addResolutionResultDict({"file_guid":guid,
##            "citation_id":precomputed_query["citation_id"],
##            "doc_position":precomputed_query["doc_position"],
##            "query_method":precomputed_query["query_method"],
##            "doc_method":doc_method ,
##            "az":precomputed_query["az"],
##            "cfc":precomputed_query["cfc"],
##            "match_guid":precomputed_query["match_guid"],
##            "doc_method":"RANDOM",
##            "mrr_score":analyticalRandomChanceMRR(files_dict[guid]["in_collection_references"]),
##            "precision_score":1/float(files_dict[guid]["in_collection_references"]),
##            "ndcg_score":0,
##            "rank":0,
##            "first_result":""
##            })
        logger.showProgressReport(guid) # prints out info on how it's going
##        rank_diff=abs(rank_per_method["section_1_full_text"][-1]-rank_per_method["full_text_1"][-1])
##                if method_overlap_temp["section_1_full_text"] == method_overlap_temp["full_text_1"]
##        if rank_diff == 0:
##            methods_overlap+=1
##        rank_differences.append(rank_diff)
##        total_overlap_points+=1

    logger.writeDataToCSV()

##    logger.showFinalSummary(overlap_in_methods=all_doc_methods.keys())
    logger.showFinalSummary()
##    print "Total rank overlap between [section_1_full_text] and [full_text_1] = ",methods_overlap,"/",total_overlap_points," = {}".format(methods_overlap / float(total_overlap_points),"2.3f")
##    print "Avg rank difference between [section_1_full_text] and [full_text_1] = {}".format(sum(rank_differences) / float(total_overlap_points),"2.3f")
##    print "Avg rank:"
##    for method in rank_per_method:
##        print method,"=",sum(rank_per_method[method])/float(len(rank_per_method[method]))

def precomputeQueries(FILE_LIST, numchunks=10, bowmethods={}, querymethods={}, full_corpus=False,
    filename="precomputed_queries.json", files_dict_filename="files_dict.json"):
    """
        Using Lucene, run Citation Resolution
    """
    logger=resultsLogger(True) # init all the logging/counting
    logger.startCounting() # for timing the process, start now

    logger.setNumItems(len(FILE_LIST))
    logger.numchunks=numchunks

    Corpus.loadAnnotators()

    all_doc_methods=context_extract.getDictOfTestingMethods(bowmethods) # convert nested dict to flat dict where each method includes its parameters in the name
    tfidfmodels={} # same as above

    precomputed_queries=[]

    files_dict=OrderedDict()
    files_dict["ALL_FILES"]={}
    files_dict["ALL_FILES"]["doc_methods"]=all_doc_methods
    if full_corpus:
        files_dict["ALL_FILES"]["tfidf_models"]=[]
        for method in all_doc_methods:
            actual_dir=Corpus.getLuceneIndexPath("ALL_FILES",all_doc_methods[method]["index_filename"],full_corpus)
            files_dict["ALL_FILES"]["tfidf_models"].append({"method":method,"actual_dir":actual_dir})

    #===================================
    # MAIN LOOP over all testing files
    #===================================
    for guid in FILE_LIST:
        logger.showProgressReport(guid) # prints out info on how it's going

        doc=Corpus.loadSciDoc(guid) # load the SciDoc JSON from the corpus
        if not doc:
            print "ERROR: Couldn't load pickled doc:",guid
            continue
        doctext=doc.getFullDocumentText() #  store a plain text representation

        # load the citations in the document that are resolvable, or generate if necessary
        tin_can=Corpus.getResolvableCitationsCache(guid, doc)
        resolvable=tin_can["resolvable"] # list of resolvable citations
        in_collection_references=tin_can["outlinks"] # list of cited documents (refereces)

        num_in_collection_references=len(in_collection_references)
        print "Resolvable citations:",len(resolvable), "In-collection references:",num_in_collection_references

        precomputed_file={"guid":guid,"in_collection_references":num_in_collection_references,
        "resolvable_citations":len(resolvable),}

        if not full_corpus:
            precomputed_file["tfidf_models"]=[]
            for method in all_doc_methods:
                # get the actual dir for each retrieval method, depending on whether full_corpus or not
                actual_dir=Corpus.getLuceneIndexPath(guid,all_doc_methods[method]["index_filename"],full_corpus)
                precomputed_file["tfidf_models"].append({"method":method,"actual_dir":actual_dir})

        files_dict[guid]=precomputed_file

        Corpus.annotators["AZ"].annotateDoc(doc)
        Corpus.annotators["CFC"].annotateDoc(doc)

        method_overlap_temp={}
        methods_overlap=0
        total_overlap_points=0

        print "Precomputing and exporting citations..."
        for m in resolvable:
            queries={}

            match=context_extract.findCitationInFullText(m["cit"],doctext)
            if not match:
                print "Weird! can't find citation in text!!"
                continue

            # this is where we are in the document
            position=match.start()
            doc_position=math.floor((position/float(len(doctext)))*numchunks)+1

            # generate all the queries from the contexts
            for method_name in querymethods:
                method=querymethods[method_name]

                if method["type"]=="window":
##                    addNewWindowQueryMethod(queries,method_name,method,match,doctext)

                    all_queries= method["function"](match, doctext, method["parameters"], options={"jump_paragraphs":True})
                    for cnt, p_size in enumerate(method["parameters"]):
                        this_method_name=method_name+str(p_size[0])+"_"+str(p_size[1])
                        queries[this_method_name]=all_queries[cnt]

                elif method["type"]=="sentence":
                    # !TODO Implement sentence as window extraction
##                    addNewSentenceQueryMethod(queries, m["cit"])
                    for param in method["parameters"]:
                        this_method_name=method_name+"_"+param
                        queries[this_method_name]=method["function"](doc, m["cit"], param, dict_key="text")
                else:
##                    assert(False,"window method type not implemented")
                    print "window method type not implemented"

            parent_s=doc.element_by_id[m["cit"]["parent_s"]]
            az=parent_s["az"] if parent_s.has_key("az") else ""
            cfc=doc.citation_by_id[m["cit"]["id"]]["cfunc"] if doc.citation_by_id[m["cit"]["id"]].has_key("cfunc") else ""

            # for every generated query for this context
            for qmethod in queries:
                precomputed_query={"file_guid":guid,
                    "citation_id":m["cit"]["id"],
                    "doc_position":doc_position,
                    "query_method":qmethod,
                    "query_text":queries[qmethod]["text"],
                    "az":az,
                    "cfc":cfc,
                    "match_guid":m["match_guid"],
                    "citation_multi": 1 if not m["cit"].has_key("multi") else m["cit"]["multi"],
                   }

                # for every method used for extracting BOWs
                precomputed_queries.append(precomputed_query)

    json.dump(precomputed_queries,open(Corpus.prebuiltBOWs_dir+filename,"w"))
    queries_by_az={zone:[] for zone in AZ_ZONES_LIST}
    queries_by_cfc=defaultdict(lambda:[])

    for precomputed_query in precomputed_queries:
        queries_by_az[precomputed_query["az"]].append(precomputed_query)
        queries_by_cfc[precomputed_query["cfc"]].append(precomputed_query)

    json.dump(files_dict,open(Corpus.prebuiltBOWs_dir+files_dict_filename,"w"))
    json.dump(queries_by_az,open(Corpus.prebuiltBOWs_dir+"queries_by_az.json","w"))
    json.dump(queries_by_cfc,open(Corpus.prebuiltBOWs_dir+"queries_by_cfc.json","w"))
    print "Precomputed citations saved."
##    logfile=open("results.txt","wb")
##    logfile.writelines(results)

##    writeTuplesToCSV("Filename citation_id doc_section query_method doc_method MRR_score NDCG_score Precision_score Correct_file".split(),overall_results,Corpus.output_dir+output_filename)


def createExplainQueriesByAZ(retrieval_results_filename="precomputed_retrieval_results.json"):
    """
        Creates _by_az and _by_cfc files from precomputed_retrieval_results
    """
    retrieval_results=json.load(open(Corpus.prebuiltBOWs_dir+retrieval_results_filename,"r"))
    retrieval_results_by_az={zone:[] for zone in AZ_ZONES_LIST}

    for retrieval_result in retrieval_results:
        retrieval_results_by_az[retrieval_result["az"]].append(retrieval_result)
##        retrieval_results_by_cfc[retrieval_result["query"]["cfc"]].append(retrieval_result)

    json.dump(retrieval_results_by_az,open(Corpus.prebuiltBOWs_dir+"retrieval_results_by_az.json","w"))
##    json.dump(retrieval_results_by_cfc,open(Corpus.prebuiltBOWs_dir+"retrieval_results_by_cfc.json","w"))


def precomputeExplainQueries(testing_methods, files_dict_filename="files_dict.json", full_corpus=False,
retrieval_results_filename="precomputed_retrieval_results.json", use_default_similarity=False):
    """
        Run the whole CitationResolution thing, store the results for each citation
        to then just test different weight values for overall or per-citation testing
    """
    test_logger=resultsLogger(False)
    test_logger.startCounting()

    lucene.initVM(maxheap="768m") # init Lucene VM
    files_dict=json.load(open(Corpus.prebuiltBOWs_dir+files_dict_filename,"r"))
    precomputed_queries=json.load(open(Corpus.prebuiltBOWs_dir+"precomputed_queries.json","r"))

##    queries_by_az=json.load(open(Corpus.prebuiltBOWs_dir+"queries_by_az.json","r"))
##    queries_by_cfc=json.load(open(Corpus.prebuiltBOWs_dir+"queries_by_cfc.json","r"))

##    print "AZs:"
##    for key in queries_by_az:
##        print key, len(queries_by_az[key])
##
##    print ""
##    print "CFCs:"
##    for key in queries_by_cfc:
##        print key, len(queries_by_cfc[key])
##    print ""

    results=[]

    if testing_methods:
        all_doc_methods=context_extract.getDictOfTestingMethods(testing_methods)
    else:
        all_doc_methods=files_dict["ALL_FILES"]["doc_methods"] # convert nested dict to flat

    annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"]=="annotated_boost"]

    logger=resultsLogger(False) # init all the logging/counting
    logger.startCounting() # for timing the process, start now

    logger.setNumItems(len(precomputed_queries),print_out=False,dot_every_xitems=20)

    tfidfmodels={}
    actual_tfidfmodels={}

    # if we're running over the full corpus, we should only load the indices once
    # at the beginning, as they're going to be the same
    # !TODO fixme: change it so we don't need the model in files_dict
    if full_corpus:
##        for model in files_dict["ALL_FILES"]["tfidf_models"]:
        for method in all_doc_methods:
            # create a Lucene search instance for each method
##            tfidfmodels[model["method"]]=precomputedExplainRetrieval(model["actual_dir"],model["method"],
##            logger=None,use_default_similarity=use_default_similarity)
            actual_dir=Corpus.getLuceneIndexPath("ALL_FILES",all_doc_methods[method]["index_filename"],full_corpus)
            tfidfmodels[method]=precomputedExplainRetrieval(actual_dir,method,logger=None,use_default_similarity=use_default_similarity)

    previous_guid=""
    logger.total_citations=len(precomputed_queries)

    print "Running Lucene to precompute retrieval results..."
    results=[]
    retrieval_results_by_az={zone:[] for zone in AZ_ZONES_LIST}
    #===================================
    # MAIN LOOP over all testing files
    #===================================
    for query in precomputed_queries:
        # for every generated query for this context
        guid=query["file_guid"]

        # if this is not a full_corpus run and we've moved on to the next test file
        # then we should load the indices for this new file
        if not full_corpus and guid != previous_guid:
            previous_guid=guid
            for method in all_doc_methods:
                # create a Lucene search instance for each method
                actual_dir=Corpus.getLuceneIndexPath(guid,all_doc_methods[method]["index_filename"],full_corpus)
                tfidfmodels[method]=precomputedExplainRetrieval(actual_dir,method,logger=None,use_default_similarity=use_default_similarity)
##                actual_tfidfmodels[model["method"]]=luceneRetrievalBoost(model["actual_dir"],model["method"],logger=None,use_default_similarity=use_default_similarity)

        retrieval_result=copy.deepcopy(query)
        retrieval_result["results_by_doc_method"]={}
        del retrieval_result["query_text"]
##        retrieval_result={"query":query,"results_by_doc_method":{}}
        # for every method used for extracting BOWs
        for doc_method in all_doc_methods:
            # ACTUAL RETRIEVAL/EXPLAIN HAPPENING - run query
            param_dict={x:1 for x in all_doc_methods[doc_method]["runtime_parameters"]}
            formulas=tfidfmodels[doc_method].precomputeExplain(query["query_text"],param_dict, guid)
            retrieval_result["results_by_doc_method"][doc_method]=formulas


##            result_dict1={"match_guid":retrieval_result["match_guid"],"query_method":retrieval_result["query_method"],"doc_method":doc_method}
##            result_dict2={"match_guid":retrieval_result["match_guid"],"query_method":retrieval_result["query_method"],"doc_method":doc_method}
##            result_dict1=test_logger.measureScoreAndLog(runPrecomputedQuery(formulas,param_dict),retrieval_result["citation_multi"],result_dict1)
##            retrieved=actual_tfidfmodels[doc_method].runQuery(query["query_text"],param_dict,guid)
##            result_dict2=test_logger.measureScoreAndLog(retrieved,retrieval_result["citation_multi"],result_dict2)
##            assert(result_dict1["precision_score"]==result_dict2["precision_score"])
##
##            param_dict={x:y for y,x in enumerate(all_doc_methods[doc_method]["runtime_parameters"])}
##            result_dict1={"match_guid":retrieval_result["match_guid"],"query_method":retrieval_result["query_method"],"doc_method":doc_method}
##            result_dict2={"match_guid":retrieval_result["match_guid"],"query_method":retrieval_result["query_method"],"doc_method":doc_method}
##            result_dict1=test_logger.measureScoreAndLog(runPrecomputedQuery(formulas,param_dict),retrieval_result["citation_multi"],result_dict1)
##            retrieved=actual_tfidfmodels[doc_method].runQuery(query["query_text"],param_dict,guid)
##            result_dict2=test_logger.measureScoreAndLog(retrieved,retrieval_result["citation_multi"],result_dict2)
##            assert(result_dict1["precision_score"]==result_dict2["precision_score"])

        results.append(retrieval_result)
        retrieval_results_by_az[retrieval_result["az"]].append(retrieval_result)
        logger.showProgressReport(guid) # prints out info on how it's going

##    json.dump(results,open(Corpus.prebuiltBOWs_dir+retrieval_results_filename,"w"))
    manualJSONdump(results,open(Corpus.prebuiltBOWs_dir+retrieval_results_filename,"w"))

##    for retrieval_result in retrieval_results:
##        retrieval_results_by_az[retrieval_result["query"]["az"]].append(retrieval_result)
##        retrieval_results_by_cfc[retrieval_result["query"]["cfc"]].append(retrieval_result)
    del results
    gc.collect()
    createExplainQueriesByAZ()
##    json.dump(retrieval_results_by_az,open(Corpus.prebuiltBOWs_dir+"retrieval_results_by_az.json","w"))

    return None


def runPrecomputedQuery(retrieval_result, parameters):
    """
        This takes a query that has already had the results added
    """
    scores=[]
    for unique_result in retrieval_result:
        formula=storedFormula(unique_result["formula"])
        score=formula.computeScore(parameters)
        scores.append((score,{"guid":unique_result["guid"]}))

    scores.sort(key=lambda x:x[0],reverse=True)
    return scores


def measurePrecomputedResolution(retrieval_results,method,parameters, all_doc_methods,citation_az="*"):
    """
        This is kind of like measureCitationResolution:
        it takes a list of precomputed retrieval_results, then applies the new parameters
        to them. This is how we recompute what Lucene gives us, avoing having to call Lucene again.

        All we need to do is adjust the weights on the already available explanation
        formulas.
    """
    logger=resultsLogger(False) # init all the logging/counting
    logger.startCounting() # for timing the process, start now

    logger.setNumItems(len(retrieval_results),print_out=False)

    # for each query-result: (results are packed inside each query for each method)
    for result in retrieval_results:
        # select only the method we're testing for
        res=result["results_by_doc_method"][method]
        retrieved=runPrecomputedQuery(res,parameters)

        result_dict={"file_guid":result["file_guid"],
        "citation_id":result["citation_id"],
        "doc_position":result["doc_position"],
        "query_method":result["query_method"],
        "doc_method":method,
        "az":result["az"],
        "cfc":result["cfc"],
        "match_guid":result["match_guid"]}

        if not retrieved or len(retrieved)==0:    # the query was empty or something
            score=0
            precision_score=0
##                        print "Error: ", doc_method , qmethod,tfidfmodels[method].indexDir
##                        logger.addResolutionResult(guid,m,doc_position,qmethod,doc_method ,0,0,0)
            result_dict["mrr_score"]=0
            result_dict["precision_score"]=0
            result_dict["ndcg_score"]=0
            result_dict["rank"]=0
            result_dict["first_result"]=""

            logger.addResolutionResultDict(result_dict)
        else:
            result=logger.measureScoreAndLog(retrieved, result["citation_multi"], result_dict)

    logger.computeAverageScores()
    results=[]
    for query_method in logger.averages:
        for doc_method in logger.averages[query_method]:
            weights=all_doc_methods[doc_method]["runtime_parameters"]
            data_line={"query_method":query_method,"doc_method":doc_method,"citation_az":citation_az}

            for metric in logger.averages[query_method][doc_method]:
                data_line["avg_"+metric]=logger.averages[query_method][doc_method][metric]
            data_line["precision_total"]=logger.scores["precision"][query_method][doc_method]

            signature=""
            for w in weights:
                data_line[w]=weights[w]
                signature+=str(w)

            data_line["weight_signature"]=signature
            results.append(data_line)

##    logger.writeDataToCSV(Corpus.output_dir+"testing_test_precision.csv")

    return results


def measureCitationResolution(files_dict, precomputed_queries, all_doc_methods, citation_az, testing_methods, full_corpus=False):
    """
        Use Citation Resolution to measure the impact of different runtime parameters.

        files_dict is a dict where each key is a guid:
            "j97-3003": {"tfidf_models": [{"actual_dir": "C:\\NLP\\PhD\\bob\\fileDB\\LuceneIndeces\\j97-3003\\ilc_az_annotated_1_20",
			"method": "az_annotated_1_ALL"}],
		"resolvable_citations": 16,
		"doc_methods": {"az_annotated_1_ALL": {
				"index": "ilc_az_annotated_1_20",
				"runtime_parameters": ["AIM",
				"BAS",
				"BKG",
				"CTR",
				"OTH",
				"OWN",
				"TXT"],
				"parameters": [1],
				"type": "annotated_boost",
				"index_filename": "ilc_az_annotated_1_20",
				"parameter": 1,
				"method": "az_annotated"
			}
		},
		"guid": "j97-3003",
		"in_collection_references": 10}
    """
    logger=resultsLogger(False) # init all the logging/counting
    logger.startCounting() # for timing the process, start now

    logger.setNumItems(len(files_dict),print_out=False)

    tfidfmodels={}

    # if we're running over the full corpus, we should only load the indices once
    # at the beginning, as they're going to be the same
    if full_corpus:
        for model in files_dict["ALL_FILES"]["tfidf_models"]:
            # create a Lucene search instance for each method
            tfidfmodels[model["method"]]=luceneRetrievalBoost(model["actual_dir"],model["method"],logger=None)

    previous_guid=""
    logger.total_citations=len(precomputed_queries)

    #===================================
    # MAIN LOOP over all testing files
    #===================================
    for query in precomputed_queries:
        # for every generated query for this context
        guid=query["file_guid"]

        # if this is not a full_corpus run and we've moved on to the next test file
        # then we should load the indices for this new file
        if not full_corpus and guid != previous_guid:
##            logger.showProgressReport(guid) # prints out info on how it's going
            previous_guid=guid
            for model in files_dict[guid]["tfidf_models"]:
                # create a Lucene search instance for each method
                tfidfmodels[model["method"]]=luceneRetrievalBoost(model["actual_dir"],model["method"],logger=None)

        # for every method used for extracting BOWs
        for doc_method in all_doc_methods:
            # ACTUAL RETRIEVAL HAPPENING - run query
            retrieved=tfidfmodels[doc_method].runQuery(query["query_text"],all_doc_methods[doc_method]["runtime_parameters"], guid)

            result_dict={"file_guid":guid,
            "citation_id":query["citation_id"],
            "doc_position":query["doc_position"],
            "query_method":query["query_method"],
            "doc_method":doc_method ,
            "az":query["az"],
            "cfc":query["cfc"],
            "match_guid":query["match_guid"]}

            if not retrieved:    # the query was empty or something
                score=0
                precision_score=0
##                        print "Error: ", doc_method , qmethod,tfidfmodels[method].indexDir
##                        logger.addResolutionResult(guid,m,doc_position,qmethod,doc_method ,0,0,0)
                result_dict["mrr_score"]=0
                result_dict["precision_score"]=0
                result_dict["ndcg_score"]=0
                result_dict["rank"]=0
                result_dict["first_result"]=""

                logger.addResolutionResultDict(result_dict)
            else:
                result=logger.measureScoreAndLog(retrieved, query["citation_multi"], result_dict)


    logger.computeAverageScores()
    results=[]
    for query_method in logger.averages:
        for doc_method in logger.averages[query_method]:
            weights=all_doc_methods[doc_method]["runtime_parameters"]
            data_line={"query_method":query_method,"doc_method":doc_method,"citation_az":citation_az}

            for metric in logger.averages[query_method][doc_method]:
                data_line["avg_"+metric]=logger.averages[query_method][doc_method][metric]
            data_line["precision_total"]=logger.scores["precision"][query_method][doc_method]

            signature=""
            for w in weights:
                data_line[w]=weights[w]
                signature+=str(w)

            data_line["weight_signature"]=signature
            results.append(data_line)

##    logger.writeDataToCSV(Corpus.output_dir+"testing_test_precision.csv")

    return results



def statsOnResults(data, metric="avg_mrr"):
    """
        Returns the mean of the top results that have the same number
    """
    res={}
    if len(data)==0:
        return
    val_to_match=data[metric].iloc[0]
    index=0
    while data[metric].iloc[index]==val_to_match:
        index+=1
    lines=data.iloc[:index]
    means=lines.mean()
    print "Averages:"
    for zone in AZ_ZONES_LIST:
        res[zone]=means[zone]
        print zone,":",means[zone]
    return res


def autoTrainWeightValues(zones_to_process=AZ_ZONES_LIST, files_dict_filename="files_dict.json", testing_methods=None, filename_add=""):
    """
        Tries different values for
    """
    lucene.initVM(maxheap="768m") # init Lucene VM

    files_dict=json.load(open(Corpus.prebuiltBOWs_dir+files_dict_filename,"r"))
    queries_by_az=json.load(open(Corpus.prebuiltBOWs_dir+"queries_by_az.json","r"))
    queries_by_cfc=json.load(open(Corpus.prebuiltBOWs_dir+"queries_by_cfc.json","r"))

    print "AZs:"
    for key in queries_by_az:
        print key, len(queries_by_az[key])

    print ""
    print "CFCs:"
    for key in queries_by_cfc:
        print key, len(queries_by_cfc[key])
    print ""

    results=[]

    if testing_methods:
        all_doc_methods=context_extract.getDictOfTestingMethods(testing_methods)
    else:
        all_doc_methods=files_dict["ALL_FILES"]["doc_methods"] # convert nested dict to flat

    annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"]=="annotated_boost"]

    counter=weightCounterList([1,3,5])
    print counter.getPossibleValues()
    # this is to run it "in polynomial time"
    # most citations are either OTH (1307) or OWN (3876)
    MAX_QUERIES_EVALUATED=650

    print "Processing zones ",zones_to_process

    for az_type in zones_to_process:
        print "Number of queries in ",az_type,"zones:",len(queries_by_az[az_type])
        if len(queries_by_az[az_type]) > MAX_QUERIES_EVALUATED:
            print "Evaluating a maximum of", MAX_QUERIES_EVALUATED
        for method in annotated_boost_methods:

            progress=progressIndicator(True) # create a progress indicator, keeps track of what's been done and how long is left
            counter.initWeights(all_doc_methods[method]["runtime_parameters"])
            all_doc_methods[method]["runtime_parameters"]=counter.weights

            # total number of combinations that'll be processed
            progress.setNumItems(counter.numCombinations())

            while not counter.allCombinationsProcessed():#  and progress.item_counter < 2:

                weight_parameter={method:{"runtime_parameters":counter.weights}}
##                print counters
                # !TODO set :MAX_QUERIES_EVALUATED and MAX_QUERIES_EVALUATED:MAX_QUERIES_EVALUATED*2
##                citations_set=queries_by_az[az_type][MAX_QUERIES_EVALUATED:MAX_QUERIES_EVALUATED*2]
                citations_set=queries_by_az[az_type][:MAX_QUERIES_EVALUATED]
                scores=measureCitationResolution(files_dict,citations_set, weight_parameter, az_type, all_doc_methods)
                results.extend(scores)
                progress.showProgressReport("",1)
                counter.nextCombination()

        # prints results per each AZ/CFC zone
        data=DataFrame(results)
        metric="avg_mrr"
        data=data.sort(metric, ascending=False)
        filename=getSafeFilename(Corpus.output_dir+"weights_"+az_type+"_"+str(counter.getPossibleValues())+filename_add+".csv")
        data.to_csv(filename)
##        statsOnResults(data, metric)

##        print data.to_string()

def autoTrainWeightValues_optimized(zones_to_process=AZ_ZONES_LIST, retrieval_results_filename="precomputed_retrieval_results.json",
testing_methods=None, filename_add="", files_dict_filename="files_dict.json",split_set=None):
    """
        Tries different values for
    """

    files_dict=json.load(open(Corpus.prebuiltBOWs_dir+files_dict_filename,"r"))
##    retrieval_results=json.load(open(Corpus.prebuiltBOWs_dir+retrieval_results_filename,"r"))
    retrieval_results_by_az=json.load(open(Corpus.prebuiltBOWs_dir+"retrieval_results_by_az.json","r"))

    results=[]

    if testing_methods:
        all_doc_methods=context_extract.getDictOfTestingMethods(testing_methods)
    else:
        all_doc_methods=files_dict["ALL_FILES"]["doc_methods"] # convert nested dict to flat

    annotated_boost_methods=[x for x in all_doc_methods if all_doc_methods[x]["type"]=="annotated_boost"]

    counter=weightCounterList([1,3,5,7])
    print counter.getPossibleValues()

    print "Processing zones ",zones_to_process

    for az_type in zones_to_process:
        print "Number of precomputed results in ",az_type,"zones:",len(retrieval_results_by_az[az_type])
        for method in annotated_boost_methods:
            counter.initWeights(all_doc_methods[method]["runtime_parameters"])

            # !TODO set :MAX_QUERIES_EVALUATED and MAX_QUERIES_EVALUATED:MAX_QUERIES_EVALUATED*2
    ##                citations_set=queries_by_az[az_type][MAX_QUERIES_EVALUATED:MAX_QUERIES_EVALUATED*2]
            progress=progressIndicator(True) # create a progress indicator, keeps track of what's been done and how long is left
            all_doc_methods[method]["runtime_parameters"]=counter.weights

            # total number of combinations that'll be processed
            progress.setNumItems(counter.numCombinations(),dot_every_xitems=15)

            print "Testing weight value combinations with precomputed formula"
            while not counter.allCombinationsProcessed():#  and progress.item_counter < 2:
                if split_set==1:
                    test_set=retrieval_results_by_az[az_type][:1000]
                elif split_set==2:
                    test_set=retrieval_results_by_az[az_type][1000:2000]
                elif split_set==3:
                    test_set=retrieval_results_by_az[az_type][1000:2000]
                elif not split_set:
                    test_set=retrieval_results_by_az[az_type][2000:]
                else:
                    assert(test_set in [None,1,2,3])

                scores=measurePrecomputedResolution(test_set,method,counter.weights, all_doc_methods, az_type)
##                scores=measurePrecomputedResolution(retrieval_results_by_az[az_type][1000:2000],method,counter.weights, all_doc_methods, az_type)
                results.extend(scores)
                progress.showProgressReport("",1)
                counter.nextCombination()

        # prints results per each AZ/CFC zone
        data=DataFrame(results)
        metric="avg_mrr"
        data=data.sort(metric, ascending=False)
        filename=getSafeFilename(Corpus.output_dir+"weights_"+az_type+"_"+str(counter.getPossibleValues())+filename_add+".csv")
        data.to_csv(filename)
##        statsOnResults(data, metric)
##        print data.to_string()

def testPrebuilt():
    bla=loadPrebuiltBOW("c92-2117.xml","inlink_context",50)
    print bla


def manualJSONdump(list_obj,file):
    """
        Trying to fix whatever is wrong with json.dump()
    """
    file.write("[")

    for index,item in enumerate(list_obj):
        file.write(json.dumps(item))
        if index < len(list_obj)-1:
            file.write(",")
    file.write("]")

def fixDumping(retrieval_results_filename="precomputed_retrieval_results.json"):
    import cPickle
    results=cPickle.load(open(Corpus.prebuiltBOWs_dir+"test.pic","r"))
    json.dump(results,open(Corpus.prebuiltBOWs_dir+retrieval_results_filename,"w"))
    createExplainQueriesByAZ()

def main():
##    buildCorpusIDFdict(r"C:\NLP\PhD\bob\bob\files\*.xml", r"C:\NLP\PhD\bob\bob\idfdict.pic", 10000)
    print "Running testing pipeline"

##    import cProfile
##    cProfile.run('blah()',None,"calls")
##    cProfile.run('testRunTest1(1, 20, "results6.csv")',None,"time")


def compareRetrievalAndPrecomputed():
    """
        This is a test function. Need to figure out what is going on that
        makes retrieval all weird.
    """
    pass


def testWeightCounter():
    counter=weightCounterList([1,3,5])
    counter.initWeights(["a","b","c"])
    while not counter.allCombinationsProcessed():
        print counter.weights
        counter.nextCombination()

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
##        {"ALL":{"AIM":"1","BAS":"1","BKG":"1","CTR":"1","OTH":"1","OWN":"1","TXT":"1","inlink_context":1},
##         "OTH":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"1","OWN":"0","TXT":"0","inlink_context":1},
##         "OWN":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"0","OWN":"1","TXT":"0","inlink_context":1},
##        }},

    # this is sentence-based ILC, annotated with AZ and CSC
##    "ilc_AZ":{"type":"annotated_boost", "index":"ilc_AZ", "parameters":["paragraph","1up_1down","1up","1only"], "runtime_parameters":
##        {"ALL":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT"]}
##        },

    "ilc_AZ":{"type":"annotated_boost", "index":"ilc_AZ", "parameters":["paragraph"], "runtime_parameters":
        {"ALL":["ilc_AZ_AIM","ilc_AZ_BAS","ilc_AZ_BKG","ilc_AZ_CTR","ilc_AZ_OTH","ilc_AZ_OWN","ilc_AZ_TXT"]}
        },

    # this is sen
##    "ilc_az_ilc_az":{"type":"ilc_annotated_boost", "index":"ilc_AZ", "parameters":[1],
##        "ilc_parameters":["1only","1up","1up1down","paragraph"],
##        "runtime_parameters":
##        {"ALL":["ilc_AZ_AIM":"1","ilc_AZ_BAS":"1","ilc_AZ_BKG":"1","ilc_AZ_CTR":"1","ilc_AZ_OTH":"1","ilc_AZ_OWN":"1","ilc_AZ_TXT":"1","ilc_AZ_AIM":1],
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
                "function":context_extract.getOutlinkContextWindowAroundCitationMulti,
                "type":"window"},

##            "sentence":{"parameters":[
##                "1only",
##                "paragraph",
##                "1up",
##                "1up_1down"
##                ],
##                "function":context_extract.getOutlinkContextSentences,
##                "type":"sentence"}
                }


    # 1. automatically get list of papers to test on with SQL query
    Corpus.TEST_FILES=Corpus.listPapers("num_in_collection_references >= 8")

    # 2a. uncomment this to generate the queries and files_dict for Citation Resolution
##    precomputeQueries(Corpus.TEST_FILES,20,testing_methods, qmethods)

    # (x) this is just testing for year in ILC - tested, little difference
##    precomputeQueries(Corpus.TEST_FILES,20,testing_methods, qmethods, files_dict_filename="ilc_year_test_file_dict.json",
##    filename="ilc_year_precomputed_queries.json")
##    precomputeQueries(Corpus.TEST_FILES,20,testing_methods, qmethods, files_dict_filename="sentence_test_file_dict.json",
##    filename="sentence_precomputed_queries.json")

    # 2b. uncomment this to generate the queries and files_dict for Full Corpus
##    precomputeQueries(Corpus.TEST_FILES,20,testing_methods,qmethods, files_dict_filename="files_dict_full_corpus.json", full_corpus=True)

    # (x) this is the old Citation Resolution run, just for testing now
##    runPrecomputedCitationResolutionLucene("results_full_corpus_test1.csv", files_dict_filename="files_dict_full_corpus.json",
##    full_corpus=True, testing_methods=testing_methods)

##    runPrecomputedCitationResolutionLucene("overlap_bulkScorer_explain.csv", files_dict_filename="files_dict.json", full_corpus=False)

##    runPrecomputedCitationResolutionLucene("results_sentence_test2.csv", files_dict_filename="sentence_test_file_dict.json", full_corpus=False, testing_methods=testing_methods, precomputed_queries_filename="sentence_precomputed_queries.json")

    # 3. uncomment this to precompute the Lucene explanations
    precomputeExplainQueries(testing_methods,use_default_similarity=True)
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
