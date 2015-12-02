#-------------------------------------------------------------------------------
# Name:        exp
# Purpose:      experiments
#
# Author:      Daniel Duma
#
# Created:     08/12/2013
# Copyright:   (c) Daniel Duma 2013
#-------------------------------------------------------------------------------

import corpora

import context_extract

import lucene
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.document import Field
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.queryparser.classic import FieldAgnosticQueryParser
from org.apache.lucene.search.similarities import DefaultSimilarity, FieldAgnosticSimilarity
from org.apache.lucene.util import Version as LuceneVersion
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import TopScoreDocCollector;
from java.io import File

##from lucene import \
##    SimpleFSDirectory, System, File, \
##    Document, Field, StandardAnalyzer, IndexWriter, IndexSearcher, QueryParser

from pandas import *

import glob, math, os, datetime, re, copy, sys
from collections import defaultdict, namedtuple, OrderedDict
from util.results_logging import progressIndicator,resultsLogger
##import heapq
from az_cfc_classification import AZ_ZONES_LIST

# this is the corpus instance used for all experiments
print "Loading ACL bob corpus..."
drive="C"
corpora.Corpus.connectCorpus(drive+r":\NLP\PhD\bob")

SPECIAL_FIELDS_FOR_TESTS=["inlink_context"]
MAX_RESULTS_RECALL=200

def formatReferenceAPA(reference):
    """
        Formats a reference/metadata in plain text APA
    """
    return ref.get("authors","")+". "+ref.get("year","")+". "+ref.get("title","")+ ". "+ref.get("publication","")

class luceneRetrieval:
    """
        Encapsulates the Lucene retrieval engine
    """
    def __init__(self, indexDir, method, logger=None):
        self.indexDir=indexDir
        directory = SimpleFSDirectory(File(self.indexDir))
        self.analyzer = StandardAnalyzer(LuceneVersion.LUCENE_CURRENT)
        self.reader=DirectoryReader.open(directory)
        self.searcher = IndexSearcher(self.reader)

        # uncomment one of these lines to change the type of parser, query and weight used
        self.query_parser=FieldAgnosticQueryParser
##        self.query_parser=QueryParser

        similarity=FieldAgnosticSimilarity()
        # by default, FieldAgnosticSimilarity uses coord factor, can be disabled
##        similarity.useCoord=False

        self.useExplainQuery=False

        self.searcher.setSimilarity(similarity)
        self.max_results=MAX_RESULTS_RECALL
        self.method=method # never used?
        self.logger=logger

    def cleanupQuery(self,query):
        """
            Remove symbols from the query that can lead to errors when parsing the query
        """
        rep_list=["~", "and", "^","\"","+","-","(",")", "{","}","[","]","?",":","*"]
        rep_list.extend(context_extract.punctuation)
        query=query.lower()
        for r in rep_list:
            query=query.replace(r," ")
        query=re.sub(r"\s+"," ",query)
        query=query.strip()
        return query

    def runQueryViaExplain(self,query, max_results):
        """
            Really crappy solution to make sure that explanations and searches are the same
            while I fix Lucene
        """
        results=[]

        index=0
        for index in range(self.reader.numDocs()):
            explanation=self.searcher.explain(query,index)
            score=explanation.getValue()
##            match=re.search(r"(.*?)\s=",explanation.toString(),re.IGNORECASE|re.DOTALL)
##            if match:
##                score=float(match.group(1))
            hit=namedtuple("Hit",["doc","score"])
            hit.doc=index
            hit.score=score
##            heapq.heappush(results,hit)
            results.append(hit)

        results.sort(key=lambda x:x.score,reverse=True)

        if max_results < self.reader.numDocs():
            results=results[:max_results]

        return results

    def runQuery(self, query):
        """
            LOTS OF SWEET LUCENE
        """
        original_query=query
        query=self.cleanupQuery(query)
        if query=="":
            return None

        self.last_query=query

        try:
            query = self.query_parser(lucene.Version.LUCENE_CURRENT, "text", self.analyzer).parse(query)
        except:
            print "Lucene exception:",sys.exc_info()[:2]
            return None

        if self.useExplainQuery:
            # this should only exist until I fix the lucene bulkScorer to give the same results
            hits=self.runQueryViaExplain(query,self.max_results)
        else:
            collector=TopScoreDocCollector.create(MAX_RESULTS_RECALL, True)
            self.searcher.search(query, collector)
            hits = collector.topDocs().scoreDocs

##        print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)
        res=[]


##        if len(hits.scoreDocs) ==0:
##            print "Original query:",original_query
##            print "Query:", query

        for hit in hits:
            doc = self.searcher.doc(hit.doc)
            metadata= json.loads(doc.get("metadata"))
            res.append((hit.score,metadata))
        return res


class luceneRetrievalBoost(luceneRetrieval):
    """
        Use Lucene for retrieval boosting different AZ fields differently
    """

    def __init__(self, indexDir, method, logger=None):
        luceneRetrieval.__init__(self, indexDir, method, logger)

    def generateLuceneQuery(self, query, parameters, test_guid=None):
        """
            Processes input, expands query to inlcude fields and weights
        """
        original_query=query
        query=self.cleanupQuery(query)
        if query=="":
            return None

        self.last_query=query

        query_text=""
        query_words=query.split()

        # this adds to the query a field specially built for this doc
        # in case a doc in the index doesn't have inlink_context but has
        # special inlink_context_special_GUID fields
        if test_guid:
            for param in [p for p in parameters if p in SPECIAL_FIELDS_FOR_TESTS]:
                parameters[context_extract.getFieldSpecialTestName(param, test_guid)]=parameters[param]

        for index,param in enumerate(parameters):
            for qindex,word in enumerate(query_words):

                query_text+=param+":\""+word+"\"^"+str(parameters[param])+" "
                if qindex < len(query_words)-1:
                    query_text+=" OR "

            if index < len(parameters)-1:
                query_text+=" OR "

        return query_text

    def runQuery(self, query, parameters, test_guid):
        """
            Run the query, return a list of tuples (score,metadata) of top docs
        """
        query_text=self.generateLuceneQuery(query,parameters, test_guid)
        try:
            query = self.query_parser(LuceneVersion.LUCENE_CURRENT, "text", self.analyzer).parse(query_text)
        except:
            print "Lucene exception:",sys.exc_info()[:2]
            return None

        if self.useExplainQuery:
            # this should only exist until I fix the lucene bulkScorer to give the same results
            hits=self.runQueryViaExplain(query,self.max_results)
        else:
            collector=TopScoreDocCollector.create(MAX_RESULTS_RECALL, True)
            self.searcher.search(query, collector)
            hits = collector.topDocs().scoreDocs
        res=[]

        # explain the query
        if self.logger:
            self.logger.logReport(query_text+"\n")

            if self.logger.full_citation_id in self.logger.citations_extra_info:
                max_explanations=len(hits)
            else:
                max_explanations=1

            for index in range(max_explanations):
                self.logger.logReport(self.searcher.explain(query,index))

        for hit in hits:
            doc = self.searcher.doc(hit.doc)
            metadata= json.loads(doc.get("metadata"))
            res.append((hit.score,metadata))

        if self.logger and self.logger.full_citation_id in self.logger.citations_extra_info:
            print query_text
            print hits
            print res
            print

##        del hits_list
        del hits
        del query
        del query_text

        return res


class weightCounter:
    """
        Deals with iterating over combinations of weights for training
    """
    def __init__(self, INITIAL_VALUE=1, UNIT_MULTI=1, MAX_COUNTER=2):
        """
            INITIAL_VALUE = weight minimum
            UNIT_MULTI = max that will be added to the minimum
            MAX_COUNTER = how many steps it will be added over
        """
        self.COUNTER_INCREMENT=1
        self.MAX_COUNTER=MAX_COUNTER       # num of combinations (+1 real ones, 0 is one)
        self.UNIT_MULTI=UNIT_MULTI        # this is as much as will be added to the multiplier in total over all counter combinations
        self.WEIGHT_MULTIPLIER=self.UNIT_MULTI/float(self.MAX_COUNTER)  # how much to add to multiplier per cycle
        self.INITIAL_VALUE=INITIAL_VALUE
        self.counters=[]

    def numCombinations(self):
        """
            Returns the number of total possible combinations of this counter
        """
        return pow((1+self.MAX_COUNTER/self.COUNTER_INCREMENT),len(self.counters))

    def initWeights(self,parameters):
        """
            Creates counters and weights from input list of parameters to change
        """
        # create a dict where every field gets a weight of x
        self.weights={x:self.INITIAL_VALUE for x in parameters}
        # make a counter for each field
        self.counters=[0 for x in range(len(self.weights))]

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
        self.counters[0]+=self.COUNTER_INCREMENT

        for index in range(len(self.counters)):
            if self.counters[index] > self.MAX_COUNTER:
                self.counters[index+1] += self.COUNTER_INCREMENT
                self.counters[index]=0
            self.weights[self.weights.keys()[index]]=self.INITIAL_VALUE+(self.WEIGHT_MULTIPLIER*self.counters[index])

    def getPossibleValues(self):
        values=[]
        for cnt in range(self.MAX_COUNTER+1):
            value="%2.2f"%(self.INITIAL_VALUE+(self.WEIGHT_MULTIPLIER*cnt))
            value=value.replace(".00","")
            values.append(value)
        return values

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
            actual_dir=corpora.Corpus.getLuceneIndexPath(guid,all_doc_methods[method]["index_filename"],full_corpus)
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
    logger.output_filename=corpora.Corpus.output_dir+output_filename

    precomputed_queries=json.load(open(corpora.Corpus.prebuiltBOWs_dir+precomputed_queries_filename,"r"))
    files_dict=json.load(open(corpora.Corpus.prebuiltBOWs_dir+files_dict_filename,"r"))

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

    corpora.Corpus.loadAnnotators()

    all_doc_methods=context_extract.getDictOfTestingMethods(bowmethods) # convert nested dict to flat dict where each method includes its parameters in the name
    tfidfmodels={} # same as above

    precomputed_queries=[]

    files_dict=OrderedDict()
    files_dict["ALL_FILES"]={}
    files_dict["ALL_FILES"]["doc_methods"]=all_doc_methods
    if full_corpus:
        files_dict["ALL_FILES"]["tfidf_models"]=[]
        for method in all_doc_methods:
            actual_dir=corpora.Corpus.getLuceneIndexPath("ALL_FILES",all_doc_methods[method]["index_filename"],full_corpus)
            files_dict["ALL_FILES"]["tfidf_models"].append({"method":method,"actual_dir":actual_dir})

    #===================================
    # MAIN LOOP over all testing files
    #===================================
    for guid in FILE_LIST:
        logger.showProgressReport(guid) # prints out info on how it's going

        doc=corpora.Corpus.loadSciDoc(guid) # load the SciDoc JSON from the corpus
        if not doc:
            print "ERROR: Couldn't load pickled doc:",guid
            continue
        doctext=doc.getFullDocumentText() #  store a plain text representation

        # load the citations in the document that are resolvable, or generate if necessary
        tin_can=corpora.Corpus.getResolvableCitationsCache(guid, doc)
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
                actual_dir=corpora.Corpus.getLuceneIndexPath(guid,all_doc_methods[method]["index_filename"],full_corpus)
                precomputed_file["tfidf_models"].append({"method":method,"actual_dir":actual_dir})

        files_dict[guid]=precomputed_file

        corpora.Corpus.annotators["AZ"].annotateDoc(doc)
        corpora.Corpus.annotators["CFC"].annotateDoc(doc)

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
                    assert(False,"window method type not implemented")

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

    json.dump(precomputed_queries,open(corpora.Corpus.prebuiltBOWs_dir+filename,"w"))
    queries_by_az={zone:[] for zone in AZ_ZONES_LIST}
    queries_by_cfc=defaultdict(lambda:[])

    for precomputed_query in precomputed_queries:
        queries_by_az[precomputed_query["az"]].append(precomputed_query)
        queries_by_cfc[precomputed_query["cfc"]].append(precomputed_query)

    json.dump(files_dict,open(corpora.Corpus.prebuiltBOWs_dir+files_dict_filename,"w"))
    json.dump(queries_by_az,open(corpora.Corpus.prebuiltBOWs_dir+"queries_by_az.json","w"))
    json.dump(queries_by_cfc,open(corpora.Corpus.prebuiltBOWs_dir+"queries_by_cfc.json","w"))
    print "Precomputed citations saved."
##    logfile=open("results.txt","wb")
##    logfile.writelines(results)

##    writeTuplesToCSV("Filename citation_id doc_section query_method doc_method MRR_score NDCG_score Precision_score Correct_file".split(),overall_results,corpora.Corpus.output_dir+output_filename)


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

##    logger.writeDataToCSV(corpora.Corpus.output_dir+"testing_test_precision.csv")

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

    files_dict=json.load(open(corpora.Corpus.prebuiltBOWs_dir+files_dict_filename,"r"))
    queries_by_az=json.load(open(corpora.Corpus.prebuiltBOWs_dir+"queries_by_az.json","r"))
    queries_by_cfc=json.load(open(corpora.Corpus.prebuiltBOWs_dir+"queries_by_cfc.json","r"))

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

    counter=weightCounter(1,4,2)
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
        filename=getSafeFilename(corpora.Corpus.output_dir+"weights_"+az_type+"_"+str(counter.getPossibleValues())+filename_add+".csv")
        data.to_csv(filename)
##        statsOnResults(data, metric)

##        print data.to_string()

def testPrebuilt():
    bla=loadPrebuiltBOW("c92-2117.xml","inlink_context",50)
    print bla


def main():
##    buildCorpusIDFdict(r"C:\NLP\PhD\bob\bob\files\*.xml", r"C:\NLP\PhD\bob\bob\idfdict.pic", 10000)
    print "Running testing pipeline"

##    import cProfile
##    cProfile.run('blah()',None,"calls")
##    cProfile.run('testRunTest1(1, 20, "results6.csv")',None,"time")



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

    "inlink_context":{"type":"standard_multi", "index":"inlink_context",
        "parameters": [10, 20, 30], "runtime_parameters":{"inlink_context":"1"}},

##    "inlink_context_year":{"type":"standard_multi", "index":"inlink_context_year",
##        "parameters": [10, 20, 30], "runtime_parameters":{"inlink_context":"1"}},

##    "ilc_passage":{"type":"ilc_mashup",  "index":"ilc_passage", "mashup_method":"passage","ilc_parameters":[10, 20, 30, 40, 50],
##        "parameters":[250,350], "runtime_parameters":{"text":"1","inlink_context":"1"}},

##    "az_annotated":{"type":"annotated_boost", "index":"az_annotated_1", "parameters":[1], "runtime_parameters":
##        {"ALL":["AIM","BAS","BKG","CTR","OTH","OWN","TXT"]
##         "OTH_only":["OTH"],
##         "OWN_only":["OWN"],
##        }},

##    "section":{"type":"annotated_boost", "index":"section_annotated_1", "parameters":[1], "runtime_parameters":
##        {
####        "title_abstract":{"title":"1","abstract":"1"},
##         "full_text":["title","abstract","text"],
##        }},

    "ilc":{"type":"ilc_annotated_boost", "index":"ilc_section_annotated", "ilc_parameters":[10, 20, 30, 40, 50], "parameters":[1], "runtime_parameters":
        {
##         "title_abstract":["title","abstract","inlink_context"],
         "full_text":["title", "abstract","text","inlink_context"],
        }},

##    "ilc_az_annotated":{"type":"ilc_annotated_boost", "index":"ilc_az_annotated", "parameters":[1], "ilc_parameters":[10, 20, 30, 40, 50], "runtime_parameters":
##        {"ALL":{"AIM":"1","BAS":"1","BKG":"1","CTR":"1","OTH":"1","OWN":"1","TXT":"1","inlink_context":1},
##         "OTH":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"1","OWN":"0","TXT":"0","inlink_context":1},
##         "OWN":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"0","OWN":"1","TXT":"0","inlink_context":1},
##        }},

##    "ilc_az_ilc_az":{"type":"ilc_annotated_boost", "index":"ilc_AZ", "parameters":[1],
##        "ilc_parameters":["AIM","BAS","BKG","CTR","OTH","OWN","TXT"],
##        "runtime_parameters":
##        {"ALL":{"AIM":"1","BAS":"1","BKG":"1","CTR":"1","OTH":"1","OWN":"1","TXT":"1","inlink_context":1},
##         "OTH":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"1","OWN":"0","TXT":"0","inlink_context":1},
##         "OWN":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"0","OWN":"1","TXT":"0","inlink_context":1},
##        }},

    }

    # this is the dict of query extraction methods
    qmethods={"window":{"parameters":[
##                (3,3),
##                (5,5),
                (10,10),
                (5,10),
                (10,5),
                (20,20),
                (20,10),
                (10,20),
                (30,30)
                ],
                "function":context_extract.getOutlinkContextWindowAroundCitationMulti,
                "type":"window"},
            "sentence":{"parameters":[
                "1only",
                "paragraph",
                "1up",
                "1up_1down"
                ],
                "function":context_extract.getOutlinkContextSentences,
                "type":"sentence"}
                }

    # automatically get list of papers to test on with SQL query
    corpora.Corpus.TEST_FILES=corpora.Corpus.listPapers("num_in_collection_references >= 8")
    # uncomment this to generate the queries and files_dict for Citation Resolution
##    precomputeQueries(corpora.Corpus.TEST_FILES,20,testing_methods, qmethods)
##    precomputeQueries(corpora.Corpus.TEST_FILES,20,testing_methods, qmethods, files_dict_filename="ilc_year_test_file_dict.json",
##    filename="ilc_year_precomputed_queries.json")
##    precomputeQueries(corpora.Corpus.TEST_FILES,20,testing_methods, qmethods, files_dict_filename="sentence_test_file_dict.json",
##    filename="sentence_precomputed_queries.json")
    # uncomment this to generate the queries and files_dict for Full Corpus
##    precomputeQueries(corpora.Corpus.TEST_FILES,20,testing_methods,qmethods, files_dict_filename="files_dict_full_corpus.json", full_corpus=True)

    # this is the old Citation Resolution run, just for testing now
##    runPrecomputedCitationResolutionLucene("results_full_corpus_test1.csv", files_dict_filename="files_dict_full_corpus.json",
##    full_corpus=True, testing_methods=testing_methods)

##    runPrecomputedCitationResolutionLucene("overlap_bulkScorer_explain.csv", files_dict_filename="files_dict.json", full_corpus=False)

##    runPrecomputedCitationResolutionLucene("results_sentence_test2.csv", files_dict_filename="sentence_test_file_dict.json", full_corpus=False, testing_methods=testing_methods, precomputed_queries_filename="sentence_precomputed_queries.json")

    # each parameter is a zone to process
    zones_to_process=["OWN"]
    if len(sys.argv) > 1:
        zones_to_process=sys.argv[1:]

    # uncomment to run all combinations
##    autoTrainWeightValues(zones_to_process, filename_add="_FA_BulkScorer_first650")
    autoTrainWeightValues(zones_to_process, filename_add="_FA_BulkScorer_second650")
    pass

if __name__ == '__main__':
    main()
