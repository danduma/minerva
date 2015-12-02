#-------------------------------------------------------------------------------
# Name:        exp
# Purpose:      experiments
#
# Author:      Daniel Duma
#
# Created:     08/12/2013
# Copyright:   (c) Daniel Duma 2013
#-------------------------------------------------------------------------------

from ACLbobCorpus import *
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
from org.apache.lucene.search.similarities import DefaultSimilarity, MySimilarity, BM25Similarity
from org.apache.lucene.util import Version as LuceneVersion
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from java.io import File

##from lucene import \
##    SimpleFSDirectory, System, File, \
##    Document, Field, StandardAnalyzer, IndexWriter, IndexSearcher, QueryParser

import glob, math, os, datetime, re
from collections import defaultdict, namedtuple
import heapq
from util.results_logging import resultsLogger

def cmp_gt(x, y):
    return y < x if hasattr(y, '__lt__') else not (x <= y)
heapq.cmp_lt = cmp_gt

from az_cfc_classification import AZannotator, CFCannotator

# this is the corpus instance used for all experiments
print "Loading ACL bob corpus..."
drive="C"
corpora.Corpus=ACLbobCorpusClass(drive+r":\NLP\PhD\bob")


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
##        similarity=MySimilarity()
        similarity=BM25Similarity()

        self.searcher.setSimilarity(similarity)
        if logger:
            logger.run_parameters["similarity"]=type(similarity).__name__

        self.max_results=100
        self.method=method # never used?
        self.logger=logger

    def cleanupQuery(self,query):
        """
            Removes things from the query that lead to errors when parsing the query
        """
        rep_list=["~", "and", "^","\"","+","-","(",")", "{","}","[","]","?",":"]
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
            query = FieldAgnosticQueryParser(lucene.Version.LUCENE_CURRENT, "text", self.analyzer).parse(query)
        except:
            print "Lucene exception:",sys.exc_info()[:2]
            return None

##        hits = self.searcher.search(query, self.max_results)
##        hits=hits.scoreDocs

        hits=self.runQueryViaExplain(query,self.max_results)
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

    def generateLuceneQuery(self,query, parameters):
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
        for index,param in enumerate(parameters):
            for qindex,word in enumerate(query_words):
                query_text+=param+":"+word+"^"+str(parameters[param])+" "
                if qindex < len(query_words)-1:
                    query_text+=" OR "

            if index < len(parameters)-1:
                query_text+=" OR "

        return query_text

    def runQuery(self, query, parameters):
        """
            LOTS OF SWEET LUCENE
        """
        query_text=self.generateLuceneQuery(query,parameters)
        if not query_text:
            print "Empty query:", query
            return None

        try:
            query = FieldAgnosticQueryParser(LuceneVersion.LUCENE_CURRENT, "text", self.analyzer).parse(query_text)

        except:
            print "Lucene exception:",sys.exc_info()[:2]
            return None

##        hits = self.searcher.search(query, self.max_results)
##        hits = hits.scoreDocs
        hits=self.runQueryViaExplain(query,self.max_results)
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
        return res


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

def measureAllScores(docs, match):
    """
        Computes all scores and position document was ranked in
    """
    # docs must be already ordered in descending order
    guids=[]
    for doc in docs:
        guid=doc[1]["guid"]
        if guid not in guids:
            guids.append(guid)

    for index, guid in enumerate(guids):
        if guid==match["match_guid"]:
            if match["cit"].has_key("multi") and index+1 <= match["cit"]["multi"]:
                score=1
            else:
                score=1/ math.log(index+2)
            return score
    return 0


def addNewQueryMethod(queries, name, parameters, function, match, doctext):
    """
        Runs a multi query generation function, adds all results with procedural
        identifier to queries dict
    """
    all_queries=function(match, doctext, parameters, maxwords=20, options={"jump_paragraphs":True})

    for cnt, p_size in enumerate(parameters):
        method=name+str(p_size[0])+"_"+str(p_size[1])
        queries[method]=all_queries[cnt]


#===============================================================================
#   TESTING PIPELINE
#===============================================================================

def runCitationResolutionLucene(FILE_LIST, numchunks=10, output_filename="results.csv", bowmethods={}, querymethods={}, logging=True):
    """
        Using Lucene, run Citation Resolution
    """
    logger=resultsLogger(True) # init all the logging/counting
    logger.startCounting() # for timing the process, start now

    lucene.initVM(maxheap="768m") # init Lucene VM

    logger.progressReportSetNumFiles(len(FILE_LIST))
    logger.numchunks=numchunks

    all_bow_methods=context_extract.getDictOfTestingMethods(bowmethods) # convert nested dict to flat
    tfidfmodels={} # same as above

    cfc_annotator=CFCannotator("trained_cfc_classifier.pickle")
    az_annotator=AZannotator("trained_az_classifier.pickle")

    methods_overlap=0
    total_overlap_points=0
    rank_differences=[]
    rank_per_method=defaultdict(lambda:[])
    precision_per_method=defaultdict(lambda:[])

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

        logger.total_citations+=len(resolvable)
        num_in_collection_references=len(in_collection_references)
        print "Resolvable citations:",len(resolvable), "In-collection references:",num_in_collection_references

        for method in all_bow_methods:
            # create a Lucene search instance for each method
            actual_dir=corpora.Corpus.fileLuceneIndex_dir+guid+os.sep+all_bow_methods[method]["index_filename"]
##            print actual_dir
            tfidfmodels[method]=luceneRetrievalBoost(actual_dir,method,logger=logger)

        az_annotator.annotateDoc(doc)
        cfc_annotator.annotateDoc(doc)

        print "Resolving citations..."
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
                addNewQueryMethod(queries,method_name,method["parameters"],method["function"],match,doctext)

            parent_s=doc.element_by_id[m["cit"]["parent_s"]]
            az=parent_s["az"] if parent_s.has_key("az") else ""
            cfc=doc.citation_by_id[m["cit"]["id"]]["cfunc"] if doc.citation_by_id[m["cit"]["id"]].has_key("cfunc") else ""

            # for every generated query for this context
            for qmethod in queries:
                logger.addQMethod(qmethod)

                # for every method used for extracting BOWs
                for method in all_bow_methods.keys():
                    # ACTUAL RETRIEVAL HAPPENING - run query
                    logger.full_citation_id=guid+"-"+m["cit"]["id"]
                    logger.logReport("\n================\nCitation: "+logger.full_citation_id+"\nQuery method:"+qmethod+" \nDoc method: "+method+"\n")
                    logger.logReport(queries[qmethod]["text"]+"\n")
                    if queries[qmethod]["text"] == "":
                        print "Warning: Empty query!",qmethod
                    retrieved=tfidfmodels[method].runQuery(queries[qmethod]["text"],all_bow_methods[method]["runtime_parameters"])

                    result_dict={"file_guid":guid,
                    "citation_id":m["cit"]["id"],
                    "doc_position":doc_position,
                    "query_method":qmethod,
                    "doc_method":method,
                    "az":az,
                    "cfc":cfc,
                    "match_guid":m["match_guid"]}

                    if not retrieved:    # the query was empty or something
                        score=0
                        precision_score=0
##                        print "Error: ", method, qmethod,tfidfmodels[method].indexDir
##                        logger.addResolutionResult(guid,m,doc_position,qmethod,method,0,0,0)
                        result_dict["mrr_score"]=0
                        result_dict["precision_score"]=0
                        result_dict["ndcg_score"]=0
                        result_dict["rank"]=0
                        result_dict["first_result"]=""

                        logger.addResolutionResultDict(result_dict)

                    else:
##                        logger.measureScoreAndLog(guid,retrieved,m,doc_position,qmethod,method, queries[qmethod]["text"], az, cfc)
                        result=logger.measureScoreAndLog(retrieved, m, result_dict)
                        rank_per_method[result["doc_method"]].append(result["rank"])
                        precision_per_method[result["doc_method"]].append(result["precision_score"])

                    # DO SOMETHING TO LOG TEXT AND LIST OF REFERENCES ETC

##                    pre_selection_text=doctext[queries[qmethod]["left_start"]-300:queries[qmethod]["left_start"]]
##                    draft_text=doctext[queries[qmethod]["left_start"]:queries[qmethod]["right_end"]]
##                    post_selection_text=doctext[queries[qmethod]["right_end"]:queries[qmethod]["left_start"]+300]
##                    draft_text=u"<span class=document_text>{}</span> <span class=selected_text>{}</span> <span class=document_text>{}</span>".format(pre_selection_text, draft_text, post_selection_text)
##                    print draft_text

                # extra method = random chance control
                logger.addResolutionResultDict({"file_guid":guid,
                        "citation_id":m["cit"]["id"],
                        "doc_position":doc_position,
                        "query_method":qmethod,
                        "doc_method":"RANDOM",
                        "mrr_score":analyticalRandomChanceMRR(num_in_collection_references),
                        "precision_score":1/float(num_in_collection_references),
                        "ndcg_score":0,
                        "rank":0,
                        "az":az,
                        "cfc":cfc,
                        "match_guid":m["match_guid"],
                        "first_result":""
                        })

                rank_diff=abs(rank_per_method["section_1_full_text"][-1]-rank_per_method["full_text_1"][-1])
##                if method_overlap_temp["section_1_full_text"] == method_overlap_temp["full_text_1"]
                if rank_diff == 0:
                    methods_overlap+=1
                rank_differences.append(rank_diff)
                total_overlap_points+=1


    methods=all_bow_methods.keys()
    methods.append("RANDOM")

    logger.writeDataToCSV(corpora.Corpus.output_dir+output_filename)

    logger.showFinalSummary()
    print "Total rank overlap between [section_1_full_text] and [full_text_1] = ",methods_overlap,"/",total_overlap_points," = %02.4f" % (methods_overlap / float(total_overlap_points))
    print "Avg rank difference between [section_1_full_text] and [full_text_1] = %02.4f" % (sum(rank_differences) / float(total_overlap_points))
    print "Avg rank:"
    for method in rank_per_method:
        print method,"= %02.4f" % (sum(rank_per_method[method])/float(len(rank_per_method[method])))
##    print "Avg precision:"
##    for method in precision_per_method:
##        print method,"=",sum(precision_per_method[method])/float(len(precision_per_method[method]))
##    logfile=open("results.txt","wb")
##    logfile.writelines(results)

##    writeTuplesToCSV("Filename citation_id doc_section query_method doc_method MRR_score NDCG_score Precision_score Correct_file".split(),overall_results,corpora.Corpus.output_dir+output_filename)




def testPrebuilt():
    bla=loadPrebuiltBOW("c92-2117.xml","inlink_context",50)
    print bla


def main():
##    buildCorpusIDFdict(r"C:\NLP\PhD\bob\bob\files\*.xml", r"C:\NLP\PhD\bob\bob\idfdict.pic", 10000)
    print "Running testing pipeline"

##    import cProfile
##    cProfile.run('blah()',None,"calls")
##    cProfile.run('testRunTest1(1, 20, "results6.csv")',None,"time")

# ACL2014 paper methods
##    methods={
##    "full_text":{"type":"standard_multi", "parameters":[1]},
##    "passage":{"type":"standard_multi", "parameters":[250,350,400]},
##    "title_abstract":{"type":"standard_multi", "parameters":[1]},
##    "inlink_context":{"type":"standard_multi", "parameters": [10, 20, 30]},
##    "ilc_title_abstract":{"type":"ilc_mashup", "mashup_method":"title_abstract", "ilc_parameters":[10,20], "parameters":[1]},
##    "ilc_full_text":{"type":"ilc_mashup", "mashup_method":"full_text", "ilc_parameters":[10,20], "parameters":[1]},
##    "ilc_passage":{"type":"ilc_mashup", "mashup_method":"passage","ilc_parameters":[10, 20], "parameters":[250,350]}
##    }

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
    "full_text":{"type":"standard_multi", "index":"full_text", "parameters":[1], "runtime_parameters":{"text":"1"}},
##    "title_abstract":{"type":"standard_multi", "index":"title_abstract", "parameters":[1], "runtime_parameter":{"text":"1"}},
##    "passage":{"type":"standard_multi", "index":"passage", "parameters":[250,350,400], "runtime_parameter":{"text":"1"}},
##
##    "inlink_context":{"type":"standard_multi", "index":"inlink_context",
##        "parameters": [10, 20, 30], "runtime_parameter":{"inlink_context":"1"}},
##
##    "ilc_passage":{"type":"ilc_mashup",  "index":"ilc_passage", "mashup_method":"passage","ilc_parameters":[10, 20, 30, 40, 50],
##        "parameters":[250,350], "runtime_parameter":{"text":"1","inlink_context":"1"}},

##    "az_annotated":{"type":"annotated_boost", "index":"ilc_az_annotated_1_20", "parameters":[1], "runtime_parameters":
##        {"ALL":{"AIM":"1","BAS":"1","BKG":"1","CTR":"1","OTH":"1","OWN":"1","TXT":"1"},
##         "OTH_only":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"1","OWN":"0","TXT":"0"},
##         "OWN_only":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"0","OWN":"1","TXT":"0"},
##        }},
    "section":{"type":"annotated_boost", "index":"section_annotated_1", "parameters":[1], "runtime_parameters":
        {
##        "title_abstract":{"title":"1","abstract":"1"},
         "full_text":{"title":"1","abstract":"1","text":"1"},
        }},

##    "ilc":{"type":"ilc_annotated_boost", "index":"ilc_section_annotated", "ilc_parameters":[10, 20, 30, 40, 50], "parameters":[1], "runtime_parameters":
##        {"title_abstract":{"title":"1","abstract":"1","inlink_context":1},
##         "full_text":{"title":"1","abstract":"1","text":"1","inlink_context":1},
##        }},

##    "ilc_az_annotated":{"type":"ilc_annotated_boost", "index":"ilc_az_annotated", "parameters":[1], "ilc_parameters":[10, 20, 30, 40, 50], "runtime_parameters":
##        {"ALL":{"AIM":"1","BAS":"1","BKG":"1","CTR":"1","OTH":"1","OWN":"1","TXT":"1","inlink_context":1},
##         "OTH":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"1","OWN":"0","TXT":"0","inlink_context":1},
##         "OWN":{"AIM":"0","BAS":"0","BKG":"0","CTR":"0","OTH":"0","OWN":"1","TXT":"0","inlink_context":1},
##        }},
    }


    qmethods={"window":{"parameters":[
##                (3,3),
                (5,5),
##                (10,10),
##                (10,5),
##                (20,20),
##                (10,20),
##                (30,30)
                ],
                "function":context_extract.getOutlinkContextWindowAroundCitationMulti}}

##    qmethods={"window":{"parameters":[
##                (10,10),
##                (10,5),
##                ],
##                "function":getOutlinkContextWindowAroundCitationMulti}}


    corpora.Corpus.TEST_FILES=corpora.Corpus.listPapers("num_in_collection_references >= 8")
    runCitationResolutionLucene(corpora.Corpus.TEST_FILES,20,"results_lucene_4_9_replicate_test1.csv",testing_methods,qmethods)
##    runWholeIndexLucene(corpora.Corpus.TEST_FILES,20,"results_wholeindex_1.csv", methods, qmethods)

##    runCitationResolutionLucene(corpora.Corpus.TEST_FILES,20,"results_lucene_compareACL14_FIXACL3.csv",methods,qmethods)
##    runCitationResolutionLucene(["j02-3001"],20,"results_lucene_compareACL14_FIXACL2.csv",methods,qmethods)


    pass

if __name__ == '__main__':
    main()
