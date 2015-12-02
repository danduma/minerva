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

from bow_methods import *

import lucene
from lucene import \
    SimpleFSDirectory, System, File, \
    Document, Field, StandardAnalyzer, IndexWriter, IndexSearcher, QueryParser

import glob, math, os, datetime

# this is the corpus instance used for all experiments
print "Loading ACL bob corpus..."
corpora.Corpus=ACLbobCorpusClass(r"G:\NLP\PhD\bob")

class luceneRetrieval:
    def __init__(self, indexDir, method):
        self.indexDir=indexDir

        self.analyzer = StandardAnalyzer(lucene.Version.LUCENE_CURRENT)
        self.searcher = IndexSearcher(SimpleFSDirectory(File(self.indexDir)))
        self.max_results=100

    def cleanupQuery(self,query):
        rep_list=["~", "and"]
        rep_list.extend(punctuation)
        query=query.lower()
        for r in rep_list:
            query=query.replace(r," ")
        query=re.sub(r"\s+"," ",query)
        query=query.strip()
        return query

    def runQuery(self, query):
        # LOTS OF SWEET LUCENE
        original_query=query
        query=self.cleanupQuery(query)
        if query=="":
            return None

        self.last_query=query

        try:
            query = QueryParser(lucene.Version.LUCENE_CURRENT, "text", self.analyzer).parse(query)
        except:
            print "Lucene exception:",sys.exc_info()[:2]
            return None

        hits = self.searcher.search(query, self.max_results)
##        print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)
        res=[]
##        logReport(self.searcher.explain(query,0))

        if len(hits.scoreDocs) ==0:
            print "Original query:",original_query
            print "Query:", query

        for hit in hits.scoreDocs:
            doc = self.searcher.doc(hit.doc)
            metadata= json.loads(doc.get("metadata"))
            res.append((hit.score,metadata))
##            print hit.score, hit.doc, hit.toString()
##            doc = searcher.doc(hit.doc)
##            print doc.get("text").encode("utf-8")
##            print doc.get("metadata").encode("utf-8")
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

def measureMRRscore(docs, match):
    """
        Returns a score for the match depending on what position in the list of
        ranked retrieved documents the correct one is
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
                score=1/ float(index+1)
            return score
    return 0

def measurePrecisionScore(docs, match):
    """
        Returns a score for the match depending on what position in the list of
        ranked retrieved documents the correct one is
    """
    # docs must be already ordered in descending order
    guids=[]
    for doc in docs:
        guid=doc[1]["guid"]
        if guid not in guids:
            guids.append(guid)

    for index, guid in enumerate(guids):
        if guid==match["match_guid"]:
            if match["cit"].has_key("multi"):
                if index+1 <= match["cit"]["multi"]:
                    score=1
                else:
                    score=0  # 1 or 0
            else:
                score=1 if index==0 else 0
            return score
    return 0


def measureDCGscore(docs, match):
    """
        Returns Discounted Cumulative Gain
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


#getOutlinkContextWindowAroundCitationMulti

# for evert document:
# for every citation, generate outlink BOW
# open every referred document, keep in memory
# extract features from that document:
    # simple Vector Space Model
        # just abstract / title
            # for documents with no marked abstract:
                # if not many, mark by hand
                # OR try first paragraph
        # all document text
# measure error:
    # for every citation, where in the ranked list of retrieved documents the right one is
    # get error per document and per corpus
    # auto-tune parameters to minimize error?

# save CSV with results

def loadMatchableReferences(matchable_citations, metadata, working_dir=r"C:\NLP\PhD\bob\bob\files\\"):
    """
        Loads all the documents for which a match was found in the collection/corpus
    """
    res={}

    for cit in matchable_citations:
##        print cit["match"]["filename"]
        fn=cit["match"]["filename"].lower()
        if fn not in res:
            doc=loadSciXMLCache(working_dir+fn, load_content=True, metadata_index=metadata)
##            print fn
            res[fn]=doc
        else:
            doc=res[fn]
        cit["doc"]=doc
    return res

report_file=None

def logReport(what):
    if report_file:
        report_file.write(what.__repr__()+"\n")


#===============================================================================
#   TESTING PIPELINE
#===============================================================================

def runCitationResolutionLucene(FILE_LIST, numchunks=10, output_filename="results.csv", bowmethods={}, querymethods={}):
    """
        Using Lucene
    """
    global report_file
    report_file=codecs.open(corpora.Corpus.output_dir+"report.txt","w")
    now1=datetime.datetime.now()

    # init Lucene VM
    lucene.initVM(maxheap="768m")

    overall_results=[]
    text_results=[]

    mrr={} # dict for Mean Reciprocal Rank scores
##    dcg={} # Discounted Cumulative Gain
##    ndcg={} # Normalized DCG
    precision={} # Exclusively precision score. 1 if right, 0 otherwise

    total_citations=0

    dot_every_xfiles=max(len(FILE_LIST) / 1000,1)
    print "Processing ", len(FILE_LIST), "papers..."

    irdocs=getDictOfTestingMethods(bowmethods) # convert nested dict to flat
    tfidfmodels={} # same as above

    count=0
    # MAIN LOOP over all testing files
    for guid in FILE_LIST:
        count+=1
        if count % dot_every_xfiles ==0:
            reportTimeLeft(count,len(FILE_LIST), now1, guid)

        doc=corpora.Corpus.loadSciDoc(guid)
        if not doc:
            print "ERROR: Couldn't load pickled doc:",guid
            continue
        doctext=doc.getFullDocumentText()

        tin_can=corpora.Corpus.loadMatchableCitations(guid)
        matchable=tin_can["matchable"]
        in_collection_references=tin_can["outlinks"]

        total_citations+=len(matchable)
        num_in_collection_references=len(in_collection_references)
        print "Matchable citations:",len(matchable), "In-collection references:",num_in_collection_references

        for method in irdocs:
            actual_dir=corpora.Corpus.fileLuceneIndex_dir+guid+os.sep+method
            tfidfmodels[method]=luceneRetrieval(actual_dir,method)
            # create a Lucene search instance for each method

        print "Resolving citations..."
        for m in matchable:
            queries={}

            match=findCitationInFullText(m["cit"],doctext)
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

            logReport("CITATION")
            logReport("")
            # for every generated query for this context
            for qmethod in queries:
                mrr[qmethod]=mrr.get(qmethod,{})
                precision[qmethod]=precision.get(qmethod,{})

                # for every method used for extracting BOWs
                for method in irdocs.keys():
                    logReport("QUERY METHOD:"+qmethod)
                    logReport("Query text:"+queries[qmethod])
                    logReport("BOW METHOD:"+method)
                    logReport("Explanation:")

                    # ACTUAL RETRIEVAL HAPPENING - run query
                    retrieved=tfidfmodels[method].runQuery(queries[qmethod])

                    if not retrieved:    # the query was empty or something
                        score=0
                        precision_score=0
##                        print "Error: ", method, qmethod,tfidfmodels[method].indexDir
                    else:
                        score=measureMRRscore(retrieved,m)
                        precision_score=measurePrecisionScore(retrieved,m)
##                        print "Success: ", method, qmethod,tfidfmodels[method].indexDir

                    logReport("Precision Score: "+ precision_score.__repr__())
                    logReport("Correct: "+ m.__repr__())
                    logReport("Results: ")
                    if retrieved:
                        for r in retrieved[:5]:
                            logReport(r)

##                    if precision_score==1:
##                        text_results.append("[Q[ "+queries[qmethod]+"]Q] == "+"")
                    source=""
                    target=""
##                    source=" ".join(queries[qmethod])
##                    target=" ".join(retrieved[0][1]["BOW"])
                    overall_results.append((guid,m["cit"]["id"], doc_position, qmethod, method, score, precision_score,
                    source, target, m["match_guid"]
                    ))
                    mrr[qmethod][method]=mrr[qmethod].get(method,0)+score
                    precision[qmethod][method]=precision[qmethod].get(method,0)+precision_score

                # extra method = random chance control
                method="RANDOM"

                score=analyticalRandomChanceMRR(num_in_collection_references)
                precision_score=1/float(num_in_collection_references)

                overall_results.append((guid,m["cit"]["id"], doc_position, qmethod, method, score, precision_score, "","","" ))
                mrr[qmethod][method]=mrr[qmethod].get(method,0)+score
                precision[qmethod][method]=precision[qmethod].get(method,0)+precision_score

    methods=irdocs.keys()
    methods.append("RANDOM")

    print
    print "========================================================================="
    print "Docs processed:", len(FILE_LIST)
    print "Total citations:", total_citations
    print
    print "Retrieval model: Lucene"
    print

    for qmethod in queries:
        print "Query method:", qmethod
        for method in methods:
##            total_mrr=mrr[qmethod][method]/float(total_citations)
##            print "MRR for", method, total_mrr
            total_precision=precision[qmethod][method]/float(total_citations)
            print "Precision for", method, total_precision

    print "Chunks:",numchunks
    print "Saved to filename:", output_filename

    now2=datetime.datetime.now()-now1
    print "Total execution time",now2

##    logfile=open("results.txt","wb")
##    logfile.writelines(results)
    writeTuplesToCSV("Filename citation_id doc_section query_method doc_method MRR_score Precision Query BOW Correct_file".split(),overall_results,corpora.Corpus.output_dir+output_filename)


def runWholeIndexLucene(FILE_LIST, numchunks=10, output_filename="results.csv", bowmethods={}, querymethods={}):
    """
        Using Lucene
    """
    now1=datetime.datetime.now()

    # init Lucene VM
    lucene.initVM(maxheap="768m")

    overall_results=[]
    text_results=[]

    count=0

    mrr={} # dict for Mean Reciprocal Rank scores
    dcg={} # Discounted Cumulative Gain
    ndcg={} # Normalized DCG
    precision={} # Exclusively precision score. 1 if right, 0 otherwise

    total_citations=0

    print "Processing ", len(FILE_LIST), "files..."

    irdocs=getDictOfTestingMethods(bowmethods) # convert nested dict to flat
    tfidfmodels={} # same as above

    # if testing over whole collection and we DON'T NEED to load the contents
##    fn_index={}
    files_with_inlinks={}
    reference_files={}

    for method in irdocs:
        actual_dir=corpora.Corpus.fullLuceneIndex_dir+method
        tfidfmodels[method]=luceneRetrieval(actual_dir,method)

    dot_every_xfiles=max(len(FILE_LIST) / 1000,1)
    print "Processing",len(FILE_LIST),"files:"

    # MAIN LOOP over all testing files
    for guid in FILE_LIST:
        count+=1
        if count % dot_every_xfiles ==0:
            reportTimeLeft(count,len(ALL_GUIDS), now1, guid)
##        print "Processing ", count," - ", guid

        doc=corpora.Corpus.loadSciDoc(guid)
        if not doc:
            print "ERROR: Couldn't load pickled doc:",guid
            continue
        doctext=doc.getFullDocumentText()

        tin_can=corpora.Corpus.loadMatchableCitations(guid)
        matchable=tin_can["matchable"]
        in_collection_references=tin_can["outlinks"]

        total_citations+=len(matchable)
        num_in_collection_references=len(in_collection_references)
        print "Matchable citations:",len(matchable), "In-collection references:",num_in_collection_references

        print "Finding best citations from whole corpus..."
        for m in matchable:
            queries={}

            match=findCitationInFullText(m["cit"],doctext)
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

            # for every generated query for this context
            for qmethod in queries:
                mrr[qmethod]=mrr.get(qmethod,{})
                ndcg[qmethod]=ndcg.get(qmethod,{})
                precision[qmethod]=precision.get(qmethod,{})

                # for every method used for extracting BOWs
                for method in irdocs.keys():

                    # ACTUAL RETRIEVAL HAPPENING - run query
                    retrieved=tfidfmodels[method].runQuery(queries[qmethod])

                    if not retrieved:    # the query was empty or something
                        score=0
                        precision_score=0
##                        print "Error: empty vocabulary/query"
                    else:
                        score=measureMRRscore(retrieved,m)
                        precision_score=measurePrecisionScore(retrieved,m)
                        ndcg_score=measureDCGscore(retrieved,m)


##                    if precision_score==1:
##                        text_results.append("[Q[ "+queries[qmethod]+"]Q] == "+"")

                    source=""
                    target=""
##                    source=" ".join(queries[qmethod])
##                    target=" ".join(retrieved[0][1]["BOW"])
                    overall_results.append((guid,m["cit"]["id"], doc_position, qmethod, method, score, precision_score,
                    source, target, m["match_guid"]
                    ))
                    mrr[qmethod][method]=mrr[qmethod].get(method,0)+score
                    precision[qmethod][method]=precision[qmethod].get(method,0)+precision_score

                # extra method = random chance control
                method="RANDOM"

                score=analyticalRandomChanceMRR(num_in_collection_references)
                precision_score=1/float(num_in_collection_references)

                overall_results.append((guid,m["cit"]["id"], doc_position, qmethod, method, score, precision_score, "","","" ))
                mrr[qmethod][method]=mrr[qmethod].get(method,0)+score
                precision[qmethod][method]=precision[qmethod].get(method,0)+precision_score

    methods=irdocs.keys()
    methods.append("RANDOM")

    print
    print "========================================================================="
    print "Docs processed:", len(FILE_LIST)
    print "Total citations:", total_citations
    print
    print "Retrieval model: Lucene"
    print

    for qmethod in queries:
        print "Query method:", qmethod
        for method in methods:
##            total_mrr=mrr[qmethod][method]/float(total_citations)
##            print "MRR for", method, total_mrr
            total_precision=precision[qmethod][method]/float(total_citations)
            print "Precision for", method, total_precision

    print "Chunks:",numchunks
    print "Saved to filename:", output_filename

    now2=datetime.datetime.now()-now1
    print "Total execution time",now2

##    logfile=open("results.txt","wb")
##    logfile.writelines(results)
    writeTuplesToCSV("Filename citation_id doc_section query_method doc_method MRR_score Precision Query BOW Correct_file".split(),overall_results,corpora.Corpus.output_dir+output_filename)


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
    methods={
    "full_text":{"type":"standard_multi", "parameters":[1]},
    "passage":{"type":"standard_multi", "parameters":[250,350,400]},
    "title_abstract":{"type":"standard_multi", "parameters":[1]},
    "inlink_context":{"type":"standard_multi", "parameters": [10, 20, 30]},
    "ilc_title_abstract":{"type":"ilc_mashup", "mashup_method":"title_abstract", "ilc_parameters":[10,20], "parameters":[1]},
    "ilc_full_text":{"type":"ilc_mashup", "mashup_method":"full_text", "ilc_parameters":[10,20], "parameters":[1]},
    "ilc_passage":{"type":"ilc_mashup", "mashup_method":"passage","ilc_parameters":[10, 20], "parameters":[250,350]}
    }

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


    qmethods={"window":{"parameters":[
##                (3,3),
                (5,5),
                (10,10),
                (10,5),
                (20,20),
##                (10,20),
                (30,30)
                ],
                "function":getOutlinkContextWindowAroundCitationMulti}}

##    qmethods={"window":{"parameters":[
##                (10,10),
##                (10,5),
##                ],
##                "function":getOutlinkContextWindowAroundCitationMulti}}


    corpora.Corpus.TEST_FILES=corpora.Corpus.listPapers("num_in_collection_references >= 8")
##    runCitationResolutionLucene(corpora.Corpus.TEST_FILES,20,"results_lucene_compareACL14_allinlinks.csv",methods,qmethods)
    runWholeIndexLucene(corpora.Corpus.TEST_FILES,20,"results_wholeindex_1.csv", methods, qmethods)

##    runCitationResolutionLucene(corpora.Corpus.TEST_FILES,20,"results_lucene_compareACL14_FIXACL3.csv",methods,qmethods)
##    runCitationResolutionLucene(["j02-3001"],20,"results_lucene_compareACL14_FIXACL2.csv",methods,qmethods)


    pass

if __name__ == '__main__':
    main()
