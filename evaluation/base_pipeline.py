# Testing pipeline classes
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import os, sys, json
from copy import deepcopy
from collections import OrderedDict

from minerva.retrieval.base_retrieval import BaseRetrieval, MAX_RESULTS_RECALL

import minerva.db.corpora as cp
from minerva.proc.results_logging import ResultsLogger, ProgressIndicator
from pipeline_functions import getDictOfTestingMethods

def analyticalRandomChanceMRR(numinlinks):
    """
        Returns the MRR score based on analytical random chance
    """
    res=0
    for i in range(numinlinks):
        res+=(1/float(numinlinks))*(1/float(i+1))
    return res

class BaseTestingPipeline(object):
    """
        Base class for testing pipelines
    """
    def __init__(self, retrieval_class=BaseRetrieval, use_celery=False):
        # This points to the the class of retrieval we are using
        self.retrieval_class=retrieval_class
        self.use_celery=use_celery
        self.tasks=[]
        self.exp={}
        self.options={}
        self.precomputed_queries=[]
        self.tfidfmodels={}
        self.files_dict={}
        self.main_all_doc_methods={}
        self.current_all_doc_methods={}
        self.save_terms=False

    def loadModel(self, guid):
        """
            Loads all the retrieval models for a single file
        """
        for model in self.files_dict[guid]["tfidf_models"]:
            # create a search instance for each method
            self.tfidfmodels[model["method"]]=self.retrieval_class(
                model["actual_dir"],
                model["method"],
                logger=None,
                use_default_similarity=self.exp["use_default_similarity"])

    def generateRetrievalModels(self, all_doc_methods, all_files,):
        """
            Generates the files_dict with the paths to the retrieval models
        """
        for guid in all_files:
            self.files_dict[guid]["tfidf_models"]=[]
            for method in all_doc_methods:
                actual_dir=cp.Corpus.getRetrievalIndexPath(guid,all_doc_methods[method]["index_filename"],self.exp["full_corpus"])
                self.files_dict[guid]["tfidf_models"].append({"method":method,"actual_dir":actual_dir})

    def addRandomControlResult(self, guid, precomputed_query):
        """
             Adds a result that is purely based on analytical chance, for
             comparison.
        """
        result_dict={"file_guid":guid,
                     "citation_id":precomputed_query["citation_id"],
                     "doc_position":precomputed_query["doc_position"],
                     "query_method":precomputed_query["query_method"],
                     "match_guid":precomputed_query["match_guid"],
                     "doc_method":"RANDOM",
                     "mrr_score":analyticalRandomChanceMRR(self.files_dict[guid]["in_collection_references"]),
                     "precision_score":1/float(self.files_dict[guid]["in_collection_references"]),
                     "ndcg_score":0,
                     "rank":0,
                     "first_result":""
                     }

        # Deal here with CoreSC/AZ/CFC annotation
        for annotation in self.exp.get("rhetorical_annotations",[]):
            result_dict[annotation]=precomputed_query.get(annotation)

        self.logger.addResolutionResultDict(result_dict)


    def initializePipeline(self):
        """
            Whatever needs to happen before we start the pipeline: inializing
            connections, VMs, whatever.

            This function should be overriden by descendant classes if anything
            is to be done.
        """
        if self.retrieval_class.__name__.startswith("Lucene"):
            import lucene
            try:
                lucene.initVM(maxheap="640m") # init Lucene VM
            except ValueError:
                # VM already up
                print(sys.exc_info()[1])

    def startLogging(self):
        """
            Creates the results logger and starts counting and logging.
        """
        output_filename=os.path.join(self.exp["exp_dir"],self.exp.get("output_filename","results.csv"))
        self.logger=ResultsLogger(False,
                                  dump_filename=output_filename,
                                  message_text="Running precomputed queries",
                                  dot_every_xitems=1,
                                  ) # init all the logging/counting
        self.logger.startCounting() # for timing the process, start now

    def loadQueriesAndFileList(self):
        """
            Loads the precomputed queries and the list of test files to process.
        """
        precomputed_queries_file_path=self.exp.get("precomputed_queries_file_path",None)
        if not precomputed_queries_file_path:
            precomputed_queries_file_path=os.path.join(self.exp["exp_dir"],self.exp.get("precomputed_queries_filename","precomputed_queries.json"))
        self.precomputed_queries=json.load(open(precomputed_queries_file_path,"r"))
        files_dict_filename=os.path.join(self.exp["exp_dir"],self.exp.get("files_dict_filename","files_dict.json"))
        self.files_dict=json.load(open(files_dict_filename,"r"))
        self.files_dict["ALL_FILES"]={}

    def populateMethods(self):
        """
            Fills dict with all the test methods, parameters and options, including
            the retrieval instances
        """
        self.tfidfmodels={}
        all_doc_methods=None

        if self.exp.get("doc_methods", None):
            all_doc_methods=getDictOfTestingMethods(self.exp["doc_methods"])
            # essentially this overrides whatever is in files_dict, if testing_methods was passed as parameter

            if self.exp["full_corpus"]:
                all_files=["ALL_FILES"]
            else:
                all_files=self.files_dict.keys()

            self.generateRetrievalModels(all_doc_methods,all_files)
        else:
            all_doc_methods=self.files_dict["ALL_FILES"]["doc_methods"] # load from files_dict

        if self.exp["full_corpus"]:
            for model in self.files_dict["ALL_FILES"]["tfidf_models"]:
                # create a search instance for each method
                self.tfidfmodels[model["method"]]=self.retrieval_class(
                        model["actual_dir"],
                        model["method"],
                        logger=None,
                        use_default_similarity=self.exp["use_default_similarity"],
                        max_results=self.exp["max_results_recall"],
                        save_terms=self.save_terms)

        self.main_all_doc_methods=all_doc_methods

    def newResultDict(self, guid, precomputed_query, doc_method):
        """
            Creates and populates a new result dict.
        """
        result_dict={"file_guid":guid,
                     "citation_id":precomputed_query["citation_id"],
                     "doc_position":precomputed_query["doc_position"],
                     "query_method":precomputed_query["query_method"],
                     "doc_method":doc_method ,
                     "match_guid":precomputed_query["match_guid"]}

        # Deal here with CoreSC/AZ/CFC annotation
        for annotation in self.exp.get("rhetorical_annotations",[]):
            result_dict[annotation]=precomputed_query.get(annotation)

        return result_dict

    def addEmptyResult(self, guid, precomputed_query, doc_method):
        """
            Adds an empty result, that is, a result with 0 score due to some error.
        """
        result_dict=self.newResultDict(guid, precomputed_query, doc_method)
        result_dict["mrr_score"]=0
        result_dict["precision_score"]=0
        result_dict["ndcg_score"]=0
        result_dict["rank"]=0
        result_dict["first_result"]=""
        self.logger.addResolutionResultDict(result_dict)

    def addResult(self, guid, precomputed_query, doc_method, retrieved_results):
        """
            Adds a normal (successful) result to the result log.
        """
        result_dict=self.newResultDict(guid, precomputed_query, doc_method)
        self.logger.measureScoreAndLog(retrieved_results, precomputed_query["citation_multi"], result_dict)
##        rank_per_method[result["doc_method"]].append(result["rank"])
##        precision_per_method[result["doc_method"]].append(result["precision_score"])

    def logTextAndReferences(self, doctext, queries, qmethod):
        """
            Extra logging, not used right now
        """
        pre_selection_text=doctext[queries[qmethod]["left_start"]-300:queries[qmethod]["left_start"]]
        draft_text=doctext[queries[qmethod]["left_start"]:queries[qmethod]["right_end"]]
        post_selection_text=doctext[queries[qmethod]["right_end"]:queries[qmethod]["left_start"]+300]
        draft_text=u"<span class=document_text>{}</span> <span class=selected_text>{}</span> <span class=document_text>{}</span>".format(pre_selection_text, draft_text, post_selection_text)
##        print(draft_text)


    def saveResultsAndCleanUp(self):
        """
            This executes after the whole pipeline is done. This is where we
            save all data that needs to be saved, report statistics, etc.
        """
        self.logger.writeDataToCSV()
        self.logger.showFinalSummary()

    def runPipeline(self, exp):
        """
            Run the whole experiment pipeline, loading everything from
            precomputed json

            :param exp: experiment dict
        """
        self.exp=exp

        self.startLogging()
        self.initializePipeline()
        self.loadQueriesAndFileList()
        self.logger.setNumItems(len(self.precomputed_queries))
        self.populateMethods()

##        methods_overlap=0
##        total_overlap_points=0
##        rank_differences=[]
##        rank_per_method=defaultdict(lambda:[])
##        precision_per_method=defaultdict(lambda:[])

        previous_guid=""

        #=======================================
        # MAIN LOOP over all precomputed queries
        #=======================================
        for precomputed_query in self.precomputed_queries:
            guid=precomputed_query["file_guid"]
            self.logger.total_citations+=self.files_dict[guid]["resolvable_citations"]

            all_doc_methods=deepcopy(self.main_all_doc_methods)

            # If we're running per-file resolution and we are now on a different file, load its model
            if not exp["full_corpus"] and guid != previous_guid:
                previous_guid=guid
                self.loadModel(guid)

            # create a dict where every field gets a weight of 1
            for method in self.main_all_doc_methods:
                all_doc_methods[method]["runtime_parameters"]={x:1 for x in self.main_all_doc_methods[method]["runtime_parameters"]}

            self.current_all_doc_methods=all_doc_methods

            # for every method used for extracting BOWs
            for doc_method in all_doc_methods:
                # Log everything if the logger is enabled
                self.logger.logReport("Citation: "+precomputed_query["citation_id"]+"\n Query method:"+precomputed_query["query_method"]+" \nDoc method: "+doc_method +"\n")
                self.logger.logReport(precomputed_query["query_text"]+"\n")

                # ACTUAL RETRIEVAL HAPPENING - run query
                retrieved=self.tfidfmodels[doc_method].runQuery(
                    precomputed_query,
                    all_doc_methods[doc_method]["runtime_parameters"],
                    guid,
                    max_results=exp.get("max_results_recall",MAX_RESULTS_RECALL))

                if not retrieved:    # the query was empty or something
                    self.addEmptyResult(guid, precomputed_query, doc_method)
                else:
                    self.addResult(guid, precomputed_query, doc_method, retrieved)

            if self.exp.get("add_random_control_result", False):
                self.addRandomControlResult(guid, precomputed_query)

            self.logger.showProgressReport(guid) # prints out info on how it's going

        self.saveResultsAndCleanUp()

class CompareExplainPipeline(BaseTestingPipeline):
    """
        This compared the results of the default similarity with those of the
        explain pipeline. Deprecated and to be discontinued.
    """
    def __init__(self):
        pass

    def populateMethods(self):
        """
        """
        super(LuceneTestingPipeline, self).populateMethods()

        if self.exp.get("compare_explain",False):
            for method in self.main_all_doc_methods:
                self.main_all_doc_methods[method+"_EXPLAIN"]=self.main_all_doc_methods[method]

    def loadModel(self, model, exp):
        """
            Overrides the default loadModel to add explain models.
        """
        super(self.__class__, self).loadModel(model, exp)

        # this is to compare bulkScorer and .explain() on their overlap
        self.tfidfmodels[model["method"]+"_EXPLAIN"]=self.retrieval_class(
            model["actual_dir"],
            model["method"],
            logger=None,
            use_default_similarity=exp["use_default_similarity"])
        self.tfidfmodels[model["method"]+"_EXPLAIN"].useExplainQuery=True

def main():
    pass

if __name__ == '__main__':
    main()
