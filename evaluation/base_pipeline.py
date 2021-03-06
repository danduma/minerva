# Testing pipeline classes
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import os, sys, json
from copy import deepcopy
from collections import defaultdict

from retrieval.base_retrieval import BaseRetrieval, MAX_RESULTS_RECALL

import db.corpora as cp
from proc.results_logging import ResultsLogger
from .pipeline_functions import getDictOfTestingMethods
from .weight_functions import addExtraWeights
from six.moves import range


def analyticalRandomChanceMRR(numinlinks):
    """
        Returns the MRR score based on analytical random chance
    """
    res = 0
    for i in range(numinlinks):
        res += (1 / float(numinlinks)) * (1 / float(i + 1))
    return res


class BaseTestingPipeline(object):
    """
        Base class for testing pipelines
    """

    def __init__(self, retrieval_class=BaseRetrieval, use_celery=False):
        # This points to the the class of retrieval we are using
        self.retrieval_class = retrieval_class
        self.use_celery = use_celery
        self.tasks = []
        self.exp = None
        self.options = None
        self.precomputed_queries = []
        self.retrieval_models = {}
        self.files_dict = {}
        self.main_all_doc_methods = {}
        self.current_all_doc_methods = {}
        self.save_terms = False
        self.max_per_class_results = 1000
        self.previous_guid = ""
        self.per_class_count = {}

    def loadModelForSingleFile(self, guid):
        """
            Loads all the retrieval models for a single file
        """
        for model in self.files_dict[guid]["tfidf_models"]:
            # create a search instance for each method
            self.retrieval_models[model["method"]] = self.retrieval_class(
                model["actual_dir"],
                model["method"],
                logger=None,
                use_default_similarity=self.exp["use_default_similarity"])

    def generateRetrievalModels(self, all_doc_methods, all_files, ):
        """
            Generates the files_dict with the paths to the retrieval models
        """
        for guid in all_files:
            self.files_dict[guid]["tfidf_models"] = []
            for method in all_doc_methods:
                actual_dir = cp.Corpus.getRetrievalIndexPath(guid, all_doc_methods[method]["index_filename"],
                                                             self.exp["full_corpus"])
                self.files_dict[guid]["tfidf_models"].append({"method": method, "actual_dir": actual_dir})

    def addRandomControlResult(self, guid, precomputed_query):
        """
             Adds a result that is purely based on analytical chance, for
             comparison.
        """
        result_dict = {"file_guid": guid,
                       "citation_id": precomputed_query["citation_id"],
                       "doc_position": precomputed_query["doc_position"],
                       "query_method": precomputed_query["query_method"],
                       "match_guids": precomputed_query["match_guids"],
                       "doc_method": "RANDOM",
                       "mrr_score": analyticalRandomChanceMRR(self.files_dict[guid]["in_collection_references"]),
                       "precision_score": 1 / float(self.files_dict[guid]["in_collection_references"]),
                       "ndcg_score": 0,
                       "rank": 0,
                       "first_result": ""
                       }

        # Deal here with CoreSC/AZ/CFC annotation
        for annotation in self.exp.get("rhetorical_annotations", []):
            result_dict[annotation] = precomputed_query.get(annotation)

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
                lucene.initVM(maxheap="640m")  # init Lucene VM
            except ValueError:
                # VM already up
                print(sys.exc_info()[1])

    def startLogging(self):
        """
            Creates the results logger and starts counting and logging.
        """
        output_filename = os.path.join(self.exp["exp_dir"], self.exp.get("output_filename", "results.csv"))
        self.logger = ResultsLogger(len(self.precomputed_queries),
                                    dump_filename=output_filename,
                                    message_text="Running precomputed queries",
                                    dot_every_xitems=1,
                                    )  # init all the logging/counting
        self.logger.startCounting()  # for timing the process, start now

    def loadQueriesAndFileList(self):
        """
            Loads the precomputed queries and the list of test files to process.
        """
        precomputed_queries_file_path = self.exp.get("precomputed_queries_file_path", None)
        if not precomputed_queries_file_path:
            precomputed_queries_file_path = os.path.join(self.exp["exp_dir"],
                                                         self.exp.get("precomputed_queries_filename",
                                                                      "precomputed_queries.json"))
        self.precomputed_queries = json.load(open(precomputed_queries_file_path, "r"))

        self.precomputed_queries = self.precomputed_queries[self.options.get("run_query_start_at", 0):]

        max_queries_to_process = self.options.get("max_queries_to_process", None)
        if max_queries_to_process:
            self.precomputed_queries = self.precomputed_queries[:max_queries_to_process]

        files_dict_filename = os.path.join(self.exp["exp_dir"], self.exp.get("files_dict_filename", "files_dict.json"))
        self.files_dict = json.load(open(files_dict_filename, "r"))
        self.files_dict["ALL_FILES"] = {}

    def populateMethods(self):
        """
            Fills dict with all the test methods, parameters and options, including
            the retrieval instances
        """
        self.retrieval_models = {}
        all_doc_methods = None

        if self.exp.get("doc_methods", None):
            all_doc_methods = getDictOfTestingMethods(self.exp["doc_methods"])
            # essentially this overrides whatever is in files_dict, if testing_methods was passed as parameter

            if self.exp["full_corpus"]:
                all_files = ["ALL_FILES"]
            else:
                all_files = list(self.files_dict.keys())

            self.generateRetrievalModels(all_doc_methods, all_files)
        else:
            all_doc_methods = self.files_dict["ALL_FILES"]["doc_methods"]  # load from files_dict

        if self.exp["full_corpus"]:
            for model in self.files_dict["ALL_FILES"]["tfidf_models"]:
                # create a search instance for each method
                self.retrieval_models[model["method"]] = self.retrieval_class(
                    model["actual_dir"],
                    model["method"],
                    logger=None,
                    use_default_similarity=self.exp["use_default_similarity"],
                    max_results=self.exp["max_results_recall"],
                    save_terms=self.save_terms,
                    multi_match_type=all_doc_methods[model["method"]].get("multi_match_type"))

        self.main_all_doc_methods = all_doc_methods

    def newResultDict(self, guid, precomputed_query, doc_method):
        """
            Creates and populates a new result dict.
        """
        result_dict = {"file_guid": guid,
                       "citation_id": precomputed_query.get("citation_id", precomputed_query.get("cit_ids", [0])[0]),
                       "doc_position": precomputed_query.get("doc_position", 0),
                       "query_method": precomputed_query.get("query_method"),
                       "doc_method": doc_method,
                       "match_guids": precomputed_query["match_guids"]}

        # Deal here with CoreSC/AZ/CFC annotation
        for annotation in self.exp.get("rhetorical_annotations", []):
            result_dict[annotation] = precomputed_query.get(annotation)

        return result_dict

    def addEmptyResult(self, guid, precomputed_query, doc_method):
        """
            Adds an empty result, that is, a result with 0 score due to some error.
        """
        result_dict = self.newResultDict(guid, precomputed_query, doc_method)
        result_dict["mrr_score"] = 0
        result_dict["precision_score"] = 0
        result_dict["ndcg_score"] = 0
        result_dict["rank"] = 0
        result_dict["first_result"] = ""
        self.logger.addResolutionResultDict(result_dict)

    def addResult(self, file_guid, precomputed_query, doc_method, retrieved_results):
        """
            Adds a normal (successful) result to the result log.

            :param file_guid: the GUID of the file that the citation is IN
            :param precomputed_query: a dict with the precomputed query
            :param doc_method: current doc_method we are testing (way of indexing the documents)
            :param retrieved_results: list of all GUIDs retrieved with the query. Used to measure score.
        """
        result_dict = self.newResultDict(file_guid, precomputed_query, doc_method)
        self.logger.measureScoreAndLog(retrieved_results,
                                       precomputed_query.get("citation_multi",
                                                             precomputed_query.get("cit_multi", 1)),
                                       result_dict)
        # if result_dict["mrr_score"] < 0.1:
        #     print("FUCK. FIX THIS", result_dict["mrr_score"], doc_method)

    ##        rank_per_method[result["doc_method"]].append(result["rank"])
    ##        precision_per_method[result["doc_method"]].append(result["precision_score"])

    ##    def logTextAndReferences(self, doctext, queries, qmethod):
    ##        """
    ##            Extra logging, not used right now
    ##        """
    ##        pre_selection_text=doctext[queries[qmethod]["left_start"]-300:queries[qmethod]["left_start"]]
    ##        draft_text=doctext[queries[qmethod]["left_start"]:queries[qmethod]["right_end"]]
    ##        post_selection_text=doctext[queries[qmethod]["right_end"]:queries[qmethod]["left_start"]+300]
    ##        draft_text=u"<span class=document_text>{}</span> <span class=selected_text>{}</span> <span class=document_text>{}</span>".format(pre_selection_text, draft_text, post_selection_text)
    ##        print(draft_text)

    def saveResultsAndCleanUp(self):
        """
            This executes after the whole pipeline is done. This is where we
            save all data that needs to be saved, report statistics, etc.
        """
        self.logger.writeDataToCSV()
        self.logger.showFinalSummary()

    def setRuntimeParameters(self, method, precomputed_query):
        """
        Sets the runtime parameters to use depending on the method and query

        """
        actual_runtime_parameters = self.main_all_doc_methods[method].get("runtime_parameters")

        if isinstance(actual_runtime_parameters, list):
            return {x: 1 for x in self.main_all_doc_methods[method]["runtime_parameters"]}
        elif isinstance(actual_runtime_parameters, dict):
            return self.main_all_doc_methods[method]["runtime_parameters"]

    def processOneQuery(self, precomputed_query):
        """
            Runs the retrieval and evaluation for a single query
        """
        if self.exp.get("queries_classification", "") not in ["", None]:
            query_class = self.exp.get("queries_classification", None)
            if query_class:
                q_type = precomputed_query[query_class]
                if self.per_class_count[q_type] < self.max_per_class_results:
                    self.per_class_count[q_type] += 1
                else:
                    # print("Too many queries of type %s already" % q_type)
                    return

        guid = precomputed_query["file_guid"]
        meta = cp.Corpus.getMetadataByGUID(guid)
        self.logger.total_citations += meta["num_resolvable_citations"]

        all_doc_methods = deepcopy(self.main_all_doc_methods)

        # If we're running per-file resolution and we are now on a different file, load its model
        if not self.exp["full_corpus"] and guid != self.previous_guid:
            self.previous_guid = guid
            self.loadModelForSingleFile(guid)

        # create a dict where every field gets a weight of 1 or whatever the pipeline uses
        for method in self.main_all_doc_methods:
            all_doc_methods[method]["runtime_parameters"] = self.setRuntimeParameters(method, precomputed_query)

        self.current_all_doc_methods = all_doc_methods

        # for every method used for extracting BOWs
        for doc_method in all_doc_methods:
            # Log everything if the logger is enabled
            ##                self.logger.logReport("Citation: "+precomputed_query["citation_id"]+"\n Query method:"+precomputed_query["query_method"]+" \nDoc method: "+doc_method +"\n")
            ##                self.logger.logReport(precomputed_query["query_text"]+"\n")

            # ACTUAL RETRIEVAL HAPPENING - run query
            retrieval_results = self.retrieval_models[doc_method].runQuery(
                precomputed_query,
                addExtraWeights(all_doc_methods[doc_method]["runtime_parameters"], self.exp),
                test_guid=None,
                max_results=self.exp.get("max_results_recall", MAX_RESULTS_RECALL))

            # print("Query:", precomputed_query["query_text"])
            # print(addExtraWeights(all_doc_methods[doc_method]["runtime_parameters"], self.exp))

            if not retrieval_results:  # the query was empty or something
                self.addEmptyResult(guid, precomputed_query, doc_method)
            else:
                self.addResult(guid, precomputed_query, doc_method, retrieval_results)

        if self.exp.get("add_random_control_result", False):
            self.addRandomControlResult(guid, precomputed_query)

        self.logger.showProgressReport("Running queries")  # prints out info on how it's going

    def processAllQueries(self):
        """
            MAIN LOOP over all precomputed queries
        """
        for precomputed_query in self.precomputed_queries:
            self.processOneQuery(precomputed_query)

    def annotateDocuments(self):
        """
            To be overriden by descendant classes: run annotators on documents
            if needed
        """
        pass

    def runPipeline(self, exp, options):
        """
            Run the whole experiment pipeline, loading everything from
            precomputed json

            :param exp: experiment dict
        """
        self.exp = exp
        self.options = options

        self.max_per_class_results = self.exp.get("max_per_class_results", self.max_per_class_results)
        self.per_class_count = defaultdict(lambda: 0)
        if self.exp.get("similiarity_tie_breaker", 0):
            for model in self.retrieval_models.items():
                model.tie_breaker = self.exp["similiarity_tie_breaker"]

        self.initializePipeline()
        self.loadQueriesAndFileList()
        self.startLogging()
        self.annotateDocuments()
        self.logger.setNumItems(len(self.precomputed_queries))
        self.populateMethods()

        self.previous_guid = ""

        # MAIN LOOP over all precomputed queries
        self.processAllQueries()

        self.saveResultsAndCleanUp()


def main():
    pass


if __name__ == '__main__':
    main()
