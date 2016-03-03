# <purpose>
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import json, os
from copy import deepcopy

from base_pipeline import BaseTestingPipeline
from base_retrieval import BaseRetrieval
from stored_formula import StoredFormula

from minerva.proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11
from minerva.proc.results_logging import ResultsLogger

from minerva.db.result_store import ElasticResultStorer
import minerva.db.corpora as cp

class PrecomputedPipeline(BaseTestingPipeline):
    """
        Pipeline for training weights. Queries are run once, the explanation of
        each result is stored and weights are trained.
    """

    def __init__(self, retrieval_class=BaseRetrieval):
        super(self.__class__, self).__init__(retrieval_class)
        self.writers={}

    def addResult(self, guid, precomputed_query, doc_method, retrieved):
        """
            Overrides BaseTestingPipeline.addResult so that for each retrieval result
            we actually run .explain() on each item and we store the precomputed
            formula.
        """
        retrieval_result=deepcopy(precomputed_query)
        retrieval_result["doc_method"]=doc_method

        del retrieval_result["query_text"]

        formulas=self.precomputeFormulas(precomputed_query,doc_method, retrieved)
        retrieval_result["formulas"]=formulas

##        self.checkPrecomputedRetrievalWorks(retrieval_result, doc_method, precomputed_query, {}, guid)

        for remove_key in ["dsl_query", "lucene_query"]:
            if remove_key in retrieval_result:
                del retrieval_result[remove_key]

        retrieval_result["experiment_id"]=self.exp["experiment_id"]

        self.writers["ALL"].addResult(retrieval_result)

        if self.exp.get("random_zoning",False):
            pass
        else:
            if retrieval_result.get("az","") != "":
                self.writers["az_"+retrieval_result["az"]].addResult(retrieval_result)
            if retrieval_result["csc_type"] == "":
                retrieval_result["csc_type"] = "Bac"
            self.writers["csc_type_"+retrieval_result["csc_type"]].addResult(retrieval_result)

    def loadQueriesAndFileList(self):
        """
            Loads the precomputed queries and the list of test files to process.
        """
        precomputed_queries_file_path=self.exp.get("precomputed_queries_file_path",None)
        if not precomputed_queries_file_path:
            precomputed_queries_file_path=os.path.join(self.exp["exp_dir"],self.exp.get("precomputed_queries_filename","precomputed_queries.json"))

        if "ALL" in self.exp.get("queries_to_process",["ALL"]):
            self.precomputed_queries=json.load(open(precomputed_queries_file_path,"r"))
##            precomputed_queries=json.load(open(self.exp["exp_dir"]+"precomputed_queries.json","r"))
        else:
            queries_filename="queries_by_"+self.exp["queries_classification"]+".json"
            queries_by_zone=json.load(open(self.exp["exp_dir"]+queries_filename,"r"))
            self.precomputed_queries=[]
            for zone in queries_by_zone[self.exp["queries_to_process"]]:
                self.precomputed_queries.extend(queries_by_zone[zone])

        print("Total precomputed queries: ",len(self.precomputed_queries))

        files_dict_filename=os.path.join(self.exp["exp_dir"],self.exp.get("files_dict_filename","files_dict.json"))
        self.files_dict=json.load(open(files_dict_filename,"r"))
        self.files_dict["ALL_FILES"]={}

        assert self.exp["name"] != "", "Experiment needs a name!"

        self.writers={}
        if self.exp.get("random_zoning",False):
            for div in RANDOM_ZONES_7:
                self.writers["RZ7_"+div]=ElasticResultStorer(self.exp["name"],"prr_az_rz11")
                self.writers["RZ7_"+div].clearResults()
            for div in RANDOM_ZONES_11:
                self.writers["RZ11_"+div]=ElasticResultStorer(self.exp["name"],"prr_rz11")
                self.writers["RZ11_"+div].clearResults()
        else:
            for div in AZ_ZONES_LIST:
                self.writers["az_"+div]=ElasticResultStorer(self.exp["name"],"prr_az_"+div)
                self.writers["az_"+div].clearResults()
            for div in CORESC_LIST:
                self.writers["csc_type_"+div]=ElasticResultStorer(self.exp["name"],"prr_csc_type_"+div)
                self.writers["csc_type_"+div].clearResults()

        self.writers["ALL"]=ElasticResultStorer(self.exp["name"],"prr_ALL")
        self.writers["ALL"].clearResults()

    def saveResultsAndCleanUp(self):
        """
            Executes after the retrieval is done.
        """
        for writer in self.writers:
            self.writers[writer].saveAsJSON(os.path.join(self.exp["exp_dir"],self.writers[writer].table_name+".json"))

    def checkPrecomputedRetrievalWorks(self, retrieval_result, doc_method, query, parameters, guid):
        """
            Compares the results of retrieval and those of running on the stored
            formulas to test it is all doing what it should.
        """
        test_logger=ResultsLogger(False,False)
        result_dict1={"match_guid":retrieval_result["match_guid"],"query_method":retrieval_result["query_method"],"doc_method":doc_method}
        result_dict2={"match_guid":retrieval_result["match_guid"],"query_method":retrieval_result["query_method"],"doc_method":doc_method}
        result_dict1=test_logger.measureScoreAndLog(self.runPrecomputedQuery(retrieval_result,parameters),retrieval_result["citation_multi"],result_dict1)
        retrieved=actual_tfidfmodels[doc_method].runQuery(query["query_text"],parameters,guid)
        result_dict2=test_logger.measureScoreAndLog(retrieved,retrieval_result["citation_multi"],result_dict2)
        assert(result_dict1["precision_score"]==result_dict2["precision_score"])

        parameters={x:y for y,x in enumerate(all_doc_methods[doc_method]["runtime_parameters"])}
        result_dict1={"match_guid":retrieval_result["match_guid"],"query_method":retrieval_result["query_method"],"doc_method":doc_method}
        result_dict2={"match_guid":retrieval_result["match_guid"],"query_method":retrieval_result["query_method"],"doc_method":doc_method}
        result_dict1=test_logger.measureScoreAndLog(self.runPrecomputedQuery(retrieval_result,parameters),retrieval_result["citation_multi"],result_dict1)
        retrieved=actual_tfidfmodels[doc_method].runQuery(query["query_text"],parameters,guid)
        result_dict2=test_logger.measureScoreAndLog(retrieved,retrieval_result["citation_multi"],result_dict2)
        assert(result_dict1["precision_score"]==result_dict2["precision_score"])


    def precomputeFormulas(self, query, doc_method, retrieved_results):
        """
            It runs the .formulaFromExplanation() method of BaseRetrieval
        """
        doc_list=[hit[1]["guid"] for hit in retrieved_results]

        results=[]
        for doc_id in doc_list:
            formula=self.tfidfmodels[doc_method].formulaFromExplanation(query, doc_id)
##            if formula.formula["coord"] != 0:
            # we assume that if a document was returned it must match
            results.append({"guid":doc_id,"formula":formula.formula})
        return results


def main():
    pass

if __name__ == '__main__':
    main()
