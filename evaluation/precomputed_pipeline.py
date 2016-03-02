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

class PrecomputedPipeline(BaseTestingPipeline):
    """
        Pipeline for training weights. Queries are run once, the explanation of
        each result is stored and weights are trained.
    """

    def __init__(self, retrieval_class=BaseRetrieval):
        super(self.__class__, self).__init__(retrieval_class)

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

        # TODO make sure what is being saved is what should be
        # TODO use better logging/saving, this writing JSON to file is ridiculous
        for remove_key in ["dsl_query", "lucene_query"]:
            if remove_key in retrieval_result:
                del retrieval_result[remove_key]

        retrieval_result["experiment_id"]=self.exp["experiment_id"]

        out_str=json.dumps(retrieval_result)+","
        self.files["ALL"].write(out_str)
        if self.exp.get("random_zoning",False):
            pass
        else:
            if retrieval_result.get("az","") != "":
                self.files["AZ_"+retrieval_result["az"]].write(out_str)
            if retrieval_result["csc_type"] == "":
                retrieval_result["csc_type"] = "Bac"
            self.files["CSC_"+retrieval_result["csc_type"]].write(out_str)

    def saveResultsAndCleanUp(self):
        """
            Executes after the retrieval is done.
        """
        if self.exp.get("random_zoning",False):
            for div in RANDOM_ZONES_7:
                self.files["RZ7_"+div].seek(-1,os.SEEK_END)
                self.files["RZ7_"+div].write("]")

            for div in RANDOM_ZONES_11:
                self.files["RZ11_"+div].seek(-1,os.SEEK_END)
                self.files["RZ1_"+div].write("]")
        else:
            for div in AZ_ZONES_LIST:
                self.files["AZ_"+div].seek(-1,os.SEEK_END)
                self.files["AZ_"+div].write("]")

            for div in CORESC_LIST:
                self.files["CSC_"+div].seek(-1,os.SEEK_END)
                self.files["CSC_"+div].write("]")

        self.files["ALL"].seek(-1,os.SEEK_END)
        self.files["ALL"].write("]")


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
            queries_by_az=json.load(open(self.exp["exp_dir"]+queries_filename,"r"))
            self.precomputed_queries=queries_by_az[self.exp["queries_to_process"]]

        files_dict_filename=os.path.join(self.exp["exp_dir"],self.exp.get("files_dict_filename","files_dict.json"))
        self.files_dict=json.load(open(files_dict_filename,"r"))

        self.files={}

        if self.exp.get("random_zoning",False):
            for div in RANDOM_ZONES_7:
                self.files["RZ7_"+div]=open(self.exp["exp_dir"]+"prr_RZ7_"+div+".json","w")
                self.files["RZ7_"+div].write("[")
            for div in RANDOM_ZONES_11:
                self.files["RZ11_"+div]=open(self.exp["exp_dir"]+"prr_RZ11_"+div+".json","w")
                self.files["RZ11_"+div].write("[")
        else:
            for div in AZ_ZONES_LIST:
                self.files["AZ_"+div]=open(self.exp["exp_dir"]+"prr_AZ_"+div+".json","w")
                self.files["AZ_"+div].write("[")

            for div in CORESC_LIST:
                self.files["CSC_"+div]=open(self.exp["exp_dir"]+"prr_CSC_"+div+".json","w")
                self.files["CSC_"+div].write("[")

        self.files["ALL"]=open(self.exp["exp_dir"]+"prr_ALL.json","w")
        self.files["ALL"].write("[")


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

    def runPrecomputedQuery(self, retrieval_result, parameters):
        """
            This takes a query that has already had the results added
        """
        scores=[]
        for unique_result in retrieval_result:
            formula=StoredFormula(unique_result["formula"])
            score=formula.computeScore(formula.formula, parameters)
            scores.append((score,{"guid":unique_result["guid"]}))

        scores.sort(key=lambda x:x[0],reverse=True)
        return scores


    def measurePrecomputedResolution(retrieval_results,method,parameters, citation_az="*"):
        """
            This is kind of like measureCitationResolution:
            it takes a list of precomputed retrieval_results, then applies the new
            parameters to them. This is how we recompute what Lucene gives us,
            avoiding having to call Lucene again.

            All we need to do is adjust the weights on the already available
            explanation formulas.
        """
        logger=ResultsLogger(False, dump_straight_to_disk=False) # init all the logging/counting
        logger.startCounting() # for timing the process, start now

        logger.setNumItems(len(retrieval_results),print_out=False)

        # for each query-result: (results are packed inside each query for each method)
        for result in retrieval_results:
            # select only the method we're testing for
            formulas=result["formulas"]
            retrieved=self.runPrecomputedQuery(formulas,parameters)

            result_dict={"file_guid":result["file_guid"],
                         "citation_id":result["citation_id"],
                         "doc_position":result["doc_position"],
                         "query_method":result["query_method"],
                         "doc_method":method,
                         "az":result["az"],
                         "cfc":result["cfc"],
                         "match_guid":result["match_guid"]}

            if not retrieved or len(retrieved)==0:    # the query was empty or something
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
    ##            weights=all_doc_methods[doc_method]["runtime_parameters"]
                weights=parameters
                data_line={"query_method":query_method,"doc_method":doc_method,"citation_az":citation_az}

                for metric in logger.averages[query_method][doc_method]:
                    data_line["avg_"+metric]=logger.averages[query_method][doc_method][metric]
                data_line["precision_total"]=logger.scores["precision"][query_method][doc_method]

##                signature=""
##                for w in weights:
##                    data_line[w]=weights[w]
##                    signature+=str(w)
    ##            data_line["weight_signature"]=signature
                results.append(data_line)

    ##    logger.writeDataToCSV(cp.cp.Corpus.dir_output+"testing_test_precision.csv")

        return results


def main():
    pass

if __name__ == '__main__':
    main()
