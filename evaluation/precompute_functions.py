# Functions decoupled from PrecomputedPipeline in order to run them
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from copy import deepcopy

import minerva.db.corpora as cp
from minerva.db.result_store import ElasticResultStorer
from minerva.proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11

def createWriters(exp_name, exp_random_zoning=False, clear_existing_prr_results=False):
    """
        Returns a dict with instances of ElasticResultStorer
    """
    writers={}
    if exp_random_zoning:
        for div in RANDOM_ZONES_7:
            writers["RZ7_"+div]=ElasticResultStorer(exp_name,"prr_az_rz11", endpoint=cp.Corpus.endpoint)
            if clear_existing_prr_results:
                writers["RZ7_"+div].clearResults()
        for div in RANDOM_ZONES_11:
            writers["RZ11_"+div]=ElasticResultStorer(exp_name,"prr_rz11", endpoint=cp.Corpus.endpoint)
            if clear_existing_prr_results:
                writers["RZ11_"+div].clearResults()
    else:
        for div in AZ_ZONES_LIST:
            writers["az_"+div]=ElasticResultStorer(exp_name,"prr_az_"+div, endpoint=cp.Corpus.endpoint)
            if clear_existing_prr_results:
                writers["az_"+div].clearResults()
        for div in CORESC_LIST:
            writers["csc_type_"+div]=ElasticResultStorer(exp_name,"prr_csc_type_"+div, endpoint=cp.Corpus.endpoint)
            if clear_existing_prr_results:
                writers["csc_type_"+div].clearResults()

    writers["ALL"]=ElasticResultStorer(exp_name,"prr_ALL", endpoint=cp.Corpus.endpoint)
    if clear_existing_prr_results:
        writers["ALL"].clearResults()

    return writers

def addPrecomputeExplainFormulas(precomputed_query, doc_method, doc_list, retrieval_model, writers, experiment_id, exp_random_zoning=False):
    """
        Runs a precomputed query using the retrieval_model, computes the formula
        for each result
    """
    retrieval_result=deepcopy(precomputed_query)
    retrieval_result["doc_method"]=doc_method

    del retrieval_result["query_text"]

    formulas=precomputeFormulas(retrieval_model, precomputed_query, doc_list)
    retrieval_result["formulas"]=formulas

    for remove_key in ["dsl_query", "lucene_query"]:
        if remove_key in retrieval_result:
            del retrieval_result[remove_key]

    retrieval_result["experiment_id"]=experiment_id

    writers["ALL"].addResult(retrieval_result)

    if exp_random_zoning:
        pass
    else:
        assert retrieval_result["csc_type"] != "", "No csc_type!"
        if retrieval_result.get("az","") != "":
            writers["az_"+retrieval_result["az"]].addResult(retrieval_result)
        if retrieval_result["csc_type"] == "":
            retrieval_result["csc_type"] = "Bac"
        writers["csc_type_"+retrieval_result["csc_type"]].addResult(retrieval_result)


def precomputeFormulas(retrieval_model, query, doc_list):
    """
        It runs the .formulaFromExplanation() method of BaseRetrieval
    """
    results=[]
##    print("Computing explain formulas...")
##    progress=ProgressIndicator(True, numitems=len(doc_list), print_out=False)
    for doc_id in doc_list:
        formula=retrieval_model.formulaFromExplanation(query, doc_id)
        # we assume that if a document was returned it must match
        results.append({"guid":doc_id,"formula":formula.formula})
##        progress.showProgressReport("Computing explain formulas -- %s" % doc_id)
    return results


def main():
    pass

if __name__ == '__main__':
    main()
