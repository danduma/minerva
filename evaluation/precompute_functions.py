# Functions decoupled from PrecomputedPipeline in order to run them
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from copy import deepcopy

import minerva.db.corpora as cp

def addPrecomputeExplainFormulas(precomputed_query, doc_method, doc_list, retrieval_model, writers, experiment_id, exp_random_zoning=False):
    """
        Runs a precomputed query using the retrieval_model, computes the formula
        for each result
    """
    # !TODO remove this, it's a temporary fix
    if precomputed_query.get("csc_type","Bac") in ["Bac"]:
        print("Ignoring query of type %s" % precomputed_query.get("csc_type",""))
        return

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
