# Functions decoupled from PrecomputedPipeline in order to run them
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from copy import deepcopy

import minerva.db.corpora as cp
import keyword_selection

def annotateKeywords(precomputed_query, doc_method, doc_list, retrieval_model, writers, experiment_id, selection_method="selectKeywordsNBest"):
    """
        Runs a precomputed query using the retrieval_model, computes the formula
        for each result

        Uses formulas to
    """
    # !TODO remove this, it's a temporary fix
##    if precomputed_query.get("csc_type","Bac") in ["Bac"]:
##        print("Ignoring query of type %s" % precomputed_query.get("csc_type",""))
##        return

    context={}

    kw_data={
            "match_guid":precomputed_query["match_guid"],
            "doc_method":doc_method,

            }

    formulas=precomputeFormulas(retrieval_model, precomputed_query, doc_list)

    kw_data["formulas"]=formulas

    func=getattr(keyword_selection, selection_method, None)
    assert(func)

    best_kws=func(formulas)

    for remove_key in ["dsl_query", "lucene_query"]:
        if remove_key in kw_data:
            del kw_data[remove_key]

    kw_data["experiment_id"]=experiment_id

    writers["ALL"].addResult(kw_data)


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
