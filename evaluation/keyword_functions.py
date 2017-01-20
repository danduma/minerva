# Functions decoupled from PrecomputedPipeline in order to run them
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function
##from copy import deepcopy
import minerva.db.corpora as cp

import keyword_selection
from minerva.proc.nlp_functions import selectSentencesToAdd
from minerva.proc.results_logging import measureScores

MISSING_FILES=[]

def addMissingFile(docfrom,precomputed_query,cit):
    print("Explanation returned empty: no matching terms. Query:",
          precomputed_query["dsl_query"], " Match_guid: ",precomputed_query["match_guid"])
    found=cp.Corpus.getMetadataByGUID(precomputed_query["match_guid"])

    mfile=[docfrom.metadata["guid"],
           precomputed_query["match_guid"],
           precomputed_query["dsl_query"]["multi_match"]["query"],
           docfrom.reference_by_id[cit["ref_id"]]["title"],
           docfrom.reference_by_id[cit["ref_id"]]["year"],
           (found is not None)
           ]
    MISSING_FILES.append(mfile)

def annotateKeywords(precomputed_query,
                     docfrom,
                     doc_method,
                     doc_list,
                     retrieval_model,
                     writers,
                     experiment_id,
                     context_extraction,
                     extraction_parameter,
                     keyword_selection_method):
    """
        Creates an annotated context.

        1. Extracts the citation's context, annotated with features.
        2. Runs a precomputed query using the retrieval_model, computes the formula for each result.
        3. Uses formulas to choose the top keywords for that context
        4. Packages it all together
    """
    # !TODO remove this, it's a temporary fix
##    if precomputed_query.get("csc_type","Bac") in ["Bac"]:
##        print("Ignoring query of type %s" % precomputed_query.get("csc_type",""))
##        return

    cit=docfrom.citation_by_id[precomputed_query["citation_id"]]

    # for now, the context is made up of sentences. Maybe change this in future.
    if context_extraction=="sentence":
        to_add=selectSentencesToAdd(docfrom, cit, extraction_parameter)
        context=[docfrom.element_by_id[x] for x in to_add]

    kw_data={
            "match_guid":precomputed_query["match_guid"],
            "doc_method":doc_method,
            "context":context,
            "parent_s":cit["parent_s"],
            "cit_id":cit["id"],
            "experiment_id":experiment_id
            }

    measureScores(doc_list, precomputed_query["match_guid"], kw_data, cit.get("multi",1))

    func=getattr(keyword_selection, keyword_selection_method, None)
    assert(func)

    kw_data["best_kws"]=func(precomputed_query, doc_list, retrieval_model)
    if len(kw_data["best_kws"])==0:
        addMissingFile(docfrom,precomputed_query,cit)
        return

    all_kws={x[0]:x[1] for x in kw_data["best_kws"]}

    for sent in docfrom.allsentences:
        for token in sent["token_features"]:
            if token["text"] in all_kws:
                token["extract"]=True
                token["weight"]=all_kws[token["text"]]
##    print(kw_data["best_kws"])
##    print("\n")

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
