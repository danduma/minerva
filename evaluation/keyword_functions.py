# Functions decoupled from PrecomputedPipeline in order to run them in parallel
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
                     keyword_selector_class,
                     keyword_selection_parameters):
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
    else:
        raise ValueError("Unkown type of context_extraction %s" % context_extraction)

    kw_data={
            "match_guid":precomputed_query["match_guid"],
            "doc_method":doc_method,
            "context":context,
            "parent_s":cit["parent_s"],
            "cit_id":cit["id"],
            "experiment_id":experiment_id
            }

    # these are the default scores that you get by just using the whole bag of words as a query
    measureScores(doc_list, precomputed_query["match_guid"], kw_data, cit.get("multi",1))

    keyword_selector_class=getattr(keyword_selection, keyword_selector_class, None)
    assert(keyword_selector_class)

    keyword_selector=keyword_selector_class()

    selected_keywords=keyword_selector.selectKeywords(precomputed_query,
                                                       doc_list,
                                                       retrieval_model,
                                                       keyword_selection_parameters
                                                       )

    kw_data["best_kws"]=selected_keywords
    if len(kw_data["best_kws"])==0:
        addMissingFile(docfrom,precomputed_query,cit) # for debugging purposes, keep a list of missing files
        return

    all_kws={x[0]:x[1] for x in kw_data["best_kws"]}

    # need to annotate this per token?
##    for sent in docfrom.allsentences:
##        for token in sent["token_features"]:
##            if token["text"] in all_kws:
##                token["extract"]=True
##                token["weight"]=all_kws[token["text"]]
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
