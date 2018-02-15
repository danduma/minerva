# Functions decoupled from PrecomputedPipeline in order to run them in parallel
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
from copy import deepcopy
import json, re
import db.corpora as cp

from . import keyword_selection
from proc.nlp_functions import selectSentencesToAdd
from proc.results_logging import measureScores
from proc.structured_query import StructuredQuery

MISSING_FILES=[]
SELECTION_EVALUATION=[]

def addMissingFile(docfrom,precomputed_query,cit):
    print("Document %s not found in index" % precomputed_query["match_guid"])
##    print("Explanation returned empty: no matching terms. Query:",
##          precomputed_query["dsl_query"], " Match_guid: ",precomputed_query["match_guid"])
    found=cp.Corpus.getMetadataByGUID(precomputed_query["match_guid"])

    mfile=[docfrom.metadata["guid"],
           precomputed_query["match_guid"],
           precomputed_query["dsl_query"]["multi_match"]["query"],
           docfrom.reference_by_id[cit["ref_id"]]["title"],
           docfrom.reference_by_id[cit["ref_id"]].get("year","None"),
           (found is not None)
           ]
    MISSING_FILES.append(mfile)

def measureOneQuery(query, weights, retrieval_model, cit):
    """
        Returns a dict of scores given a precomputed_query, weights, the
        retrieval_model in use and the cit dict

    """
    retrieved_results=retrieval_model.runQuery(query, weights)
    new_scores={} # need to pass a dict to measureScores(), results will be stored there
    doc_list=doc_list=[hit[1]["guid"] for hit in retrieved_results]
    measureScores(doc_list, query["match_guid"], new_scores, cit.get("multi",1))
    return new_scores

def measureKeywordSelection(precomputed_query, selected_keywords, retrieval_model, cit, weights, kw_data):
    """
        Computes scores for the provided keywords, adds them to the [kw_data] dict

        :param precomputed_query: original precomputed_query as provided by testing pipeline
        :param selected_keywords: tuples of (keyword,weight) coming out of exp["keyword_selector"].selectKeywords()
        :param retrieval_model: retrieval instance (e.g. ElasticRetrievalBoost)
        :param cit: citation dict
        :param weights: the weights used for retrieval at this point, configurable in the experiment
        :param kw_data: dict that will eventually be stored, containing the precomputed_query, selected kws, etc.
    """
    kw_counts={x[0]:x[1] for x in precomputed_query["structured_query"]}

    query=deepcopy(precomputed_query)
    query["structured_query"]=StructuredQuery([[x[0],kw_counts.get(x[0],0),1] for x in selected_keywords])
    kw_data["kw_selection_scores"]=measureOneQuery(query, weights, retrieval_model, cit)

    query=deepcopy(precomputed_query)
    query["structured_query"]=StructuredQuery([[x[0],kw_counts.get(x[0],0),x[1]] for x in selected_keywords])
    kw_data["kw_selection_weight_scores"]=measureOneQuery(query, weights, retrieval_model, cit)

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
                     keyword_selection_parameters,
                     weights,
                     ):
    """
        Creates an annotated context.

        1. Extracts the citation's context, annotated with features.
        2. Runs a precomputed query using the retrieval_model, computes the formula for each result.
        3. Uses formulas to select the top keywords for that context
        4. Measures scores of selected keywords
        5. Packages it all together
    """
    cit=docfrom.citation_by_id[precomputed_query["citation_id"]]

    # for now, the context is made up of sentences. Maybe change this in future.
    if context_extraction=="sentence":
        to_add=selectSentencesToAdd(docfrom, cit, extraction_parameter)
        context=[deepcopy(docfrom.element_by_id[x]) for x in to_add]
    else:
        raise ValueError("Unkown type of context_extraction %s" % context_extraction)

    kw_data={
            "file_guid":precomputed_query["file_guid"],
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

    measureKeywordSelection(precomputed_query,
                            selected_keywords,
                            retrieval_model,
                            cit,
                            weights,
                            kw_data)

    cit_num=int(re.sub("cit","",cit["id"], flags=re.IGNORECASE))

    # need to annotate this per token
    for sent in context:
        if sent.get("token_features",None) is None:
            print("Sentence does not have token_features annotated ",sent)
        for token in sent.get("token_features",[]):
            if token["text"] in all_kws:
                token["extract"]=True
                token["weight"]=all_kws[token["text"]]

            # for each token, we only want to keep the distance to its relevant citation
            for key in token:
                if key.startswith("dist_cit_"):
                    if key == "dist_cit_" + str(cit_num):
                        token["dist_cit"] = token[key]
                    del token[key]


##    print("Best KWs", kw_data["best_kws"])
##    print("original_scores: { \n ndcg_score:",kw_data["ndcg_score"], "\n precision_score:", kw_data["precision_score"], "\n rank:",  kw_data["rank"], "\n mrr_score:",  kw_data["mrr_score"], "\n}")
##    print("kw_selection_scores",json.dumps(kw_data["kw_selection_scores"], indent=2))
##    print("kw_selection_weight_scores",json.dumps(kw_data["kw_selection_weight_scores"], indent=2))
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
