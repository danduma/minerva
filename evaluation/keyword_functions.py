# Functions decoupled from PrecomputedPipeline in order to run them in parallel
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
from copy import deepcopy
import re
import os
import db.corpora as cp

from evaluation import kw_selector_classes
from evaluation.keyword_selection import getTermScores
from proc.nlp_functions import selectSentencesToAdd, CIT_MARKER
from proc.results_logging import measureScores
from evaluation.keyword_selection_measurement import runQueryAndMeasureKeywordSelection, runAndMeasureOneQuery

MISSING_FILES = []
SELECTION_EVALUATION = []


def addMissingFile(docfrom, precomputed_query, cit, missing_guid):
    # FIXME: This no longer works now it's a list
    print("Document %s not found in index" % missing_guid)
    ##    print("Explanation returned empty: no matching terms. Query:",
    ##          precomputed_query["dsl_query"], " Match_guid: ",precomputed_query["match_guid"])
    found = cp.Corpus.getMetadataByGUID(missing_guid)

    mfile = [docfrom.metadata["guid"],  # file_guid, match_guid, query, match_title, match_year, in_papers
             missing_guid,
             precomputed_query["dsl_query"]["multi_match"]["query"],
             docfrom.reference_by_id[cit["ref_id"]]["title"],
             docfrom.reference_by_id[cit["ref_id"]].get("year", "None"),
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
                     keyword_selection_parameters,
                     weights,
                     annotator
                     ):
    """
        Creates an annotated context.

        1. Extracts the citation's context, annotated with features.
        2. Runs a precomputed query using the retrieval_model, computes the formula for each result.
        3. Uses formulas to select the top keywords for that context
        4. Measures scores of selected keywords
        5. Packages it all together
    """
    cit = docfrom.citation_by_id[precomputed_query["citation_id"]]

    # for now, the context is made up of sentences. Maybe change this in future.
    if context_extraction == "sentence":
        to_add = selectSentencesToAdd(docfrom, cit, extraction_parameter)
        context = [deepcopy(docfrom.element_by_id[x]) for x in to_add]
    else:
        raise ValueError("Unkown type of context_extraction %s" % context_extraction)

    if cit.get("multi", 1) > 1:
        cit_ids = cit["group"]
    else:
        cit_ids = [cit["id"]]

    kw_data = {
        "file_guid": precomputed_query["file_guid"],
        "match_guids": precomputed_query["match_guids"],
        "doc_method": doc_method,
        "context": context,
        "parent_s": cit["parent_s"],
        "cit_ids": cit_ids,
        "cit_multi": cit.get("multi", 1),
        "experiment_id": experiment_id
    }

    # these are the default scores that you get by just using the whole bag of words as a query
    measureScores(doc_list, precomputed_query["match_guids"], kw_data)

    keyword_selector_class = getattr(kw_selector_classes, keyword_selector_class, None)
    assert keyword_selector_class

    keyword_selector = keyword_selector_class()

    selected_keywords = keyword_selector.selectKeywords(precomputed_query,
                                                        doc_list,
                                                        retrieval_model,
                                                        keyword_selection_parameters,
                                                        cit,
                                                        weights
                                                        )

    kw_data["best_kws"] = selected_keywords
    if len(kw_data["best_kws"]) == 0:
        addMissingFile(docfrom, precomputed_query, cit, precomputed_query["match_guids"])  # for debugging purposes, keep a list of missing files
        return

    all_kws = {x[0]: x[1] for x in kw_data["best_kws"]}

    runQueryAndMeasureKeywordSelection(precomputed_query,
                                       selected_keywords,
                                       retrieval_model,
                                       weights,
                                       kw_data)

    cit_num = int(re.sub("cit", "", cit["id"], flags=re.IGNORECASE))

    # annotate all token features for this context
    context = annotator.annotate_context(context, cit, docfrom)

    # need to annotate this per token
    for sent in context:
        for token in sent.get("token_features", []):
            if token["text"] in all_kws:
                token["extract"] = True
                token["weight"] = all_kws[token["text"]]

    writers["ALL"].addResult(kw_data)


def runVariousKeywordSelections(precomputed_query,
                                docfrom,
                                doc_method,
                                doc_list,
                                retrieval_model,
                                writers,
                                experiment_id,
                                context_extraction,
                                extraction_parameter,
                                keyword_selection_parameters,
                                weights,
                                annotator
                                ):
    """
        Like annotateKeywords() but it tests different KW selection classes with
        different values
    """
    cit = docfrom.citation_by_id[precomputed_query["citation_id"]]

    # for now, the context is made up of sentences. Maybe change this in future.
    if context_extraction == "sentence":
        to_add = selectSentencesToAdd(docfrom, cit, extraction_parameter)
        context = [deepcopy(docfrom.element_by_id[x]) for x in to_add]
    else:
        raise ValueError("Unknown type of context_extraction %s" % context_extraction)

    if cit.get("multi", 1) > 1:
        cit_ids = cit["group"]
    else:
        cit_ids = [cit["id"]]

    kw_data = {
        "file_guid": precomputed_query["file_guid"],
        "match_guids": precomputed_query["match_guids"],
        "doc_method": doc_method,
        "context": context,
        "parent_s": cit["parent_s"],
        "cit_ids": cit_ids,
        "cit_multi": cit.get("multi", 1),
        "experiment_id": experiment_id
    }

    # these are the default scores that you get by just using the whole bag of words as a query
    measureScores(doc_list, precomputed_query["match_guids"], kw_data)
    term_scores = getTermScores(precomputed_query, doc_list, retrieval_model)

    for entry in keyword_selection_parameters:
        class_name = keyword_selection_parameters[entry]["class"]
        params = keyword_selection_parameters[entry]["parameters"]

        keyword_selector_class = getattr(kw_selector_classes, class_name, None)
        assert keyword_selector_class

        kw_data_copy = deepcopy(kw_data)
        kw_data_copy["keyword_selection_entry"] = entry
        context_text = " ".join([s.get("text") for s in kw_data_copy["context"]])
        del kw_data_copy["context"]
        kw_data_copy["context"] = context_text

        keyword_selector = keyword_selector_class()
        selected_keywords = keyword_selector.selectKeywords(precomputed_query,
                                                            doc_list,
                                                            retrieval_model,
                                                            params,
                                                            cit,
                                                            weights,
                                                            term_scores=term_scores
                                                            )

        kw_data_copy["best_kws"] = selected_keywords

        if len(kw_data_copy["best_kws"]) == 0:
            addMissingFile(docfrom, precomputed_query, cit, precomputed_query["match_guids"])  # for debugging purposes, keep a list of missing files
            return

        runQueryAndMeasureKeywordSelection(precomputed_query,
                                           selected_keywords,
                                           retrieval_model,
                                           weights,
                                           kw_data_copy)

        writers["ALL"].addResult(kw_data_copy)


def multiAnnotateKeywords(precomputed_query,
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
                          annotator):
    """
    Get best set of keywords for a set of citations

    :param precomputed_query:
    :param docfrom:
    :param doc_method:
    :param doc_list:
    :param retrieval_model:
    :param writers:
    :param experiment_id:
    :param context_extraction:
    :param extraction_parameter:
    :param keyword_selector_class:
    :param keyword_selection_parameters:
    :param weights:
    :param annotator:
    :return:
    """
    pass


def precomputeFormulas(retrieval_model, query, doc_list):
    """
        It runs the .formulaFromExplanation() method of BaseRetrieval
    """
    results = []
    ##    print("Computing explain formulas...")
    ##    progress=ProgressIndicator(True, numitems=len(doc_list), print_out=False)
    for doc_id in doc_list:
        formula = retrieval_model.formulaFromExplanation(query, doc_id)
        # we assume that if a document was returned it must match
        results.append({"guid": doc_id, "formula": formula.formula})
    ##        progress.showProgressReport("Computing explain formulas -- %s" % doc_id)
    return results


def main():
    pass


if __name__ == '__main__':
    main()
