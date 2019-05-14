# Functions decoupled from PrecomputedPipeline in order to run them in parallel
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import

import json
from copy import deepcopy
from six import string_types
import db.corpora as cp

from evaluation import kw_selector_classes
from evaluation.keyword_annotation import getNormalisedTermScores, getDictOfDocFreq, getCountsInQueryForMatchingTerms
from evaluation.kw_selector_classes import makeStructuredQueryAgain
from proc.nlp_functions import selectSentencesToAdd, CIT_MARKER
from proc.results_logging import measureScores
from evaluation.keyword_annotation_measurement import runQueryAndMeasureKeywordSelection

MISSING_FILES = []
SELECTION_EVALUATION = []


def addMissingFile(docfrom, precomputed_query, cit, missing_guid):
    # FIXME: This no longer works now it's a list
    print("Document %s not found in index" % missing_guid)
    ##    print("Explanation returned empty: no matching terms. Query:",
    ##          precomputed_query["dsl_query"], " Match_guid: ",precomputed_query["match_guid"])
    try:
        found = cp.Corpus.getMetadataByGUID(missing_guid)
    except:
        found = None

    mfile = [docfrom.metadata["guid"],  # file_guid, match_guid, query, match_title, match_year, in_papers
             missing_guid,
             precomputed_query["dsl_query"]["multi_match"]["query"],
             docfrom.reference_by_id[cit["ref_id"]]["title"],
             docfrom.reference_by_id[cit["ref_id"]].get("year", "None"),
             (found is not None)
             ]
    MISSING_FILES.append(mfile)


def annotateContext(sentences, annotator, cit, docfrom, all_kws):
    # annotate all token features for this context
    sentences = annotator.annotate_context(sentences, cit, docfrom)

    if not sentences:
        return

    # need to annotate this per token
    for sent in sentences:
        if "pos_tagged" in sent:
            del sent["pos_tagged"]
        for token in sent.get("token_features", []):
            if token["text"] in all_kws:
                token["extract"] = True
                token["weight"] = all_kws[token["text"]]


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
    LOG_MISSING_FILES = False

    cit = docfrom.citation_by_id[precomputed_query["citation_id"]]

    # for now, the context is made up of sentences. Maybe change this in future.
    if context_extraction == "sentence":
        to_add = selectSentencesToAdd(docfrom, cit, extraction_parameter)
        sentences = [deepcopy(docfrom.element_by_id[x]) for x in to_add]
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
        "context": sentences,
        "parent_s": cit["parent_s"],
        "cit_ids": cit_ids,
        "cit_multi": cit.get("multi", 1),
        "experiment_id": experiment_id
    }

    # these are the default scores that you get by just using the whole bag of words as a query
    measureScores(doc_list, precomputed_query["match_guids"], kw_data)

    if isinstance(keyword_selector_class, string_types):
        keyword_selector_class = getattr(kw_selector_classes, keyword_selector_class, None)
    assert keyword_selector_class

    keyword_selector = keyword_selector_class("")

    norm_term_scores, formulas, match_formulas = getNormalisedTermScores(precomputed_query,
                                                                         doc_list,
                                                                         retrieval_model)
    docFreq, maxDocs = getDictOfDocFreq(formulas)

    selected_keywords = keyword_selector.selectKeywords(precomputed_query,
                                                        doc_list,
                                                        retrieval_model,
                                                        keyword_selection_parameters,
                                                        cit,
                                                        weights,
                                                        norm_term_scores,
                                                        docFreq
                                                        )

    kw_data["best_kws"] = selected_keywords
    if len(kw_data["best_kws"]) == 0 and LOG_MISSING_FILES:
        addMissingFile(docfrom, precomputed_query, cit,
                       precomputed_query["match_guids"])  # for debugging purposes, keep a list of missing files
        return

    all_kws = {x[0]: x[1] for x in kw_data["best_kws"]}

    runQueryAndMeasureKeywordSelection(precomputed_query,
                                       selected_keywords,
                                       retrieval_model,
                                       weights,
                                       kw_data)

    annotateContext(sentences, annotator, cit, docfrom, all_kws)
    writers["ALL"].addResult(kw_data)


ALL_CONTEXTS = []
SPECIAL_EXPORT = False


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
    # if len(precomputed_query["match_guids"]) < 2:
    # print("Query with only one match_guid")
    # return

    cit = docfrom.citation_by_id[precomputed_query["citation_id"]]

    # for now, the context is made up of sentences. Maybe change this in future.
    if context_extraction == "sentence":
        to_add = selectSentencesToAdd(docfrom, cit, extraction_parameter)
        sentences = [deepcopy(docfrom.element_by_id[x]) for x in to_add]
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
        "context": sentences,
        "parent_s": cit["parent_s"],
        "cit_ids": cit_ids,
        "cit_multi": cit.get("multi", 1),
        "experiment_id": experiment_id
    }

    # these are the default scores that you get by just using the whole bag of words as a query
    # print("Query:", precomputed_query["query_text"])

    # makeStructuredQueryAgain(precomputed_query)

    # if precomputed_query["file_guid"] == 'b884c939-144c-4d95-9d30-097f8b83e1d3' and precomputed_query[
    #     "citation_id"] == 'cit9':
    #     print("aha")

    measureScores(doc_list, precomputed_query["match_guids"], kw_data)
    norm_term_scores, formulas, match_formulas = getNormalisedTermScores(precomputed_query,
                                                                         doc_list,
                                                                         retrieval_model)
    docFreq, maxDocs = getDictOfDocFreq(formulas)

    raw_data = {
        "formulas": formulas,
        "match_formulas": match_formulas
    }

    for entry in keyword_selection_parameters:
        print("[%s]" % entry)
        keyword_selector_class = keyword_selection_parameters[entry]["class"]
        params = keyword_selection_parameters[entry]["parameters"]

        if keyword_selection_parameters[entry].get("instance"):
            keyword_selector = keyword_selection_parameters[entry]["instance"]
        else:
            if isinstance(keyword_selector_class, string_types):
                keyword_selector_class = getattr(kw_selector_classes, keyword_selector_class, None)
            keyword_selector = keyword_selector_class()

        assert keyword_selector_class

        kw_data_copy = deepcopy(kw_data)
        kw_data_copy["keyword_selection_entry"] = entry
        context_text = " ".join([s.get("text") for s in kw_data_copy["context"]])
        del kw_data_copy["context"]
        kw_data_copy["context"] = context_text

        # if "24d28074-a14b-420f-baa4-c064282f59da" in precomputed_query["match_guids"]:
        #     print("aa")

        selected_keywords = keyword_selector.selectKeywords(precomputed_query,
                                                            doc_list,
                                                            retrieval_model,
                                                            params,
                                                            cit,
                                                            weights,
                                                            norm_term_scores=norm_term_scores,
                                                            docFreq=docFreq,
                                                            maxDocs=maxDocs,
                                                            rawScores=raw_data
                                                            )

        kw_data_copy["best_kws"] = selected_keywords

        # print("Query:", precomputed_query["vis_text"], "\n\n")
        # print("Chosen terms:", selected_keywords, "\n\n")

        if kw_data_copy["best_kws"] is None:
            # addMissingFile(docfrom, precomputed_query, cit,
            #                precomputed_query["match_guids"])  # for debugging purposes, keep a list of missing files
            print("Missing file(s) ", precomputed_query["match_guids"])
            continue

        runQueryAndMeasureKeywordSelection(precomputed_query,
                                           selected_keywords,
                                           retrieval_model,
                                           weights,
                                           kw_data_copy)

        print("--- Before --- ", int(
            sum([x if x != -1 else 200 for x in kw_data["per_guid_rank"].values()]) / len(kw_data["per_guid_rank"])))
        for guid in kw_data["per_guid_rank"]:
            print(guid, kw_data["per_guid_rank"][guid])
        print("--- After (w/weight) --- ", int(
            sum([x if x != -1 else 200 for x in
                 kw_data_copy["kw_selection_weight_scores"]["per_guid_rank"].values()]) / len(
                kw_data["per_guid_rank"])))
        for guid in kw_data["per_guid_rank"]:
            print(guid, kw_data_copy["kw_selection_weight_scores"]["per_guid_rank"][guid])
        print("--- After (plain) --- ",
              int(sum(
                  [x if x != -1 else 200 for x in kw_data_copy["kw_selection_scores"]["per_guid_rank"].values()]) / len(
                  kw_data["per_guid_rank"])))
        for guid in kw_data["per_guid_rank"]:
            print(guid, kw_data_copy["kw_selection_scores"]["per_guid_rank"][guid])
        print()

        # experiment: just for thesis
        if SPECIAL_EXPORT:
            all_kws = {x[0]: x[1] for x in kw_data_copy["best_kws"]}
            kw_data_copy["context"] = sentences
            annotateContext(sentences, annotator, cit, docfrom, all_kws)
            print(json.dumps(kw_data_copy))
            ALL_CONTEXTS.append(kw_data_copy)

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
