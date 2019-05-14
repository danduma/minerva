# Base class and functions to annotate the best keywords from each context according to their retrieval scores
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function
from __future__ import absolute_import

import math
import re

TERM_POSITION_IN_TUPLE = 6


def termScoresInFormula(part):
    """
        Returns list of all term matching elements in formula

        :param part: tuple, list or dict
        :returns: list of all term matching elements in formula
    """
    if isinstance(part, tuple) or isinstance(part, list):
        return part
    elif isinstance(part, dict):
        if "type" not in part:
            return None
        if part["type"] in ["*", "+", "max"]:
            scores = [termScoresInFormula(sub_part) for sub_part in part["parts"]]
            result = []
            for score in scores:
                if isinstance(score, list):
                    result.extend(score)
                else:
                    result.append(score)
            return result
        elif part["type"] in ["const", "coord"]:
            pass

    return []


def getDictOfTermScores(formula, op="max"):
    """
        Returns the score of each term in the formula as a dict

        :param formula: formula dict
        :rtype: dict
    """
    term_scores = termScoresInFormula(formula)
    res = {}
    if not term_scores:
        return res

    for score in term_scores:
        # score=(field,qw,fw,tf,docFreq,maxDocs,term)
        old_score = res.get(score[TERM_POSITION_IN_TUPLE], 0)
        new_score = (score[1] * score[2])
        if op == "add":
            res[score[TERM_POSITION_IN_TUPLE]] = old_score + new_score
        elif op == "max":
            res[score[TERM_POSITION_IN_TUPLE]] = max(old_score, new_score)

    ##    print(res)
    return res


def getDictOfDocFreq(formulas):
    """
    Returns a dict where the key is the term and the value its docFreq over the collection, and also an integer which
    is the maxDocs

    :param formulas:
    :return:
    """
    res = {}
    maxDocs = 0
    for formula in formulas:
        term_scores = termScoresInFormula(formula.formula)
        if not term_scores:
            continue

        for score in term_scores:
            term = score[TERM_POSITION_IN_TUPLE]
            if term not in res:
                res[term] = score[4]
            if not maxDocs:
                maxDocs = score[5]

            assert maxDocs == score[5]

        ##    print(res)
    return res, maxDocs


# def getFormulaTermWeights(unique_result):
#     """
#         Computes a score for each matching keyword in the formula for the
#         matching files in the index
#
#         unique_result is a dict
#         {"match_guid":"<guid>","formulas":[{"guid":"<guid>","formula":<formula>}]
#
#         :rtype: dict
#     """
#     idf_scores = defaultdict(lambda: 0)
#     max_scores = defaultdict(lambda: 0)
#
#     formula_term_scores = []
#     match_result = None
#
#     for formula in unique_result["formulas"]:
#         term_scores = getDictOfTermScores(formula["formula"], "max")
#
#         formula_term_scores.append((formula, term_scores))
#         if formula["guid"] == unique_result["match_guid"]:
#             match_result = term_scores
#
#         for term in term_scores:
#             idf_scores[term] = idf_scores[term] + term_scores[term]
#             if term_scores[term] > max_scores[term]:
#                 max_scores[term] = term_scores[term]
#
#     if not match_result:
#         return None
#
#     for term in idf_scores:
#         idf_scores[term] = log((max_scores[term] * len(unique_result["formulas"])) / (1 + idf_scores[term]), 2)
#
#     for term in match_result:
#         match_result[term] = match_result[term] * idf_scores[term]
#
#     return match_result


# def makeStructuredQueryFromKeywords(keywords):
#     """
#         This is just to get around my former use of this query storage format
#     """
#     query = StructuredQuery()
#     for kw in keywords:
#         query.addToken(kw[0], 1, boost=kw[1])
#     return query


# def evaluateKeywordSelection(precomputed_queries, extracted_keywords, exp, use_keywords=True, metric="mrr",
#                              index_field="text"):
#     """
#         Get the batch scores of a set of queries
#
#         :param precomputed_queries: ditto
#         :param extracted_keywords: a list of lists of tuples, one list for each query
#     """
#     from proc.results_logging import measureScores
#     from retrieval.elastic_retrieval import ElasticRetrieval
#
#     retrieval_model = ElasticRetrieval(exp["features_index_name"], "", es_instance=cp.Corpus.es)
#
#     scores_list = []
#
#     for index, precomputed_query in precomputed_queries:
#         scores = {}
#         if use_keywords:
#             query = makeStructuredQueryFromKeywords(extracted_keywords[index])
#         else:
#             query = precomputed_query
#
#         retrieved = retrieval_model.runQuery(query, max_results=exp.get("max_results_recall", 200))
#         measureScores(retrieved, precomputed_query["match_guid"], scores)
#         scores_list.append(scores[metric])
#
#     return sum(scores_list) / float(len(scores_list))


def listOfTermValuesInFormulas(formulas):
    """
        Returns a dict where {term: [list of values]} in all formulas
    """
    term_stats = {}
    for formula in formulas:
        term_scores = getDictOfTermScores(formula.formula)
        for term in term_scores:
            if term not in term_stats:
                term_stats[term] = []

            term_stats[term].append(term_scores[term])

    return term_stats


def getIDFfromFormulas(formulas):
    """
        Returns a dict where {term: [list of values]} in all formulas
    """
    doc_counts = {}
    for formula in formulas:
        term_scores = getDictOfTermScores(formula.formula)
        for term in term_scores:
            if term not in doc_counts:
                doc_counts[term] = 0
            doc_counts[term] += 1
    return doc_counts


def getNormalisedTermScores(precomputed_query, doc_list, retrieval_model):
    """
    Returns the NORMALISED term scores from the explain query for each match_guid document

    :param precomputed_query: dict with the query keywords
    :param doc_list: top N retrieved documents (200 by default)
    :param retrieval_model:
    :return: dict of dicts of term scores {match_guid: {term: score}}
    """
    formulas = [retrieval_model.formulaFromExplanation(precomputed_query, doc_id) for doc_id in doc_list]
    raw_term_scores = listOfTermValuesInFormulas(formulas)
    formula_docfreq = getIDFfromFormulas(formulas)
    for term in raw_term_scores:
        raw_term_scores[term] = sum(raw_term_scores[term])

    sum_raw_term_scores = sum(raw_term_scores.values())

    normalised_term_scores = {}

    match_formulas = []

    for match_guid in precomputed_query["match_guids"]:
        match_formula = retrieval_model.formulaFromExplanation(precomputed_query, match_guid)
        match_formulas.append(match_formula)

        match_term_scores = getDictOfTermScores(match_formula.formula, "max")
        sum_match_term_scores = sum(match_term_scores.values())

        for term in match_term_scores:
            match_term_scores[term] = match_term_scores[term] / sum_match_term_scores

            # divisor = 1 + (raw_term_scores.get(term, 0) / sum_raw_term_scores)
            divisor = 1 + (raw_term_scores.get(term, 0))
            # squaring the divisor to decrease more
            match_term_scores[term] = match_term_scores[term] / float(pow(divisor, 2))
            # match_term_scores[term] = match_term_scores[term] / float(pow(idf, 2))

            # idf = 1 + math.log(len(formulas) / 1 + formula_docfreq.get(term, 0))
            # match_term_scores[term] = match_term_scores[term] / float(pow(divisor, pow(idf, 3)))

        normalised_sum = sum(match_term_scores.values())
        for term in match_term_scores:
            match_term_scores[term] /= normalised_sum

        normalised_term_scores[match_guid] = match_term_scores

    return normalised_term_scores, formulas, match_formulas


def filterTermScores(term_scores, docFreq, min_docs_to_match, max_docs_to_match, min_term_len=0, stopword_list=None):
    """
    filter terms that don't appear in a minimum of documents across the corpus
    """
    removed = {}
    filter_term_scores = {}
    for guid in term_scores:
        filter_term_scores[guid] = {}
        for term in term_scores[guid]:
            to_remove = False

            if stopword_list and term in stopword_list:
                to_remove = True
            elif max_docs_to_match and docFreq.get(term, 0) > max_docs_to_match:
                to_remove = True
            elif min_docs_to_match and docFreq.get(term, 0) < min_docs_to_match:
                to_remove = True
            elif len(term) < min_term_len:
                to_remove = True
            if to_remove:
                removed[term] = removed.get(term, 0) + 1
            else:
                filter_term_scores[guid][term] = term_scores[guid][term]

    print("Removed", removed)
    return filter_term_scores


class BaseKeywordSelector(object):
    """
    """

    def __init__(self, name):
        """
        """
        self.name = name

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters, cit, weights,
                       norm_term_scores=None,
                       docFreq=None, maxDocs=None, rawScores=None):
        """
        """
        pass

    def saveResults(self, path):
        pass


def main():
    pass


if __name__ == '__main__':
    main()


def addUpAllTermScores(term_scores, options={}):
    """
    Combines of the term scores over serveral guids

    :param term_scores:
    :return:
    """
    all_term_scores = {}
    num_docs_with_term = {}

    mode = options.get("terms_weight_mode", "add")
    multiplier = options.get("multiplier", 1)
    power = options.get("power", 1)
    min_val = options.get("min_val", None)
    add_val = options.get("add_val", 0)

    for guid in term_scores:
        for term in term_scores[guid]:
            all_term_scores[term] = all_term_scores.get(term, [])
            all_term_scores[term].append(term_scores[guid][term])

            num_docs_with_term[term] = num_docs_with_term.get(term, 0) + 1

    for term in all_term_scores:
        if mode == "add":
            all_term_scores[term] = sum(all_term_scores[term])
        elif mode == "max":
            all_term_scores[term] = max(all_term_scores[term])
        elif mode == "avg":
            all_term_scores[term] = sum(all_term_scores[term]) / len(all_term_scores[term])
        elif mode == "mul":
            res = 1
            for number in all_term_scores[term]:
                res *= number
            all_term_scores[term] = res

        all_term_scores[term] *= multiplier
        all_term_scores[term] = pow(all_term_scores[term], power)
        if min_val:
            all_term_scores[term] = max(min_val, all_term_scores[term])
        all_term_scores[term] += add_val

    return all_term_scores


def getCountsInQueryForMatchingTerms(precomputed_query):
    # x[1] is count, x[2] is weight in the structured query
    res = {x[0]: x[1] for x in precomputed_query["structured_query"]}
    for kp_tuple in precomputed_query.get("keyphrases", []):
        kp = " ".join(kp_tuple)
        pattern = kp.replace(" ", r"\W+")
        try:
            all_matches = re.findall(pattern, precomputed_query["query_text"])
        except Exception as e:
            print(e)
            continue

        count = len(all_matches)
        res[kp] = count
    return res