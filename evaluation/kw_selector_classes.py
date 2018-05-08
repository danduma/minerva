from copy import deepcopy

import six

from evaluation.keyword_selection_measurement import runQueryAndMeasureKeywordSelection
from evaluation.keyword_selection import BaseKeywordSelector, getTermScores


def addUpAllTermScores(term_scores):
    all_term_scores = {}
    num_docs_with_term = {}
    for guid in term_scores:
        for term in term_scores[guid]:
            all_term_scores[term] = all_term_scores.get(term, 0) + term_scores[guid][term]
            num_docs_with_term[term] = num_docs_with_term.get(term, 0) + 1

    return all_term_scores


class NBestSelector(BaseKeywordSelector):
    """
        Selects just the general top scoring keywords from the query
    """

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters, cit, weights, term_scores=None):
        """
            Returns a selection of matching keywords for the document that should
            maximize its score

            Simplest implementation: take N-best keywords from the explanation for
            that paper

            :param precomputed_query: dict with the citation/doc details plus the structured_query
            :param doc_list: list of retrieval results
            :param retrieval_model: retrieval model we can use to get explanations from
            :param parameters: dict with {"N": <number>} for N-best
        """
        if not term_scores:
            term_scores = getTermScores(precomputed_query, doc_list, retrieval_model)

        all_term_scores = addUpAllTermScores(term_scores)

        terms = sorted(six.iteritems(all_term_scores), key=lambda x: x[1], reverse=True)
        terms = terms[:parameters["N"]]
        ##        print("Chosen terms:", terms,"\n\n")
        return terms


class MinimalSetSelector(BaseKeywordSelector):
    """
        Selects a minimal set of keywords from the query until it minimizes the rank
    """

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters, cit, weights, term_scores=None):
        """
            Returns a selection of matching keywords for the document that should
            maximize its score

            :param precomputed_query: dict with the citation/doc details plus the structured_query
            :param doc_list: list of retrieval results
            :param retrieval_model: retrieval model we can use to get explanations from
            :param parameters: dict with {"N": <number>} for N-best
        """
        if not term_scores:
            term_scores = getTermScores(precomputed_query, doc_list, retrieval_model)
        # remove all words that have less than 3 characters: simple way to get rid of the "i" tokens
        ##        for term in term_scores:
        ##            if len(term) < 3:
        ##                del term_scores[term]

        all_term_scores = addUpAllTermScores(term_scores)
        terms = sorted(six.iteritems(all_term_scores), key=lambda x: x[1], reverse=True)

        BIG_VALUE = 250
        rank = BIG_VALUE
        index = 0
        selected_terms = []

        history = []

        while rank != -1 and rank > 1 and index < len(terms):
            selected_terms.append(terms[index])
            kw_data = {}

            if terms[index][0] not in all_term_scores:
                print(terms[index][0], "not in", all_term_scores)

            selected_kws = [(term[0], all_term_scores.get(term[0], 0)) for term in selected_terms]

            runQueryAndMeasureKeywordSelection(precomputed_query,
                                               selected_kws,
                                               retrieval_model,
                                               weights,
                                               kw_data)

            for k in ["kw_selection_scores", "kw_selection_weight_scores"]:
                if kw_data[k]["rank"] == -1:
                    kw_data[k]["rank"] = BIG_VALUE

            rank_kw = kw_data["kw_selection_scores"]["rank"]
            rank_kw_weight = kw_data["kw_selection_weight_scores"]["rank"]

            rank = (rank_kw + rank_kw_weight) / 2.0

            history.append((deepcopy(selected_terms), rank))
            index += 1

        ##        print("Chosen terms:", terms,"\n\n")
        pick = min(history, key=lambda x: x[1])
        if pick[1] == BIG_VALUE:
            pick = min(history, key=lambda x: len(x[0]))

        return pick[0]
