from __future__ import print_function

from collections import Counter
from copy import deepcopy

import six
import scipy.optimize as opt
import time
import numpy
import re

from proc.structured_query import StructuredQuery

numpy.warnings.filterwarnings('ignore')

from evaluation.keyword_annotation_measurement import runQueryAndMeasureKeywordSelection
from evaluation.keyword_annotation import BaseKeywordSelector, filterTermScores, getDictOfTermScores, \
    addUpAllTermScores, getCountsInQueryForMatchingTerms

from models.keyword_features import tokenWeight

from proc.stopword_proc import getStopwords, c3_stopwords

BIG_VALUE = 200


def selectTerms(params, terms):
    selected_kws = []
    for idx, param in enumerate(params):
        if param > 0:
            selected_kws.append(terms[idx])
    return selected_kws


def getSortedTerms(term_scores, precomputed_query, options={}):
    all_term_scores = addUpAllTermScores(term_scores, options=options)

    terms = sorted(six.iteritems(all_term_scores), key=lambda x: x[1], reverse=True)
    counts = getCountsInQueryForMatchingTerms(precomputed_query)
    terms = [(term[0], tokenWeight(term)) for term in terms]

    return all_term_scores, terms, counts


def getKPScore(parameters, term_scores, kp):
    score_computing = parameters.get("kp_score_compute", "add")
    if score_computing == "add":
        score = sum([term_scores[term] for term in kp])
    elif score_computing == "avg":
        score = sum([term_scores[term] for term in kp]) / len(kp)
    elif score_computing == "const":
        score = float(parameters.get("kp_score_const", 1.0))
    else:
        raise ValueError("Unknown kp_score_compute value")

    mul = float(parameters.get("kp_score_mul", 1.0))
    score = score * mul

    return score


def filterStopwords(retrieval_model, term_scores, docFreq):
    stopwords = getStopwords(retrieval_model.index_name)
    term_scores = filterTermScores(term_scores,
                                   docFreq,
                                   min_docs_to_match=1,
                                   max_docs_to_match=None,
                                   min_term_len=3,
                                   stopword_list=stopwords)
    return term_scores


def filterC3Stopwords(norm_term_scores):
    stopwords = c3_stopwords
    for s in stopwords:
        for guid in norm_term_scores:
            if s in norm_term_scores[guid]:
                del norm_term_scores[guid][s]

    return norm_term_scores


class KPtester(BaseKeywordSelector):
    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters, cit,
                       weights, norm_term_scores=None,
                       docFreq=None, maxDocs=None, rawScores=None):

        if parameters.get("use_c3_stopword_list", False):
            norm_term_scores = filterC3Stopwords(norm_term_scores)

        if parameters.get("filter_stopwords", True):
            norm_term_scores = filterStopwords(retrieval_model, norm_term_scores, docFreq)

        all_term_scores, terms, counts = getSortedTerms(norm_term_scores,
                                                        precomputed_query,
                                                        options=parameters
                                                        )

        if not parameters.get("use_weights", True):
            # if parameters.get("use_counts", True):
            #     # (term, count, weight)
            #     terms2 = [(term[0], term[2], 1) for term in terms]
            # else:
            terms2 = [(term[0], 1) for term in terms]
            terms = terms2

        use_kps = parameters.get("use_kps", False)
        if not use_kps:
            return [(term[0], tokenWeight(term)) for term in terms]

        kp_method = parameters.get("kp_method", "add")

        norm_term_scores = {term[0]: tokenWeight(term) for term in terms}

        if kp_method == "add":
            res = []
            for kp in precomputed_query.get("keyphrases", []):
                if any([term not in norm_term_scores for term in kp]):
                    # KP doesn't fully match, can't add it
                    continue

                score = getKPScore(parameters, norm_term_scores, kp)

                kp = " ".join(kp)
                res.append((kp, score))

            return terms + res
        elif kp_method == "sub":
            res = []
            term_counts = {term[0]: tokenWeight(term) for term in terms}
            for kp in precomputed_query.get("keyphrases", []):
                if any([term not in norm_term_scores for term in kp]):
                    # KP doesn't fully match, can't add it
                    continue

                score = getKPScore(parameters, norm_term_scores, kp)

                for term in kp:
                    term_counts[term] -= 1

                kp = " ".join(kp)
                res.append((kp, score))
            terms = [(term, norm_term_scores[term]) for term in term_counts if term_counts[term] > 0]
            return terms + res
        else:
            raise ValueError("Unknown kp_method")


def termsPercentOfFormulas(formulas, terms):
    """
        Returns a dict where {term: [list of values]} in all formulas
    """
    term_stats = {term: [] for term in terms}

    for formula in formulas:
        term_scores = getDictOfTermScores(formula.formula)
        sum_all_terms = sum(term_scores.values())

        for term in terms:
            if sum_all_terms > 0:
                pct = term_scores.get(term, 0) / float(sum_all_terms)
            else:
                pct = 0
            term_stats[term].append(pct)
    return term_stats


def termsPercentOfTermScores(all_term_scores, terms):
    """
    Returns the pct of the normalised match_term_scores that each term in terms is taking
    """
    res = {term: [] for term in terms}
    sum_term_scores = sum(all_term_scores.values())

    for term in terms:
        if term in all_term_scores:  # if it matches any of the match_guid docs
            # add its normalised contribution to the combined score
            res[term].append(all_term_scores[term] / sum_term_scores)
            # this_context.append(norm_term_scores[term])
        else:
            res[term].append(0)

    return res



def makeStructuredQueryAgain(precomputed_query):
    original_query = precomputed_query["vis_text"]
    original_query = original_query.replace("__cit", " ")
    original_query = original_query.replace("__author", " ")
    original_query = original_query.replace("__ref", " ")

    all_tokens = re.findall(r"\w+", original_query.lower())
    terms = list(set(all_tokens))
    counts = Counter(all_tokens)

    norm_term_scores = {guid: {t: 1 for t in terms} for guid in
                        precomputed_query["match_guids"]}
    all_term_scores = addUpAllTermScores(norm_term_scores)

    precomputed_query["structured_query"] = StructuredQuery([
        {"token": term, "boost": 1, "count": counts.get(term, 0)} for term in all_term_scores])
    # all_term_scores = addUpAllTermScores(norm_term_scores)

    return precomputed_query


class QueryTester(BaseKeywordSelector):
    """
    Runs the original query, including all terms, not just the ones that match with match_guid docs
    """

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters, cit,
                       weights, norm_term_scores=None,
                       docFreq=None, maxDocs=None, rawScores=None):

        if parameters.get("use_all_original_text", False):
            original_query = precomputed_query["vis_text"]
            original_query = original_query.replace("__cit", " ")
            original_query = original_query.replace("__author", " ")
        else:
            original_query = precomputed_query["query_text"]

        # terms = original_query.lower().split()
        all_tokens = re.findall(r"\w+", original_query.lower())
        terms = list(set(all_tokens))
        counts = Counter(all_tokens)

        if parameters.get("use_weights", True):
            norm_term_scores = {guid: {t: norm_term_scores[guid].get(t, 0.0) for t in terms} for guid in
                                precomputed_query["match_guids"]}
        else:
            norm_term_scores = {guid: {t: 1.0 for t in terms} for guid in
                                precomputed_query["match_guids"]}

        if parameters.get("use_c3_stopword_list", False):
            norm_term_scores = filterC3Stopwords(norm_term_scores)

        if parameters.get("filter_stopwords", True):
            norm_term_scores = filterStopwords(retrieval_model, norm_term_scores, docFreq)

        all_term_scores = addUpAllTermScores(norm_term_scores)
        terms = sorted(six.iteritems(all_term_scores), key=lambda x: x[1], reverse=True)

        precomputed_query["structured_query"] = StructuredQuery([
            {"token": term, "boost": all_term_scores[term], "count": counts.get(term, 0)} for term in all_term_scores])

        terms = [(term[0], tokenWeight(term)) for term in terms]

        return terms


class StopwordTester(BaseKeywordSelector):
    """
    Collects statistics about how much weight stopwords contribute, raw and normalised
    """

    def __init__(self, name):
        super(StopwordTester, self).__init__(name)
        self.pct_of_formulas = {}
        self.pct_of_match_formulas = {}
        self.pct_of_norm_term_scores = {}
        self.num_results = 0

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters, cit,
                       weights, norm_term_scores=None,
                       docFreq=None, maxDocs=None, rawScores=None):

        stopwords = getStopwords(retrieval_model.index_name)
        stopwords = [t for t in list(stopwords) if len(t) > 1]
        norm_term_scores = filterTermScores(norm_term_scores,
                                            docFreq,
                                            min_docs_to_match=0,
                                            max_docs_to_match=None,
                                            min_term_len=2)

        this_pct_of_formulas = termsPercentOfFormulas(rawScores["formulas"], stopwords)
        this_pct_of_match_formulas = termsPercentOfFormulas(rawScores["match_formulas"], stopwords)

        # normalised term scores are the scores for the matches (added) divided by the total term value

        all_term_scores, terms, counts = getSortedTerms(norm_term_scores,
                                                        precomputed_query,
                                                        options=parameters
                                                        )

        this_pct_of_norm_term_scores = termsPercentOfTermScores(all_term_scores, stopwords)

        if len(self.pct_of_formulas) == 0:
            for term2 in stopwords:
                self.pct_of_formulas[term2] = []
                self.pct_of_match_formulas[term2] = []
                self.pct_of_norm_term_scores[term2] = []

        for term in this_pct_of_formulas:
            self.pct_of_formulas[term].append(sum(this_pct_of_formulas[term]) / len(this_pct_of_formulas[term]))
            self.pct_of_match_formulas[term].append(
                sum(this_pct_of_match_formulas[term]) / len(this_pct_of_match_formulas[term]))
            self.pct_of_norm_term_scores[term].append(
                sum(this_pct_of_norm_term_scores[term]) / len(this_pct_of_norm_term_scores[term]))

        self.num_results += 1
        print("pct_of_formulas - the: %0.2f%%" % (
                sum(self.pct_of_formulas["the"]) / len(self.pct_of_formulas["the"]) * 100))
        print("pct_of_norm_term_scores - the: %0.2f%%" % (
                sum(self.pct_of_norm_term_scores["the"]) / len(self.pct_of_norm_term_scores["the"]) * 100))
        return []

    def saveResults(self, path):

        def writeResults(f, stopwords_pct, numformulas, label, top_res=20):
            all_sum = sum(stopwords_pct.values())

            total_pct = all_sum / numformulas

            f.write("\n[" + label + "]\n\n")
            f.write("TOTAL PCT %0.2f%%\n" % (total_pct * 100))

            stopwords_pct = sorted(stopwords_pct.items(), key=lambda x: x[1], reverse=True)

            f.write(
                "PCT TOP %d %0.2f%%\n" % (top_res, (sum([x[1] for x in stopwords_pct[:top_res]]) / numformulas) * 100))
            for term in stopwords_pct:
                f.write("%s\t%0.2f%%\n" % (term[0], term[1] * 100))

        def writeCSV(filename, items, numformulas, top_res=20):
            import pandas as pd

            stopwords_pct = items[0]["terms_pct"]
            stopwords_pct = sorted(stopwords_pct.items(), key=lambda x: x[1], reverse=True)

            lines = []
            for term in stopwords_pct:
                line = {"term": term[0]}
                for item in items:
                    line[item["label"]] = item["terms_pct"][term[0]] * 100

                lines.append(line)

            line = {"term": "TOTAL PCT"}
            for item in items:
                line[item["label"]] = (sum(item["terms_pct"].values())) * 100
            lines.append(line)

            line = {"term": "AVG TOP %d " % top_res}
            for item in items:
                stopwords_pct = sorted(item["terms_pct"].items(), key=lambda x: x[1], reverse=True)
                line[item["label"]] = (sum([x[1] for x in stopwords_pct[:top_res]])) * 100
            lines.append(line)

            df = pd.DataFrame(lines)
            df.to_csv(filename)

        print("Saving StopwordTester data")
        for term in self.pct_of_formulas:
            for to_normalise in [self.pct_of_formulas, self.pct_of_match_formulas, self.pct_of_norm_term_scores]:
                if len(to_normalise[term]) > 0:
                    to_normalise[term] = sum(to_normalise[term]) / float(self.num_results)
                else:
                    to_normalise[term] = 0.0

        avg_pct_of_formulas = sum(self.pct_of_formulas.values()) / len(self.pct_of_formulas)
        avg_pct_of_match_formulas = sum(self.pct_of_match_formulas.values()) / len(self.pct_of_match_formulas)
        avg_pct_of_norm_term_scores = sum(self.pct_of_norm_term_scores.values()) / len(self.pct_of_norm_term_scores)

        print("Report for %s" % self.name)
        print("Sum raw pct contributed to")
        # print("formulas: ", avg_pct_of_formulas)
        # print("match_formulas: ", avg_pct_of_match_formulas)
        # print("norm_term_scores: ", avg_pct_of_norm_term_scores)
        print("formulas: %0.3f%%" % (sum(self.pct_of_formulas.values()) * 100))
        print("match_formulas: %0.3f%%" % (sum(self.pct_of_match_formulas.values()) * 100))
        print("norm_term_scores: %0.3f%%" % (sum(self.pct_of_norm_term_scores.values()) * 100))
        print("\n")

        numformulas = self.num_results

        with open(path, "w") as f:
            writeResults(f, self.pct_of_formulas, numformulas, "pct_of_formulas")
            writeResults(f, self.pct_of_match_formulas, numformulas, "pct_of_match_formulas")
            writeResults(f, self.pct_of_norm_term_scores, numformulas, "pct_of_norm_term_scores")

        path = path.replace(".txt", ".csv")
        writeCSV(path, [
            {"terms_pct": self.pct_of_formulas, "label": "pct_of_formulas"},
            {"terms_pct": self.pct_of_match_formulas, "label": "pct_of_match_formulas"},
            {"terms_pct": self.pct_of_norm_term_scores, "label": "pct_of_norm_term_scores"},
        ], numformulas)


class NBestSelector(BaseKeywordSelector):
    """
        Selects just the general top scoring keywords from the query
    """

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters, cit,
                       weights, norm_term_scores=None,
                       docFreq=None, maxDocs=None, rawScores=None):
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

        if parameters.get("filter_stopwords", True):
            norm_term_scores = filterStopwords(retrieval_model, norm_term_scores, docFreq)

        all_term_scores, terms, counts = getSortedTerms(norm_term_scores, precomputed_query)

        terms = terms[:parameters["N"]]
        ##        print("Chosen terms:", terms,"\n\n")
        return terms


class AllSelector(BaseKeywordSelector):
    """
        Selects just the general top scoring keywords from the query
    """

    def selectKeywords(self, precomputed_query, doc_list,
                       retrieval_model, parameters, cit,
                       weights, norm_term_scores=None,
                       docFreq=None, maxDocs=None, rawScores=None):
        """
            Returns all matching keywords with the score of each

            :param precomputed_query: dict with the citation/doc details plus the structured_query
            :param doc_list: list of retrieval results
            :param retrieval_model: retrieval model we can use to get explanations from
            :param parameters: dict with {"N": <number>} for N-best
        """

        if parameters.get("filter_stopwords", True):
            norm_term_scores = filterStopwords(retrieval_model, norm_term_scores, docFreq)

        all_term_scores, terms, counts = getSortedTerms(norm_term_scores, precomputed_query)

        # terms = terms[:parameters["N"]]
        ##        print("Chosen terms:", terms,"\n\n")
        return terms


class MinimalSetSelector(BaseKeywordSelector):
    """
        Selects a minimal set of keywords from the query until it minimizes the rank
    """

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters, cit,
                       weights, norm_term_scores=None,
                       docFreq=None, maxDocs=None, rawScores=None):
        """
            Returns a selection of matching keywords for the document that should
            maximize its score

            :param precomputed_query: dict with the citation/doc details plus the structured_query
            :param doc_list: list of retrieval results
            :param retrieval_model: retrieval model we can use to get explanations from
            :param parameters: dict with {"N": <number>} for N-best
        """

        if parameters.get("filter_stopwords", True):
            norm_term_scores = filterStopwords(retrieval_model, norm_term_scores, docFreq)

        all_term_scores, terms, counts = getSortedTerms(norm_term_scores, precomputed_query)

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


        if len(history) == 0:
            return []

        pick = min(history, key=lambda x: x[1])
        if pick[1] == BIG_VALUE:
            pick = min(history, key=lambda x: len(x[0]))

        return pick[0]


class MultiMaximalSetSelector(BaseKeywordSelector):
    """
        Selects a minimal set of keywords from the query until it minimizes the rank
        of all relevant documents together
    """

    def getQueryRank(self, precomputed_query, selected_kws, retrieval_model, weights):
        if len(selected_kws) == 0:
            return {"rank_kw": float(BIG_VALUE),
                    "rank_kw_weight": float(BIG_VALUE),
                    "rank_avg": float(BIG_VALUE)}

        scores = {}
        runQueryAndMeasureKeywordSelection(precomputed_query,
                                           selected_kws,
                                           retrieval_model,
                                           weights,
                                           scores)

        for k in ["kw_selection_scores", "kw_selection_weight_scores"]:
            if scores[k]["rank"] == -1:
                assert False
                # scores[k]["rank"] = BIG_VALUE

        rank_kw = scores["kw_selection_scores"]["rank"]
        rank_kw_weight = scores["kw_selection_weight_scores"]["rank"]
        rank_avg = (rank_kw + rank_kw_weight) / 2.0

        res = {"rank_kw": float(rank_kw),
               "rank_kw_weight": float(rank_kw_weight),
               "rank_avg": float(rank_avg)}
        return res

    def optimizeSciPy(self, precomputed_query, terms, weights, retrieval_model, all_terms_scores):
        def OptimizeFunc(params):
            print('.', end='')
            rank_res = self.getQueryRank(precomputed_query,
                                         selectTerms(params, terms),
                                         retrieval_model,
                                         weights)["rank_kw_weight"]
            return rank_res

        # initial_guess = [0 for term in terms]
        slice_obj = [0 for term in terms]
        method_results = {}

        # for method in [
        #     "COBYLA", "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP",
        #     # "trust-constr",
        #     #  "dogleg",
        #     # "trust-ncg",
        #     # "trust-exact", "trust-krylov"
        #     ]:
        method = "COBYLA"
        start_time = time.time()
        pick = opt.minimize(OptimizeFunc,
                            slice_obj,
                            method=method,
                            # bounds=[(0, 1) for term in terms],
                            options={"maxiter": 500},
                            # constraints=[{'type': 'eq', 'fun': lambda x: max([x[i] - int(x[i]) for i in range(len(x))])}]
                            )
        took = time.time() - start_time
        #     method_results[method] = (pick.fun, took)
        #     print()
        #     print(method, pick.fun, took)
        #     # print("--- Took %s seconds ---" % (time.time() - start_time))
        #
        # for method in method_results:
        #     print(method, method_results[method])
        # assert False

        selected_kws = selectTerms(pick.x, terms)
        score = self.getQueryRank(precomputed_query, selected_kws, retrieval_model, weights)
        if score["rank_kw_weight"] == BIG_VALUE:
            print("COBYLA FAIL!!")

        return selected_kws, score, "Scipy-COBYLA", took

    def optimizeAdding(self, precomputed_query, terms, weights, retrieval_model):
        index = 0
        selected_terms = []
        rank = BIG_VALUE

        start_time = time.time()
        history = []
        while rank != -1 and rank > 1 and index < len(terms):
            selected_terms.append(terms[index])

            if terms[index][0] not in self.all_term_scores:
                print(terms[index][0], "not in", self.all_term_scores)

            selected_kws = [(term[0], self.all_term_scores.get(term[0], 0)) for term in selected_terms]

            scores = self.getQueryRank(precomputed_query,
                                       selected_kws,
                                       retrieval_model,
                                       weights)

            history.append((deepcopy(selected_terms),
                            scores["rank_kw"],
                            scores["rank_kw_weight"],
                            scores["rank_avg"]))
            index += 1


        if len(history) == 0:
            pick = ([],0)
        else:
            #        print("Chosen terms:", terms,"\n\n")
            pick = min(history, key=lambda x: tokenWeight(x))
            if pick[1] == BIG_VALUE:
                pick = min(history, key=lambda x: len(x[0]))

        took = time.time() - start_time

        return pick[0], float(pick[1]), "Adding", took

    def optimizeAddSub(self, precomputed_query, terms, weights, retrieval_model):

        def doOnePass(pick_x):
            selected_kws = selectTerms(pick_x, terms)
            # selected_terms = [term[0] for term in selected_kws]

            scores = self.getQueryRank(precomputed_query,
                                       selected_kws,
                                       retrieval_model,
                                       weights)

            current_score = (deepcopy(selected_kws),
                             scores["rank_kw"],
                             scores["rank_kw_weight"],
                             scores["rank_avg"])

            self.current_score = current_score
            if not self.best_score:
                self.improvement_this_pass = True
            elif scores["rank_kw_weight"] == self.best_score[2]:
                if len(selected_kws) > len(self.best_score[0]):
                    self.improvement_this_pass = True
            elif scores["rank_kw_weight"] < self.best_score[2]:
                self.improvement_this_pass = True

            if self.improvement_this_pass:
                self.best_score = current_score

        rank = BIG_VALUE

        start_time = time.time()
        pick_x = [1 for term in terms]
        self.best_score = None
        self.current_score = None
        pass_num = 0

        doOnePass(pick_x)

        for curr_pass in ["-", "+"]:
            if rank == 1:
                break

            for index in reversed(range(len(pick_x))):
                self.improvement_this_pass = False
                previous_weights = deepcopy(pick_x)

                pass_num += 1

                if curr_pass == "+":
                    if pick_x[index] > 0:
                        continue
                    pick_x[index] = 1
                elif curr_pass == "-":
                    if pick_x[index] == 0:
                        continue
                    pick_x[index] = 0

                doOnePass(pick_x)

                rank = self.best_score[2]
                print("Pass %d: Current: %0.1f Best: %0.1f" % (pass_num, self.current_score[2], self.best_score[2],))
                # print(previous_weights)
                print(pick_x)

                # if nothing's changed, back to previous weights
                if not self.improvement_this_pass:
                    pick_x = deepcopy(previous_weights)

                if rank < 2:
                    break

        took = time.time() - start_time

        return self.best_score[0], float(self.best_score[2]), "AddSub", took

    def selectKeywords(self, precomputed_query, doc_list,
                       retrieval_model, parameters, cit,
                       weights, norm_term_scores=None,
                       docFreq=None, maxDocs=None, rawScores=None):
        """
            Returns a selection of matching keywords for the document that should
            maximize its score

            :param precomputed_query: dict with the citation/doc details plus the structured_query
            :param doc_list: list of retrieval results
            :param retrieval_model: retrieval model we can use to get explanations from
            :param parameters: dict with {"N": <number>} for N-best
        """

        self.docFreq = docFreq
        self.maxDocs = maxDocs

        # if len(precomputed_query["match_guids"]) > 1:
        #     print("Multi citation")

        if parameters.get("filter_stopwords", True):
            norm_term_scores = filterStopwords(retrieval_model, norm_term_scores, docFreq)

        all_term_scores = addUpAllTermScores(norm_term_scores)

        terms = sorted(six.iteritems(all_term_scores), key=lambda x: x[1], reverse=True)

        opt_results = [
            # self.optimizeSciPy(precomputed_query, terms, weights, retrieval_model),
            # self.optimizeAdding(precomputed_query, terms, weights, retrieval_model),
            self.optimizeAddSub(precomputed_query, terms, weights, retrieval_model)
        ]

        opt_results = sorted(opt_results, key=lambda x: x[1], reverse=False)
        print("\nOptimisation results:")
        for res in opt_results:
            # print(f"{res[2]}\t{res[1]}\t{res[3]}\t{res[0]}")
            print("%s\t%s\t%s\t%s" % (str(res[2]), str(res[1]), str(res[3]), str(res[0])))

        selected_kws = opt_results[0][0]
        return selected_kws

