# Classes to annotate the best keywords from each context according to their retrieval scores
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
from collections import defaultdict
from math import log

import db.corpora as cp
from proc.structured_query import StructuredQuery
import six

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
            scores=[termScoresInFormula(sub_part) for sub_part in part["parts"]]
            result=[]
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
    term_scores=termScoresInFormula(formula)
    res={}
    if not term_scores:
        return res

    for score in term_scores:
        # score=(field,qw,fw,term)
        old_score=res.get(score[3],0)
        new_score=(score[1]*score[2])
        if op=="add":
            res[score[3]]=old_score+new_score
        elif op=="max":
            res[score[3]]=max(old_score,new_score)

##    print(res)
    return res

def getFormulaTermWeights(unique_result):
    """
        Computes a score for each matching keyword in the formula for the
        matching files in the index

        unique_result is a dict
        {"match_guid":"<guid>","formulas":[{"guid":"<guid>","formula":<formula>}]

        :rtype: dict
    """
    idf_scores=defaultdict(lambda:0)
    max_scores=defaultdict(lambda:0)

    formula_term_scores=[]
    match_result=None

    for formula in unique_result["formulas"]:
        term_scores=getDictOfTermScores(formula["formula"],"max")

        formula_term_scores.append((formula,term_scores))
        if formula["guid"] == unique_result["match_guid"]:
            match_result=term_scores

        for term in term_scores:
            idf_scores[term]=idf_scores[term]+term_scores[term]
            if term_scores[term] > max_scores[term]:
                max_scores[term] = term_scores[term]

    if not match_result:
        return None

    for term in idf_scores:
        idf_scores[term] = log((max_scores[term] * len(unique_result["formulas"])) / (1 + idf_scores[term]), 2)

    for term in match_result:
        match_result[term] = match_result[term] * idf_scores[term]

    return match_result


def makeStructuredQueryFromKeywords(keywords):
    """
        This is just to get around my former use of this query storage format
    """
    query=StructuredQuery()
    for kw in keywords:
        query.addToken(kw[0],1,boost=kw[1])
    return query

def evaluateKeywordSelection(precomputed_queries, extracted_keywords, exp, use_keywords=True, metric="mrr", index_field="text"):
    """
        Get the batch scores of a set of queries

        :param precomputed_queries: ditto
        :param extracted_keywords: a list of lists of tuples, one list for each query
    """
    from proc.results_logging import measureScores
    from retrieval.elastic_retrieval import ElasticRetrieval

    retrieval_model=ElasticRetrieval(exp["features_index_name"],"", es_instance=cp.Corpus.es)

    scores_list=[]

    for index, precomputed_query in precomputed_queries:
        scores={}
        if use_keywords:
            query=makeStructuredQueryFromKeywords(extracted_keywords[index])
        else:
            query=precomputed_query

        retrieved=retrieval_model.runQuery(query,max_results=exp.get("max_results_recall",200))
        measureScores(retrieved, precomputed_query["match_guid"],scores)
        scores_list.append(scores[metric])

    return sum(scores_list) / float(len(scores_list))


def listOfTermValuesInFormulas(formulas):
    """
        Returns a dict where {term: [list of values]} in all formulas
    """
    term_stats={}
    for formula in formulas:
        term_scores=getDictOfTermScores(formula.formula)
        for term in term_scores:
            if term not in term_stats:
                term_stats[term]=[]
            term_stats[term].append(term_scores[term])
    return term_stats

class BaseKeywordSelector(object):
    """
    """
    def __init__(self):
        """
        """

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters):
        """
        """
        pass

class NBestSelector(BaseKeywordSelector):
    """
        Selects just the general top scoring keywords from the query
    """
    def __init__(self):
        """
        """

    def selectKeywords(self, precomputed_query, doc_list, retrieval_model, parameters):
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
        res=[]

        formulas=[retrieval_model.formulaFromExplanation(precomputed_query, doc_id) for doc_id in doc_list]
        term_values=listOfTermValuesInFormulas(formulas)

        match_formula=retrieval_model.formulaFromExplanation(precomputed_query, precomputed_query["match_guid"])

        term_scores=getDictOfTermScores(match_formula.formula,"max")
        for term in term_scores:
            divisor=1+sum(term_values.get(term,[0.0]))
##            print("Term: %s, %.3f / %.3f" % (term, term_scores[term], float(divisor)))
            term_scores[term]=term_scores[term]/float(divisor*divisor)

        # remove all words that have less than 3 characters: simple way to get rid of the "i" tokens
##        for term in term_scores:
##            if len(term) < 3:
##                del term_scores[term]

        terms=sorted(six.iteritems(term_scores),key=lambda x:x[1], reverse=True)
        terms=terms[:parameters["N"]]
##        print("Chosen terms:", terms,"\n\n")
        return terms


def main():
    pass

if __name__ == '__main__':
    main()
