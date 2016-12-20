# <description>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from collections import defaultdict
from math import log


def termScoresInFormula(part):
    """
        Returns list of all term matching elements in formula

        :param part: tuple, list or dict
        :returns: list of all term matching elements in formula
    """
    if isinstance(part, tuple) or isinstance(part, list):
        return part
    elif isinstance(part, dict):
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


def getDictOfTermScores(formula):
    """
        Returns the score of each term in the formula as a dict

        :param formula: formula
        :rtype: dict
    """
    term_scores=termScoresInFormula(formula)
    res={}
    for score in term_scores:
        # score=(field,qw,fw,term)
        res[score[3]]=res.get(score[3],0)+(score[1]*score[2])
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
        term_scores=getDictOfTermScores(formula["formula"])
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


def main():
    pass

if __name__ == '__main__':
    main()
