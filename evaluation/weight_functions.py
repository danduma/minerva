# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import random
from copy import deepcopy
from multiprocessing import Pool, cpu_count

from sklearn import cross_validation

from pipeline_functions import getDictOfTestingMethods
from minerva.retrieval.stored_formula import StoredFormula

##def runSingleFormula(result_tuple):
##    """
##        Computes the score of a single formula given the parameters
##
##        :param result_tuple: this was to run with multiprocessing. Tuple of
##            2 elementS: (unique_result, parameters)
##    """
##    unique_result, parameters = result_tuple
##    formula=StoredFormula(unique_result["formula"])
##    score=formula.computeScore(formula.formula, parameters)
##    return (score,{"guid":unique_result["guid"]})


def addExtraWeights(weights, exp):
    """
        Deep copies the weight dictionary and adds the fixed weights if any
        are specified in the experiment
    """
    if isinstance(weights,list):
        weights={x:1 for x in weights}
    res=deepcopy(weights)
    for extra_method in exp.get("fixed_runtime_parameters",{}):
        res[extra_method]=exp["fixed_runtime_parameters"][extra_method]
    return res

def runPrecomputedQuery(retrieval_results, parameters):
    """
        This takes a query that has already had the results added and parameters
        and returns the computed scores
    """
    scores=[]
##    numproc=cpu_count()
##    pool = Pool(numproc)
####    print("Spawning workers")
##    for index in range(0, len(retrieval_results), numproc):
##        new_scores=pool.map(runSingleFormula, [(r, parameters) for r in retrieval_results[index:index+numproc]])
##        scores.extend(new_scores)
##    pool.close()
##    pool.join()
    for unique_result in retrieval_results:
        formula=StoredFormula(unique_result["formula"])
        score=formula.computeScore(formula.formula, parameters)
        scores.append((score,{"guid":unique_result["guid"]}))
##        scores.append(runSingleFormula([unique_result,parameters]))

    scores.sort(key=lambda x:x[0],reverse=True)
    return scores


def main():
    pass

if __name__ == '__main__':
    main()
