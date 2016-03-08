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
from run_single_formula import runSingleFormula

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
        scores.append(runSingleFormula([unique_result,parameters]))

    scores.sort(key=lambda x:x[0],reverse=True)
    return scores


def main():
    pass

if __name__ == '__main__':
    main()
