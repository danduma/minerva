# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


from stored_formula import StoredFormula

def runSingleFormula(result_tuple):
    """
    """
    unique_result, parameters = result_tuple
    formula=StoredFormula(unique_result["formula"])
    score=formula.computeScore(formula.formula, parameters)
    return (score,{"guid":unique_result["guid"]})


def main():
    pass

if __name__ == '__main__':
    main()
