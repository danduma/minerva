# Testing pipeline classes
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import glob, math, os, re, sys, gc, random, json
from copy import deepcopy
from collections import defaultdict, namedtuple, OrderedDict

import lucene
from minerva.evaluation.lucene_retrieval import (LuceneRetrieval,
LuceneRetrievalBoost,precomputedExplainRetrieval, MAX_RESULTS_RECALL)

from base_pipeline import BasePipeline

class LuceneTestingPipeline(BasePipeline):
    """
        Testing pipeline using local Lucene indexes
    """
    def __init__(self):
        self.retrieval_class=LuceneRetrievalBoost
        pass

    def initializePipeline(self):
        """
        """
        try:
            lucene.initVM(maxheap="640m") # init Lucene VM
        except ValueError:
            # VM already up
            print(sys.exc_info()[1])


def main():
    pass

if __name__ == '__main__':
    main()
