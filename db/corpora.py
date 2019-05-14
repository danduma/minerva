# Unit that allows sharing a single Corpus object for the whole app. Allows
# lazy loading of different kinds of Corpus objects
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

##from base_corpus import BaseCorpus

##global Corpus
from __future__ import absolute_import
from db.elastic_corpus import ElasticCorpus
Corpus=ElasticCorpus()

def useLocalCorpus():
    """
    """
    global Corpus
    from .local_corpus import LocalCorpus
    Corpus=LocalCorpus()

def useElasticCorpus():
    """
        22/12/16 NOTE Now deprecated. ElasticCorpus is now the default, so this
        method no longer does anything.
    """
    pass