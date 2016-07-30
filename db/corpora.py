# Unit that allows sharing a single Corpus object for the whole app. Allows
# lazy loading of different kinds of Corpus objects
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from base_corpus import BaseCorpus

##global Corpus
Corpus=BaseCorpus()

def useLocalCorpus():
    """
    """
    global Corpus
    from local_corpus import LocalCorpus
    Corpus=LocalCorpus()

def useElasticCorpus():
    """
    """
    global Corpus
    from elastic_corpus import ElasticCorpus
    Corpus=ElasticCorpus()
