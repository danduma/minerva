# Unit that allows sharing a single Indexer object for the whole app.
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from base_index import BaseIndexer

global Indexer
Indexer=BaseIndexer()

def useLocalIndexer():
    """
    """
    global Indexer
    from lucene_index import LuceneIndexer
    Indexer=LuceneIndexer()

def useElasticCorpus():
    """
    """
    global Indexer
    from elastic_index import ElasticIndexer
    Indexer=ElasticIndexer()
