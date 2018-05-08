# Writer for Elasticsearch indexing
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from .elastic_retrieval import ES_TYPE_DOC
from copy import deepcopy


class ElasticWriter(object):
    """
        Identifies with a specific elastic index in a specific elasticsearch
        instance to encapsulate writing to it. More or less emulates lucene
        Writer class.
    """

    def __init__(self, index_name, es_instance):
        """

        """
        self.es = es_instance
        self.index_name = index_name

    def createIndex(self, index_name):
        """
        """

    def addDocument(self, doc):
        """
            Emulate LuceneIndexWriter.addDocument for elastic

            :param doc: dict of doc to index, with all fields as they will be \
            stored. Metadata contains the GUID to index it by
            :type doc:dict
        """
        id = doc["metadata"]["guid"]
        # print("Indexing {} in index {}".format(id, self.index_name))
        print(doc)
        self.es.index(
            index=self.index_name,
            doc_type=ES_TYPE_DOC,
            op_type="index",
            id=id,
            body=doc
        )

    def close(self):
        """
            Do nothing, for there is nothing to do.
        """
        pass


def main():
    pass


if __name__ == '__main__':
    main()
