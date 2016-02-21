# Prebuild bag-of-words representations
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import sys, json, datetime, math

import minerva.db.corpora as cp
import minerva.proc.doc_representation as doc_representation
from minerva.proc.general_utils import loadFileText, writeFileText, ensureDirExists

from elasticsearch import Elasticsearch
from base_index import BaseIndexer
from elastic_retrieval import ES_TYPE_DOC

class ElasticWriter(object):
    """
        Identifies with a specific elastic index in a specific elasticsearch
        instance to encapsulate writing to it. More or less emulates lucene
        Writer class.
    """
    def __init__(self, index_name, es_instance):
        """

        """
        self.es=es_instance
        self.index_name=index_name

    def addDocument(self,doc):
        """
            Emulate LuceneIndexWriter.addDocument for elastic

            :param doc: dict of doc to index, with all fields as they will be \
            stored. Metadata contains the GUID to index it by
            :type doc:dict
        """
        id=doc["metadata"]["guid"]
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

class ElasticIndexer(BaseIndexer):
    """
        Prebuilds BOWs etc. for tests
    """
    def __init__(self, endpoint={"host":"localhost", "port":9200}):
        """
        """
        self.es=Elasticsearch([endpoint])
        self.es.retry_on_timeout=True

    def createIndexWriter(self, actual_dir, max_field_length=20000000):
        """
            Returns an IndexWriter object created for the actual_dir specified
        """
        res=ElasticWriter(actual_dir, self.es)
        return res

    def addDocument(self, writer, new_doc, metadata, fields_to_process, bow_info):
        """
            Add a document to the index. Does this using direct Lucene access.

            :param new_doc: dict of fields with values
            :type new_doc:dict
            :param metadata: ditto
            :type metadata:dict
            :param fields_to_process: only add these fields from the doc dict
            :type fields_to_process:list
        """

        body={"guid": metadata["guid"],
              "metadata": metadata,
              "bow_info": bow_info,
              }

        for field in fields_to_process:
            body[field]=new_doc[field]
            # TODO figure out what to do with per-field boosting
##            boost=1 / float(math.sqrt(total_numTerms)) if total_numTerms > 0 else float(0)
##            field_object.setBoost(float(boost))
##            doc.add(field_object)

        writer.addDocument(body)

def main():

    pass

if __name__ == '__main__':
    main()
