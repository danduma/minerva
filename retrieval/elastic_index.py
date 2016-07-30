# Indexer that uses Elasticsearch as a backend
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import json

from elasticsearch import Elasticsearch
from base_index import BaseIndexer
from minerva.retrieval.elastic_retrieval import ES_TYPE_DOC
import index_functions
from elastic_writer import ElasticWriter

class ElasticIndexer(BaseIndexer):
    """
        Prebuilds BOWs etc. for tests
    """
    def __init__(self, endpoint={"host":"localhost", "port":9200}, use_celery=False):
        """
        """
        super(self.__class__, self).__init__(use_celery)
        self.es=Elasticsearch([endpoint])
        self.es.retry_on_timeout=True

    def createIndex(self, index_name, fields):
        """
        """
        settings={
            "number_of_shards" : 1,
            "number_of_replicas" : 0
        }

        fields.append("_full_text")
        properties={field: {"type":"string", "index":"analyzed", "store":True} for field in fields}
        properties["metadata"] = {"type":"nested"}
        properties["guid"] = {"type":"string", "index": "not_analyzed"}
##        properties["bow_info"] = {"type":"string", "index": "analyzed", "store":True}

        if not self.es.indices.exists(index=index_name):
            print("Creating index %s " % index_name)
            self.es.indices.create(
                index=index_name,
                body={"settings":settings,"mappings":{ES_TYPE_DOC:{"properties":properties}}})
        else:
            print("Index %s already exists" % index_name)


    def createIndexWriter(self, actual_dir, max_field_length=20000000):
        """
            Returns an IndexWriter object created for the actual_dir specified
        """
        res=ElasticWriter(actual_dir, self.es)
        return res

def addDocument(writer, new_doc, metadata, fields_to_process, bow_info):
    """
        Add a document to the index. Does this using direct Elastic access.

        :param new_doc: dict of fields with values
        :type new_doc:dict
        :param metadata: ditto
        :type metadata:dict
        :param fields_to_process: only add these fields from the doc dict
        :type fields_to_process:list
    """

    body={"guid": metadata["guid"],
          "metadata": metadata,
##          "bow_info": json.dumps(bow_info),
          "bow_info": bow_info,
          }

    for field in fields_to_process:
        body[field]=new_doc[field]
        # TODO figure out what to do with per-field boosting
##            boost=1 / float(math.sqrt(total_numTerms)) if total_numTerms > 0 else float(0)
##            field_object.setBoost(float(boost))
##            doc.add(field_object)

    writer.addDocument(body)

index_functions.ADD_DOCUMENT_FUNCTION=addDocument


def main():
##    import minerva.db.corpora as cp
##    cp.useElasticCorpus()
##    cp.Corpus.connectCorpus()
    ei=ElasticIndexer(use_celery=False)
    writer=ei.createIndexWriter("idx_az_annotated_pmc_2013")
    writer.addDocument({"metadata":{"guid":"test"},"test":"test"})
    pass

if __name__ == '__main__':
    main()
