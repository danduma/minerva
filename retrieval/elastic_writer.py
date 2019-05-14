# Writer for Elasticsearch indexing
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from .elastic_retrieval import ES_TYPE_DOC
from elasticsearch.helpers import bulk
import time, datetime


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

    def flushBuffer(self):
        """
            Does nothing, there is no buffer to flush
        """
        pass


class BufferedElasticWriter(object):
    """
        Like ElasticWriter but writes out using the bulk API
    """

    def __init__(self, index_name, es_instance, bufsize=100):
        """

        """
        self.es = es_instance
        self.index_name = index_name
        self.buffer = []
        self.bufsize = bufsize

    def createIndex(self, index_name):
        """
            This is done somewhere else in fact
        """
        pass

    def setIndexRefresh(self, seconds):
        """

        :param seconds:
        """
        put = self.es.indices.put_settings(
            index=self.index_name,
            body='''{
                "index": {
                    "refresh_interval": \"%s\"
                }
            }''' % str(seconds),  # "blocks.write": true
            ignore_unavailable=True
        )

    def flushBuffer(self):
        actions = []

        self.setIndexRefresh("-1")

        for doc in self.buffer:
            id = doc["metadata"]["guid"]
            actions.append({
                '_index': self.index_name,
                '_type': ES_TYPE_DOC,
                '_id': id,
                # '_routing': 5,
                # 'pipeline': 'my-ingest-pipeline',
                '_source': doc
                # {
                #     "body": doc
                # }
            })

        success = False
        while not success:
            try:
                bulk(self.es, actions)
                success=True
            except Exception as e:
                print(e)
                print(datetime.datetime.now(),"Exception running bulk(). Waiting for 5 sec and retrying.")
                time.sleep(5)

        try:
            self.setIndexRefresh("10s")
        except Exception as e:
            print(e)
            print(datetime.datetime.now(),"Exception running setIndexRefresh(). Waiting for 5 sec and retrying.")
            time.sleep(5)
            self.setIndexRefresh("10s")


        try:
            self.es.indices.forcemerge(index=self.index_name)
        except Exception as e:
            print("Exception running forcemerge(). Waiting 5 sec")
            print(e)
            time.sleep(5)

        self.buffer = []

    def addDocument(self, doc):
        """
            Emulate LuceneIndexWriter.addDocument for elastic

            :param doc: dict of doc to index, with all fields as they will be \
            stored. Metadata contains the GUID to index it by
            :type doc:dict
        """
        self.buffer.append(doc)
        if len(self.buffer) >= self.bufsize:
            self.flushBuffer()

    def close(self):
        """
            Make sure to flush the buffer
        """
        self.flushBuffer()


def main():
    pass


if __name__ == '__main__':
    main()
