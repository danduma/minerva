# <purpose>
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import json

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionTimeout, ConnectionError

class ElasticResultStorer(object):
    def __init__(self, namespace, table_name, endpoint={"host":"localhost", "port":9200}):
        """
        """
        assert isinstance(namespace, basestring)
        assert isinstance(table_name, basestring)

        self.namespace=namespace
        self.table_name=table_name
        self.index_name=namespace.lower()+"_"+table_name.lower()

        self.endpoint=endpoint
        self.es=None
        self.connect()
        self.createTable()

    def connect(self, ):
        """
            Connect to elasticsearch server
        """
        self.es = Elasticsearch([self.endpoint], timeout=60)
        self.es.retry_on_timeout=True

    def createTable(self):
        """
        """
        settings={
            "number_of_shards" : 1,
            "number_of_replicas" : 0
        }
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                body={"settings":settings})

    def deleteTable(self):
        """
        """
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])

    def clearResults(self):
        """
            Deletes the result table and recreates it
        """
        self.deleteTable()
        self.createTable()

    def addResult(self, result):
        """
            Stores a result in the db
        """
        assert isinstance(result, dict)

        body=self.resultPreStore(result)

        self.es.index(
            index=self.index_name,
            doc_type="result",
            op_type="create",
            body=body
            )

    def resultPreStore(self, result):
        """
        """
        return {key:json.dumps(result[key]) for key in result}

    def resultPreLoad(self, result):
        """
        """
        return {key:json.loads(result[key]) for key in result}

    def readResults(self):
        """
            Returns a list of all results in the table.
        """
        scroll_time="2m"

        res=self.es.search(
            index=self.index_name,
            doc_type="result",
##            size=10000,
            search_type="scan",
            scroll=scroll_time,
            body={"query":{"match_all": {}}}
            )

        results = [self.resultPreLoad(r["_source"]) for r in  res['hits']['hits']]
        scroll_size = res['hits']['total']
        while (scroll_size > 0):
            try:
                scroll_id = res['_scroll_id']
                rs = self.es.scroll(scroll_id=scroll_id, scroll=scroll_time)
                results.extend([self.resultPreLoad(r["_source"]) for r in rs['hits']['hits']])
                scroll_size = len(rs['hits']['hits'])
            except:
                break

        return results

    def saveAsCSV(self, filename):
        """
            Wraps elasticsearch querying to enable auto scroll for retrieving
            large amounts of results


        """

        csv_file=open(filename,"w")
        SEPARATOR=u","
        columns=[]
        first=True

        def writeResults(results):
            """
            """
            if len(results) == 0:
                return

            if first:
                columns=results[0]["_source"].keys()
                line=SEPARATOR.join(columns).strip(SEPARATOR)+u"\n"
                csv_file.write(line)

            for result in results:
                line=SEPARATOR.join([unicode(result["_source"][key]) for key in columns]).strip(SEPARATOR)+u"\n"
                csv_file.write(line)

        scroll_time="2m"

        res=self.es.search(
            index=self.index_name,
            doc_type="result",
            size=100,
            search_type="scan",
            scroll=scroll_time,
            body={"query":{"match_all": {}}}
            )

        writeResults(res['hits']['hits'])

        scroll_size = res['hits']['total']
        while (scroll_size > 0):
            try:
                scroll_id = res['_scroll_id']
                rs = self.es.scroll(scroll_id=scroll_id, scroll=scroll_time)
                writeResults(rs['hits']['hits'])
                scroll_size = len(rs['hits']['hits'])
            except:
                break

        csv_file.close()

    def saveAsJSON(self, filename):
        """
            Reads all results into a list, dumps that JSON to a file.

            WARNING! Very large numbers results may incur in MemoryError
        """
        results=self.readResults()
        json.dump(results,file(filename, "w"))

def basicTest():
    """
    """
    rs=ElasticResultStorer("exp1","test_table")
    rs.addResult({"filename":"bla", "position":1, "result":"awesome"})
    rs.addResult({"filename":"bla", "position":2, "result":"awesome"})
    rs.addResult({"filename":"bla", "position":3, "result":"awesome"})
    rs.addResult({"filename":"bla", "position":4, "result":"awesome"})
    rs.saveAsCSV(r"G:\NLP\PhD\pmc_coresc\experiments\pmc_lrec_experiments\test.csv")
    rs.deleteTable()

def main():
    basicTest()
    pass

if __name__ == '__main__':
    main()
