# ResultStorer and ResultIncrementalReader: storing and reading results in Elastic
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import json, sys, time, os

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionTimeout, ConnectionError, TransportError
from minerva.proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, RANDOM_ZONES_7, RANDOM_ZONES_11
from minerva.proc.general_utils import ensureDirExists

import minerva.db.corpora as cp

def createResultStorers(exp_name, exp_random_zoning=False, clear_existing_prr_results=False):
    """
        Returns a dict with instances of ElasticResultStorer

        :param exp_name:
        :string
    """
    writers={}
    if exp_random_zoning:
        for div in RANDOM_ZONES_7:
            writers["RZ7_"+div]=ElasticResultStorer(exp_name,"prr_az_rz11", endpoint=cp.Corpus.endpoint)
            if clear_existing_prr_results:
                writers["RZ7_"+div].clearResults()
        for div in RANDOM_ZONES_11:
            writers["RZ11_"+div]=ElasticResultStorer(exp_name,"prr_rz11", endpoint=cp.Corpus.endpoint)
            if clear_existing_prr_results:
                writers["RZ11_"+div].clearResults()
    else:
        for div in AZ_ZONES_LIST:
            writers["az_"+div]=ElasticResultStorer(exp_name,"prr_az_"+div, endpoint=cp.Corpus.endpoint)
            if clear_existing_prr_results:
                writers["az_"+div].clearResults()
        for div in CORESC_LIST:
            writers["csc_type_"+div]=ElasticResultStorer(exp_name,"prr_csc_type_"+div, endpoint=cp.Corpus.endpoint)
            if clear_existing_prr_results:
                writers["csc_type_"+div].clearResults()

    writers["ALL"]=ElasticResultStorer(exp_name,"prr_ALL", endpoint=cp.Corpus.endpoint)
    if clear_existing_prr_results:
        writers["ALL"].clearResults()

    return writers

class ElasticResultStorer(object):
    def __init__(self, namespace, table_name, endpoint={"host":"localhost", "port":9200}):
        """
        """
        assert isinstance(namespace, basestring)
        assert isinstance(table_name, basestring)
        assert namespace != ""

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
##        self.deleteTable()
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

    def getResult(self, res_id):
        """
        """
        attempts=0
        source={}
        while attempts < 2:
            try:
                source=self.es.get(id=res_id,
                                   index=self.index_name,
                                   doc_type="result",
                                   ignore=[404])["_source"]
                break
            except ConnectionError:
                attempts+=1
                time.sleep(attempts+1)
                continue
            except Exception as e:
                print("Exception in ElasticResultStorer.getResult(): ",sys.exc_info()[1])
                break

        return self.resultPreLoad(source)

    def getResultList(self, max_results=sys.maxint):
        """
            Returns a list of all results in the table.
        """
        scroll_time="2m"

        res=self.es.search(
            index=self.index_name,
            doc_type="result",
            size=5000,
            search_type="scan",
            scroll=scroll_time,
            _source=False,
            body={"query":{"match_all": {}}},
            ignore=[404],
            )

        res_ids = [r["_id"] for r in  res['hits']['hits']]
        scroll_size = res['hits']['total']
        while (scroll_size > 0) and len(res_ids) < max_results:
            try:
                scroll_id = res['_scroll_id']
                rs = self.es.scroll(scroll_id=scroll_id, scroll=scroll_time)
                res=rs
                res_ids.extend([r["_id"] for r in rs['hits']['hits']])
                scroll_size = len(rs['hits']['hits'])
            except TransportError as e:
                break
            except Exception as e:
                print("Exception in getResultList():",sys.exc_info()[1])
                break

        return res_ids[:max_results]

    def readResults(self, max_results=None):
        """
        """
        res_ids=self.getResultList(max_results=max_results)
        results=[self.getResult(res_id) for res_id in res_ids]
        return results

    def getResultCount(self):
        """
            Returns the number of results already available
        """
        try:
            return int(self.es.count(index=self.index_name, doc_type="result")["count"])
        except TransportError:
            return 0

    def saveAsCSV(self, filename):
        """
            Wraps elasticsearch querying to enable auto scroll for retrieving
            large amounts of results
        """

        csv_file=open(filename,"w")
        SEPARATOR=u","
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
            body={"query":{"match_all": {}}},
            ignore=[404]
            )

        writeResults(res['hits']['hits'])

        scroll_size = res['hits']['total']
        while scroll_size > 0:
            try:
                scroll_id = res['_scroll_id']
                rs = self.es.scroll(scroll_id=scroll_id, scroll=scroll_time, ignore=[404])
                res=rs
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
        result_ids=self.getResultList()

        f=file(filename, "w")
        f.write("[")
        for index, res_id in enumerate(result_ids):
            res=self.getResult(res_id)
            json_str=json.dumps(res)
            f.write(json_str)
            if index < len(result_ids) -1:
                f.write(",")
        f.write("]")


class ResultIncrementalReader(object):
    """
    """
    def __init__(self, result_storer, res_ids=None, max_results=sys.maxint):
        """
        """
        self.result_storer=result_storer
        if res_ids:
            self.res_ids=res_ids
        else:
            self.res_ids=result_storer.getResultList(max_results=max_results)
        self.bufsize=15
        self.held_from=None
        self.held_to=None
        self.ids_held={}

    def retrieveItem(self, res_id):
        """
        """
        return self.result_storer.getResult(res_id)

    def fillBuffer(self, key):
        """
        """
##        print("fillBuffer", key)
        self.ids_held={}
        if int(key) < len(self.res_ids):
            for cnt in range(min(len(self.res_ids)-int(key),self.bufsize)):
                self.ids_held[key+cnt]=self.retrieveItem(self.res_ids[key+cnt])

    def __getitem__(self, key):
        """
        """
##        print("getitem",key)
        if key not in self.ids_held:
            self.fillBuffer(key)
            return self.ids_held[key]
        else:
            return self.ids_held[key]

    def __iter__(self):
        """
        """
##        print("iter")
        for key in self.res_ids:
            yield self.retrieveItem(key)

    def __len__(self):
##        print("len")
        return len(self.res_ids)

    def subset(self, items):
        """
            Selects a subset of the items it holds, returns a new instance with
            those
        """
        new=ResultIncrementalReader(self.result_storer, [self.res_ids[i] for i in items])
        return new


class ResultDiskReader(ResultIncrementalReader):
    """
        Like ResultIncrementalReader but it keeps a local cache of everything to avoid
        having to call Elastic every few results
    """
    def __init__(self, result_storer, cache_dir, res_ids=None, max_results=sys.maxint):
        """
            Creates cache directory if it doesn't exist
        """
        super(self.__class__, self).__init__(result_storer, res_ids=res_ids, max_results=max_results)
        self.cache_dir=cache_dir
        self.own_dir=os.path.join(cache_dir, self.result_storer.table_name)
        ensureDirExists(cache_dir)
        ensureDirExists(self.own_dir)

    def retrieveItem(self, res_id):
        """
        """
        res_path=os.path.join(self.own_dir,res_id+".json")
        if os.path.exists(res_path):
            return json.load(file(res_path,"r"))
        else:
            res=self.result_storer.getResult(res_id)
            json.dump(res,file(res_path,"w"))
            return res

    def subset(self, items):
        """
            Selects a subset of the items it holds, returns a new instance with
            those
        """
        new=ResultDiskReader(self.result_storer, self.cache_dir, res_ids=[self.res_ids[i] for i in items])
        return new


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

def dumpResultsToDisk():
    from minerva.proc.nlp_functions import CORESC_LIST
    for zone in CORESC_LIST:
        rs=ElasticResultStorer("pmc_lrec_experiments","prr_csc_type_"+zone)
        rs.saveAsJSON(r"g:\NLP\PhD\pmc_coresc\experiments"+"\\" + "prr_csc_type_"+zone+".json")

def main():
##    basicTest()


    pass

if __name__ == '__main__':
    main()
