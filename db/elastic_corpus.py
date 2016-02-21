# ElasticCorpus. Uses an ElasticSearch endpoint as the database of papers
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import os, sys, json, glob, uuid, unicodedata, datetime, re

from elasticsearch import Elasticsearch

from minerva.proc.general_utils import AttributeDict, ensureTrailingBackslash, ensureDirExists
from minerva.scidoc.scidoc import SciDoc
from base_corpus import BaseCorpus

ES_INDEX_PAPERS="papers"
ES_INDEX_SCIDOCS="scidocs"
ES_INDEX_CACHE="cache"
ES_INDEX_AUTHORS="authors"
ES_INDEX_LINKS="links"
ES_INDEX_VENUES="venues"
ES_INDEX_MISSING_REFERENCES="missing_references"

ES_TYPE_PAPER="paper"
ES_TYPE_SCIDOC="scidoc"
ES_TYPE_CACHE="cache"
ES_TYPE_AUTHOR="author"
ES_TYPE_LINK="link"
ES_TYPE_VENUE="venue"
ES_TYPE_MISSING_REFERENCES="missing_reference"

index_equivalence={
    "papers":{"index":ES_INDEX_PAPERS,"type":ES_TYPE_PAPER, "source":"metadata"},
    "scidocs":{"index":ES_INDEX_SCIDOCS,"type":ES_TYPE_SCIDOC, "source":"scidoc"},
    "authors":{"index":ES_INDEX_AUTHORS,"type":ES_TYPE_AUTHOR, "source":"author"},
    "cache":{"index":ES_INDEX_CACHE,"type":ES_TYPE_CACHE, "source":"data"},
    "links":{"index":ES_INDEX_LINKS,"type":ES_TYPE_LINK, "source":"link"},
    "venues":{"index":ES_INDEX_VENUES,"type":ES_TYPE_VENUE, "source":"venue"},
    "missing_references":{"index":ES_INDEX_MISSING_REFERENCES,"type":ES_TYPE_MISSING_REFERENCES, "source":"missing_reference"},
}

ES_ALL_INDECES=[ES_INDEX_PAPERS, ES_INDEX_SCIDOCS, ES_INDEX_CACHE,
                ES_INDEX_LINKS, ES_INDEX_AUTHORS]

class ElasticCorpus(BaseCorpus):
    """
        ElasticSearch connection corpus
    """
    def __init__(self):
        """
            Basic creator
        """
        super(self.__class__, self).__init__()
        self.es=None

        self.query_filter=""

        self.ALL_FILES=[]
        self.TEST_FILES=[]
        self.FILES_TO_IGNORE=[]
        self.metadata_index=None
        self.paths.fullLuceneIndex="index_"

    def connectCorpus(self, base_directory, endpoint={"host":"localhost", "port":9200}, initializing_corpus=False,suppress_error=False):
        """
            If DB has been created, connect to it. If not, initialize it first.

            Args:
                base_directory: root dir of this corpus
                initializing_corpus: if True, create DB and directories
                suppress_error: if true, db doesn't complain if it's connected already
        """
        self.endpoint=endpoint
        self.setPaths(ensureTrailingBackslash(base_directory))

        if initializing_corpus:
            self.createAndInitializeDatabase()
        self.connectToDB(suppress_error)

    def createAndInitializeDatabase(self):
        """
            Ensures that the directory structure is in place and creates
            the SQLite database and tables
        """
        settings={
            "number_of_shards" : 3,
            "number_of_replicas" : 3
        }

        properties={
            "guid": {"type":"string", "index":"not_analyzed"},
            "metadata": {"type":"object"},
            "norm_title": {"type":"string", "index":"not_analyzed"},
            "author_ids":{"type":"string", "index":"not_analyzed", "store":True},
            "num_in_collection_references": {"type":"integer"},
            "num_resolvable_citations": {"type":"integer"},
            "num_inlinks": {"type":"integer"},
            "collection_id": {"type":"string", "index":"not_analyzed", "store":True},
            "import_id": {"type":"string", "index":"not_analyzed", "store":True},
            "time_created": {"type":"date"},
            "time_modified": {"type":"date"},
            "has_scidoc": {"type":"boolean","index":"not_analyzed", "store":True},
            "flags": {"type":"string","index":"not_analyzed", "store":True},
            # This is all now accessed through the nested metadata
##            "filename": {"type":"string", "index":"not_analyzed", "store":True},
##            "corpus_id": {"type":"string", "index":"not_analyzed"},
##            "title": {"type":"string", "store":True},##            "surnames": {"type":"string"},
##            "year": {"type":"integer"},
##            "in_collection_references": {"type":"string", "index":"not_analyzed", "store":True},
##            "inlinks": {"type":"string", "index":"not_analyzed", "store":True},
            }

        if not self.es.indices.exists(index=ES_INDEX_PAPERS):
            self.es.indices.create(
                index=ES_INDEX_PAPERS,
                body={"settings":settings,"mappings":{ES_TYPE_PAPER:{"properties":properties}}})

        if not self.es.indices.exists(index=ES_INDEX_SCIDOCS):
            properties={
                "scidoc": {"type":"string", "index": "no", "store":True},
                "time_created": {"type":"date"},
                "time_modified": {"type":"date"},
                }
            self.es.indices.create(
                index=ES_INDEX_SCIDOCS,
                body={"settings":settings,"mappings":{ES_TYPE_SCIDOC:{"properties":properties}}})

        settings={
            "number_of_shards" : 1,
            "number_of_replicas" : 1
        }

        if not self.es.indices.exists(index=ES_INDEX_CACHE):
            properties={
                "data": {"type":"string", "index": "no", "store":True},
                "time_created": {"type":"date"},
                "time_modified": {"type":"date"},
                }
            self.es.indices.create(
                index=ES_INDEX_CACHE,
                body={"settings":settings,"mappings":{ES_TYPE_CACHE:{"properties":properties}}})

        if not self.es.indices.exists(index=ES_INDEX_LINKS):
            properties={
                "link":{"type":"nested"},
##                "guid_from": {"type":"string", "index":"not_analyzed", "store":True},
##                "guid_to": {"type":"string", "index":"not_analyzed", "store":True},
##                "authors_from": {"type":"string", "index":"not_analyzed", "store":True},
##                "authors_to": {"type":"string", "index":"not_analyzed", "store":True},
##                "self_citation": {"type":"boolean", "index":"not_analyzed", "store":True},
##                "year_from": {"type":"integer", "index":"not_analyzed", "store":True},
##                "year_to": {"type":"integer", "index":"not_analyzed", "store":True},
##                "numcitations": {"type":"integer", "index":"not_analyzed", "store":True},
                "time_created": {"type":"date"}}

            self.es.indices.create(
                index=ES_INDEX_LINKS,
                body={"settings":settings,"mappings":{ES_TYPE_LINK:{"properties":properties}}})

        if not self.es.indices.exists(index=ES_INDEX_AUTHORS):
            properties={
##                "author_id": {"type":"string", "index":"not_analyzed", "store":True},
                "author": {"type":"nested"},
##                "given": {"type":"string", "index":"analyzed", "store":True},
##                "middle": {"type":"string", "index":"analyzed", "store":True},
##                "family": {"type":"string", "index":"analyzed", "store":True},
##                "papers": {"type":"string", "index":"not_analyzed", "store":True},
##                "papers_first_author": {"type":"string", "index":"not_analyzed", "store":True},
##                "papers_last_author": {"type":"string", "index":"not_analyzed", "store":True},
##                "affiliations": {"type":"nested", "index":"not_analyzed", "store":True},
                "time_created": {"type":"date"}
                }

            self.es.indices.create(
                index=ES_INDEX_AUTHORS,
                body={"settings":settings,"properties":properties})


        if not self.es.indices.exists(index=ES_INDEX_VENUES):
            properties={
                "venue": {"type":"nested"},
                "time_created": {"type":"date"}
                }

            self.es.indices.create(
                index=ES_INDEX_VENUES,
                body={"settings":settings,"properties":properties})


    def connectedToDB(self):
        """
            returns True if connected to DB, False otherwise
        """
        return self.es is not None

    def getRetrievalIndexPath(self, guid, index_filename, full_corpus=False):
        """
            Returns the path to the Lucene index for a test file in
            the corpus

            if full_corpus is True, this is the general index for that method
                else
            when using Citation Resolution (resolving only from the
            references at the bottom of a paper) it is the specific index for
            that file guid
        """
        guid=guid.lower()
        if full_corpus:
            return "idx_"+index_filename
##            return index_filename
        else:
            return "idx_"+guid+"_"+index_filename

    def getRecord(self, id, table="papers"):
        """
            Abstracts over getting data from a row in the db. Returns one field
            for one type of table.

            All other "getter" functions like getMetadataByGUID and loadSciDoc
            are aliases for this function
        """
        self.checkConnectedToDB()

        if record_type not in index_equivalence:
            raise ValueError("Unknown record type")

        res=self.es.get(
            index=index_equivalence[table]["index"],
            doc_type=index_equivalence[table]["type"],
            id=id,
            _source=index_equivalence[table]["source"]
            )

        if not res:
            raise IndexError("Can't find record with id %s" % id)
        return res["_source"]["data"]

    def cachedJsonExists(self, type, guid, params=None):
        """
            True if the cached JSON associated with the given parameters exists
        """
        self.checkConnectedToDB()

        return self.es.exists(
            index=ES_INDEX_CACHE,
            doc_type=ES_TYPE_CACHE,
            id=self.cachedDataIDString(type, guid, params)
            )

    def saveCachedJson(self, path, data):
        """
            Save anything as JSON

            :param path: unique ID of resource to load
            :param data: json-formatted string or any data
        """
        self.checkConnectedToDB()

        timestamp=datetime.datetime.now()
        self.es.index(
            index=ES_INDEX_CACHE,
            doc_type=ES_TYPE_CACHE,
            id=path,
            op_type="index",
            body={
                "data": data,
                "time_created": timestamp,
                "time_modified": timestamp,
                }
            )

    def loadCachedJson(self,path):
        """
            Load precomputed JSON

            :param path: unique ID of resource to load
        """
        return self.getRecord(path,"cache")

    def loadSciDoc(self,guid):
        """
            If a SciDocJSON file exists for guid, it returns it, otherwise None
        """
        return self.getRecord(guid,"papers")

    def saveSciDoc(self,doc):
        """
            Saves the document as JSON in the index
        """
        self.checkConnectedToDB()

        attempts=0
        success=False

        while attempts < 3 and not success:
            try:
                timestamp=datetime.datetime.now()
                self.es.index(
                    index=ES_INDEX_SCIDOCS,
                    doc_type=ES_TYPE_SCIDOC,
                    id=doc["metadata"]["guid"],
                    op_type="index",
                    body={
                        "scidoc": doc.data,
                        "time_created": timestamp,
                        "time_modified": timestamp,
                        }
                    )
                success=True
            except ConnectionTimeout:
                attempts+=1

    def connectToDB(self, suppress_error=False):
        """
            Connects to database
        """
        self.es = Elasticsearch([self.endpoint])
        self.es.retry_on_timeout=True

    def getMetadataByGUID(self,guid):
        """
            Returns a paper's metadata by GUID
        """
        return self.getRecord(guid, "papers")

    def getMetadataByField(self,field,value):
        """
            Returns a paper's metadata by any other field
        """
        assert self.connectedToDB(), "Not connected to DB!"

        query=self.filterQuery("%s:\"%s\"" % (field,value))

        res=self.es.search(
            index=ES_INDEX_PAPERS,
            doc_type=ES_TYPE_PAPER,
            q=query)

        hits=res["hits"]["hits"]
        if len(hits) == 0:
            return None

        return hits[0]["_source"]

    def filterQuery(self, query):
        """
            Adds a global filter to the query so it only matches the selected
            collection, date, etc.

            :param query: string
        """
##        query=re.sub( query.replace(">=",":")
        return self.query_filter+" "+query


    def listFieldByField(self,field1,field2,value,table="papers"):
        """
            Returns a list: for each paper, field1 if field2==value
        """
        self.checkConnectedToDB()

        query=self.filterQuery("%s:\"%s\"" % (field2,value))

        hits=self.unlimitedQuery(
                q=query,
                index=index_equivalence[table]["index"],
                doc_type=index_equivalence[table]["type"],
                _source=field1,
        )

        return [r["_source"][field1] for r in hits]

    def listPapers(self,conditions,field="guid"):
        """
            Return a list of GUIDs in papers table where [conditions]
        """
        self.checkConnectedToDB()

        query=self.filterQuery("guid:* AND (%s)" % conditions)

        hits=self.unlimitedQuery(
                q=query,
                index=ES_INDEX_PAPERS,
                doc_type=ES_TYPE_PAPER,
                _source=field,
        )

        return [r["_source"][field] for r in hits]

    def listAllPapers(self,field="guid"):
        """
            Return a list of ALL GUIDs of all papers in papers table

            :param field: which field to return. Default: `guid`
            :type field:string
            :return: list
        """
        assert self.connectedToDB(), "Not connected to DB!"

        hits=self.unlimitedQuery(
                    index=ES_INDEX_PAPERS,
            doc_type=ES_TYPE_PAPER,
            _source=field,
            q="guid:*")

##        res=self.es.search(
##            index=ES_INDEX_PAPERS,
##            doc_type=ES_TYPE_PAPER,
##            _source=field,
##            q="guid:*")
##        hits=res["hits"]["hits"]
        return [r["_source"][field] for r in hits]

    def runSingleValueQuery(self,query):
        raise NotImplementedError

    def addAuthor(self, author):
        """
            Make sure author is in database
        """
        self.checkConnectedToDB()

        author["author_id"]=self.generateAuthorID
        self.updateAuthor(author,"create")

    def updateAuthorsFromPaper(self, metadata):
        """
            Make sure authors are in database

            :param metadata: a paper's metadata, with an "authors" key
        """
        self.checkConnectedToDB()

        for index, author in enumerate(metadata["authors"]):
            author_record=self.matcher.matchAuthor(author)
            if not author_record:
                author_record=copy.deepcopy(author)
                author_record["papers"]=[]

            if metadata["guid"] not in author_record["papers"]:
                author_record["papers"].append(metadata["guid"])
                if index==0:
                    author_record["papers_first_author"].append(metadata["guid"])
                if index==len(metadata["authors"]):
                    author_record["papers_last_author"].append(metadata["guid"])

            updateAuthor(author)

    def updateAuthor(self, author, op_type="index"):
        """
            Updates an existing author in the db

            :param author: author data
            :param op_type: one of ["index", "create"]
        """
        self.checkConnectedToDB()

        timestamp=datetime.datetime.now()

        body={
            "author":author,
        }

        author["time_updated"]=timestamp

        if op_type=="create":
            body["time_created"]=timestamp

        self.es.index(
            index=ES_INDEX_AUTHORS,
            doc_type=ES_TYPE_AUTHOR,
            op_type=op_type,
            id=author["author_id"],
            body=body
            )

    def addPaper(self, metadata, check_existing=True, has_scidoc=True):
        """
            Add paper metadata to database
        """
        op_type="create" if check_existing else "index"
        self.updatePaper(metadata, op_type, has_scidoc)
        self.updateAuthorsFromPaper(metadata)

    def updatePaper(self, metadata, op_type="index", has_scidoc=True):
        """
            Updates an existing record in the db

            :param metadata: metadata of paper
            :param op_type: one of ["index", "create"]
            :param has_scidoc: True if SciDoc for this paper exists in scidocs \
                index, False otherwise
        """
        self.checkConnectedToDB()

        timestamp=datetime.datetime.now()
        body={"guid": metadata["guid"],
                "metadata": metadata,
                "norm_title": metadata["norm_title"],
                "num_in_collection_references": metadata["num_in_collection_references"],
                "num_resolvable_citations": metadata["num_resolvable_citations"],
                "num_inlinks": len(metadata["inlinks"]),
                "time_modified": timestamp,
                "has_scidoc": has_scidoc
##                "corpus_id": metadata["corpus_id"],
##                "filename": metadata["filename"],
##                 "collection_id": metadata["collection_id"],
##                "import_id": metadata["import_id"],
##                "title": metadata["title"],
##                "surnames": metadata["surnames"],
##                "year": metadata["year"],
                  }

        if op_type=="create":
            body["time_created"]=timestamp

        self.es.index(
            index=ES_INDEX_PAPERS,
            doc_type=ES_TYPE_PAPER,
            op_type=op_type,
            id=metadata["guid"],
            body=body
            )


    def addLink(self,GUID_from,GUID_to,authors_from,authors_to,year_from,year_to,numcitations):
        """
            Add a link in the citation graph.
        """
        self.checkConnectedToDB()

        self.es.create(
            index=ES_INDEX_LINKS,
            doc_type=ES_TYPE_LINK,
            body={
                "guid_from": GUID_from,
                "guid_to": GUID_to,
                "authors_from": authors_from,
                "authors_to": authors_from,
                "year_from": year_from,
                "year_to": year_to,
                "numcitations": numcitations,
                "time_created": datetime.datetime.now(),
            })

    def addMissingPaper(self, metadata):
        """
            Inserts known data about a paper with no SciDoc
        """
##        self.addPaper(metadata,check_existing=True,has_scidoc=False)
        raise NotImplementedError

    def createDBindeces(self):
        """
            Call this after importing the metadata into the corpus and before
            matching in-collection references, it should speed up search
        """
        self.checkConnectedToDB()

        for index in ES_ALL_INDECES:
            if self.es.indeces.exists(index=index):
                self.es.optimize(index=index)
        pass

    def deleteAll(self, record_type):
        """
            WARNING! This function deletes all the records in a given "table" or
            of a given type.

            :param record_type: one of ["papers","links","authors","scidocs","cache"]

        """
        self.checkConnectedToDB()

        if record_type not in index_equivalence:
            raise ValueError("Unknown record type")

        es_table=index_equivalence[record_type]["index"]
        es_type=index_equivalence[record_type]["type"]

        if self.es.indices.exists(index=es_table):
            print("Deleting ALL files in %s" % es_table)
            # ignore 404 and 400
            self.es.indices.delete(index=es_table, ignore=[400, 404])
            self.createAndInitializeDatabase()

    def deleteByQuery(self, record_type, query):
        """
            Delete the entries from a table that match the query.

            :param record_type: one of the tables that exist, e.g. ["papers","links","authors","scidocs","cached"]
            :type record_type: string
            :param query: a query to select documents to delete
            :type query: string
        """
        es_table=index_equivalence[record_type]["index"]
        es_type=index_equivalence[record_type]["type"]

        to_delete=self.unlimitedQuery(
            index=es_table,
            doc_type=es_type,

            q=query)

        bulk_commands=[]
        for item in to_delete:
            bulk_commands.append( "{ \"delete\" : {  \"_id\" : \"%s\" } }" % item["_id"] )

        if len(bulk_commands) > 0:
            self.es.bulk(
                body="\n".join(bulk_commands),
                index=es_table,
                doc_type=es_type,
            )

    def unlimitedQuery(self, *args, **kwargs):
        """
            Wraps elasticsearch querying to enable auto scroll for retrieving
            large amounts of results

            It does more or less what elasticsearch.helpers.scan does, only this
            one actually works.
        """
        scroll_time="2m"

        res=self.es.search(
            *args,
            size=10000,
            search_type="scan",
            scroll=scroll_time,
            **kwargs
            )

        results = []
        scroll_size = res['hits']['total']
        while (scroll_size > 0):
            try:
                scroll_id = res['_scroll_id']
                rs = self.es.scroll(scroll_id=scroll_id, scroll=scroll_time)
                results += rs['hits']['hits']
                scroll_size = len(rs['hits']['hits'])
            except:
                break

        return results

    def setCorpusFilter(self, collection_id=None, import_id=None, date=None):
        """
            Sets the filter query to limit all queries to a collection (corpus)
            or an import date

            :param collection: identifier of corpus, e.g. "ACL" or "PMC". This is set at import time.
            :type collection:basestring
            :param import: identifier of import, e.g. "initial"
            :type import:basestring
            :param date: comparison with a date, e.g. ">[date]", "<[date]",
            :type collection:basestring
        """
        query_items=[]
        if collection_id:
            query_items.append("collection_id:\"%s\"" % collection_id)
        if import_id:
            query_items.append("import:\"%s\"" % import_id)
        if date:
            query_items.append("time_created:%s" % date)

        self.query_filter=" AND ".join(query_items)+" AND "

DOCTEST = True

if __name__ == '__main__':

    if DOCTEST:
        import doctest
        doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
