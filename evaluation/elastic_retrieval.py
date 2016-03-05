# classes that encapsulate all elasticsearch retrieval
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

import re,json,sys
from string import punctuation
from collections import namedtuple

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError

import minerva.db.corpora as cp
from base_retrieval import BaseRetrieval, SPECIAL_FIELDS_FOR_TESTS, MAX_RESULTS_RECALL
from stored_formula import StoredFormula

ES_TYPE_DOC="doc"

class ElasticRetrieval(BaseRetrieval):
    """
        Interfaces with the Elasticsearch API
    """
    def __init__(self, index_name, method, logger=None, use_default_similarity=True):
        self.index_name=index_name
        self.es=Elasticsearch()

        self.method=method # never used?
        self.logger=logger

    def rewriteQueryAsDSL(self, structured_query, parameters):
        """
            Creates a multi_match DSL query for elasticsearch.

            :param structured_query: a StructuredQuery dict, optionally under the
                key "structured_query"
            :param parameters: dict of [field]=weight to replace in the query
        """
        if "structured_query" in structured_query:
            structured_query=structured_query["structured_query"]

        original_query=structured_query
        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query=structured_query

        lucene_query=""

        for qindex, token in enumerate(structured_query):
            # TODO proper computing of the boost formula. Different methods?
            boost=token["boost"]*token["count"]
            bool=token.get("bool", None) or ""

            lucene_query+="%s%s" % (bool,token["token"])
            if boost != 1:
                lucene_query+="^%s" %str(boost)
            lucene_query+=" "

        fields=[]
        for index, param in enumerate(parameters):
            fields.append(param+"^"+str(parameters[param]))

        dsl_query={
          "multi_match" : {
            "query": lucene_query,
            "type":  "best_fields",
            "fields": fields,
            "operator": "or",
##            "tie_breaker": 0.3
          }
        }

        return dsl_query


    def runQuery(self, structured_query, max_results=MAX_RESULTS_RECALL):
        """
            Interfaces with the elasticsearch query API
        """
        if self.useExplainQuery:
            # this is a leftover from the old retrieval. It's unworkable with Elastic.
            raise NotImplementedError
            return

##        original_query=dict(structured_query)

        if not structured_query or len(structured_query) == 0 :
            return []

        self.last_query=dict(structured_query)
        dsl_query=self.rewriteQueryAsDSL(structured_query["structured_query"], ["text"])

        res=self.es.search(
            body={"query":dsl_query},
            size=max_results,
            index=self.index_name,
            doc_type=ES_TYPE_DOC
            )

        structured_query["dsl_query"]=dsl_query
        hits=res["hits"]["hits"]
##        print("Found %d document(s) that matched query '%s':" % (res['hits']['total'], query))

##        if len(hits.scoreDocs) ==0:
##            print "Original query:",original_query
##            print "Query:", query
        result=[]
        for hit in hits:
            metadata= hit["_source"]["metadata"]
            result.append((hit["_score"],metadata))
        return result

    def formulaFromExplanation(self, query, doc_id):
        """
            Runs .explain() for one query/doc pair, generates and returns a \
            StoredFormula instance from it

            :param query: StructuredQuery dict, with a "dsl_query" key
            :param doc_id: id of document to run .explain() for
            :returns:
        """
        explanation=None
        retries=0
        while retries < 2:
            try:
                explanation=self.es.explain(
                    index=self.index_name,
                    doc_type=ES_TYPE_DOC,
                    body={"query":query["dsl_query"]},
                    id=doc_id
                    )
            except:
                retries+=1

        formula=StoredFormula()
        if explanation:
            formula.fromElasticExplanation(explanation)
        return formula

class ElasticRetrievalBoost(ElasticRetrieval):
    """
        Use ElasticSearch for retrieval boosting different (AZ) fields differently
    """

    def __init__(self, index_path, method, logger=None, use_default_similarity=False):
        ElasticRetrieval.__init__(self, index_path, method, logger, use_default_similarity)

    def runQuery(self, structured_query, parameters, test_guid, max_results=MAX_RESULTS_RECALL):
        """
            Run the query, return a list of tuples (score,metadata) of top docs
        """
##        if self.useExplainQuery:
##            # this is a leftover from the old retrieval. It's unworkable with Elastic.
##            raise NotImplementedError
##            return

        if not structured_query or len(structured_query) == 0 :
            return []

        self.last_query=structured_query

        query_text=self.rewriteQuery(structured_query,parameters,test_guid)
        dsl_query=self.rewriteQueryAsDSL(structured_query, parameters)

        if query_text=="":
            print("MAAAC! Empty query!")
            hits=[]
        else:
            try:
                res=self.es.search(
                    body={"query":dsl_query},
                    size=max_results,
                    index=self.index_name,
                    doc_type=ES_TYPE_DOC,
                    _source=["guid","metadata"]
                    )

                hits=res["hits"]["hits"]
                structured_query["dsl_query"]=dsl_query
##                if len(query_text) > 2800:
##                    print("Query > 2800 and no problems! Len: ",len(query_text))
            except ConnectionError as e:
                logging.exception()
                cp.Corpus.global_counters["query_error"]=cp.Corpus.global_counters.get("query_error",0)+1
                print("Query error. Query len: ",len(query_text))
                hits=[]
##                assert False

        # explain the query
        if self.logger:
            self.logger.logReport(query_text+"\n")

            if self.logger.full_citation_id in self.logger.citations_extra_info:
                max_explanations=len(hits)
            else:
                max_explanations=1

            for index in range(max_explanations):
                explanation=self.es.explain(
                    body=dsl_query,
                    size=max_results,
                    index=self.index_name,
                    doc_type=ES_TYPE_DOC
                )
                self.logger.logReport(explanation)

        result=[]
        for hit in hits:
            metadata= hit["_source"]["metadata"]
            result.append((hit["_score"],metadata))

        if self.logger and self.logger.full_citation_id in self.logger.citations_extra_info:
            print(query_text,"\n", hits, "\n", result, "\n")

        return result

def testExplanation():
    """
    """
    from minerva.proc.query_extraction import WindowQueryExtractor

    ext=WindowQueryExtractor()

    er=ElasticRetrieval("papers","",None)

    text="method method MATCH method method sdfakjesf"
    match_start=text.find("MATCH")

    queries=ext.extract({"parameters":[(5,5)],
                         "match_start": match_start,
                         "match_end": match_start+5,
                         "doctext": text,
                         "method_name": "test"
                        })

    q=er.rewriteQueryAsDSL(queries[0]["structured_query"], {"metadata.title":1})
    print(q)

    hits=er.es.search(index="papers", doc_type="paper", body={"query":q}, _source="guid")
    doc_ids=[hit["_id"] for hit in hits["hits"]["hits"]]
    print(doc_ids)
    global ES_TYPE_DOC
    ES_TYPE_DOC="paper"
    formula=er.formulaFromExplanation(q, doc_ids[0])
    print(formula.formula)

def main():
    testExplanation()
    pass

if __name__ == '__main__':
    main()
