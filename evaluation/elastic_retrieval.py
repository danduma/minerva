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
    def __init__(self, index_name, method, logger=None, use_default_similarity=False):
        self.index_name=index_name
        self.es=Elasticsearch()

        self.method=method # never used?
        self.logger=logger

    def rewriteQueryAsDSL(self, structured_query):
        """
            Creates a multi_match query for elasticsearch
        """
        original_query=structured_query
        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query=structured_query

        lucene_query=""

        for qindex, token in enumerate(structured_query):
            # TODO proper computing of the boost formula. Different methods?
            boost=token["boost"]*token["count"]
            bool=token.get("bool", None) or ""

            lucene_query+=bool+token["token"]
            if boost != 1:
                lucene_query+="\"^"+str(boost)
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

        original_query=structured_query

        if not structured_query or len(structured_query) == 0 :
            return []

        self.last_query=structured_query
        query_text=self.rewriteQuery(structured_query, ["text"])

        res=self.es.search(
            q=query_text,
            size=max_results,
            index=self.index_name,
            doc_type=ES_TYPE_DOC
            )

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

            :param query: Elastic DSL Query
            :param doc_id: id of document to run .explain() for
            :returns:
        """
        explanation=self.es.explain(
            index=self.index_name,
            doc_type=ES_TYPE_DOC,
            query=query,
            id=idoc_idd
            )

        formula=StoredFormula()
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

        if query_text=="":
            print("MAAAC! Empty query!")
            hits=[]
        else:
            try:
                res=self.es.search(
                    q=query_text,
                    size=max_results,
                    index=self.index_name,
                    doc_type=ES_TYPE_DOC,
                    _source=["guid","metadata"]
                    )

                hits=res["hits"]["hits"]
                if len(query_text) > 2800:
                    print("Query > 2800 and no problems! Len: ",len(query_text))
            except ConnectionError as e:
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
                    q=query_text,
                    size=max_results,
                    index=self.index_name,
                    doc_type=ES_TYPE_DOC
                )
                self.logger.logReport(self.searcher.explain(query,index))

        result=[]
        for hit in hits:
            metadata= hit["_source"]["metadata"]
            result.append((hit["_score"],metadata))

        if self.logger and self.logger.full_citation_id in self.logger.citations_extra_info:
            print(query_text,"\n", hits, "\n", result, "\n")

        return result

class precomputedExplainRetrieval(ElasticRetrievalBoost):
    """
        Class that runs the explain pipeline and extracts the data from a lucene
        explanation to then be able to run the same retrieval with different
        field boosts.

        All it does is this one-time retrieval + explanation. The further testing
        of parameters is done in measurePrecomputedResolution()
    """

    def __init__(self, index_path, method, logger=None, use_default_similarity=False):
        ElasticRetrievalBoost.__init__(self, index_path, method, logger, use_default_similarity)

    def precomputeExplain(self, query, parameters, test_guid, max_results=MAX_RESULTS_RECALL):
        """
            Runs elastic retrieval, extracts and stores the explanations in a dict:
            one entry per potential paper, fields to set weights to as sub-entries
        """

        # TODO where does the query come from?
        query=query

        results=[]

        index=0

        # we first run a search, so that we only need to run the explain pipeline
        # for doc that actually match the query. This is just to speed up
        # whole-collection precomputing of retrieval formulas

        hits=self.runQuery(query,parameters,test_guid,max_results)
        doc_list=[hit["_id"] for hit in hits]

        for id in doc_list:
##            explanation=self.es.explain(
##                index=self.index_name,
##                doc_type=ES_TYPE_DOC,
##                query=query,
##                id=id
##                )
##
##            formula=StoredFormula()
##            formula.fromElasticExplanation(explanation)
            self.formulaFromExplanation()

            if formula.formula["coord"] != 0:
                results.append({"index":index,"guid":id,"formula":formula.formula})
        return results


def main():
    pass

if __name__ == '__main__':
    main()
