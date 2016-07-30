# classes that encapsulate all elasticsearch retrieval
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

import logging, sys

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, TransportError

import minerva.db.corpora as cp
from base_retrieval import BaseRetrieval, SPECIAL_FIELDS_FOR_TESTS, MAX_RESULTS_RECALL
from stored_formula import StoredFormula
from minerva.proc.structured_query import StructuredQuery

ES_TYPE_DOC="doc"
QUERY_TIMEOUT=500 # this is in seconds!

class ElasticRetrieval(BaseRetrieval):
    """
        Interfaces with the Elasticsearch API
    """
    def __init__(self, index_name, method, logger=None, use_default_similarity=True, max_results=None, es_instance=None, save_terms=False):
        self.index_name=index_name
        if es_instance:
            self.es=es_instance
        else:
            if cp.Corpus.__class__.__name__ == "ElasticCorpus":
                self.es=cp.Corpus.es
            else:
                self.es=Elasticsearch()

        if max_results:
            self.max_results=max_results
        else:
            self.max_results=MAX_RESULTS_RECALL

        self.method=method # never used?
        self.logger=logger
        self.last_query={}
        self.save_terms=save_terms
        self.default_field="text"

    def rewriteQueryAsDSL(self, structured_query, parameters):
        """
            Creates a multi_match DSL query for elasticsearch.

            :param structured_query: a StructuredQuery dict, optionally under the
                key "structured_query"
            :param parameters: dict of [field]=weight to replace in the query
        """
        if "structured_query" in structured_query:
            structured_query=structured_query["structured_query"]

        if not isinstance(structured_query,StructuredQuery):
            structured_query=StructuredQuery(structured_query)

        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query=structured_query

        lucene_query=""

        for  token in structured_query:
            # TODO proper computing of the boost formula. Different methods?
##            boost=token["boost"]*token["count"]
            boost=token.boost*token.count
##            bool_val=token.get("bool", None) or ""
            bool_val=token.bool or ""

##            lucene_query+="%s%s" % (bool_val,token["token"])
            lucene_query+="%s%s" % (bool_val,token.token)
            if boost != 1:
                lucene_query+="^%s" %str(boost)
            lucene_query+=" "

        fields=[]
        for param in parameters:
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


    def runQuery(self, structured_query, max_results=None):
        """
            Interfaces with the elasticsearch query API
        """
        if not structured_query or len(structured_query) == 0 :
            return []

        if not max_results:
            max_results=self.max_results

        self.last_query=dict(structured_query)
        dsl_query=self.rewriteQueryAsDSL(structured_query["structured_query"], [self.default_field])

        res=self.es.search(
            body={"query":dsl_query},
            size=max_results,
            index=self.index_name,
            doc_type=ES_TYPE_DOC,
            request_timeout=QUERY_TIMEOUT,
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
                    id=doc_id,
                    request_timeout=QUERY_TIMEOUT,
                    )
                break
            except Exception as e:
                logging.exception("Exception, retrying...")
                retries+=1

        formula=StoredFormula()
        if explanation:
            formula.fromElasticExplanation(explanation, self.save_terms)
        return formula

class ElasticRetrievalBoost(ElasticRetrieval):
    """
        Use ElasticSearch for retrieval boosting different (AZ) fields differently
    """

    def __init__(self, index_name, method, logger=None, use_default_similarity=True, max_results=None, es_instance=None, save_terms=False):
        super(self.__class__,self).__init__(index_name, method, logger, use_default_similarity, max_results, es_instance, save_terms)
        self.return_fields=["guid"]

    def runQuery(self, structured_query, parameters, test_guid, max_results=None):
        """
            Run the query, return a list of tuples (score,metadata) of top docs
        """
        if not structured_query or len(structured_query) == 0 :
            return []

        if not max_results:
            max_results=self.max_results

        self.last_query=structured_query

        query_text=self.rewriteQuery(structured_query,parameters,test_guid)
        dsl_query=self.rewriteQueryAsDSL(structured_query, parameters)

        if query_text=="":
            print("MAAAC! Empty query!")
            hits=[]
        else:
##            assert(False)
            try:
                res=self.es.search(
                    body={"query":dsl_query},
                    size=max_results,
                    index=self.index_name,
                    doc_type=ES_TYPE_DOC,
                    _source=self.return_fields,
                    request_timeout=QUERY_TIMEOUT,
                    )

                hits=res["hits"]["hits"]
                structured_query["dsl_query"]=dsl_query
            except ConnectionError as e:
                logging.exception("Error connecting to ES. Timeout?")
                print("Exception:", sys.exc_info()[:2])
                cp.Corpus.global_counters["query_error"]=cp.Corpus.global_counters.get("query_error",0)+1
                print("Query error. Query len: ",len(query_text))
                hits=[]
            except TransportError as e:
                logging.exception("Error in query:")
                print("Exception:", sys.exc_info()[:2])
                hits=[]

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
##            metadata= hit["_source"]["metadata"]
##            result.append((hit["_score"],{"guid":hit["_source"]["guid"]}))
            result.append((hit["_score"],hit["_source"]))

        if self.logger and self.logger.full_citation_id in self.logger.citations_extra_info:
            print(query_text,"\n", hits, "\n", result, "\n")

        return result

def testExplanation():
    """
    """
    from minerva.proc.query_extraction import WindowQueryExtractor

    ext=WindowQueryExtractor()

##    er=ElasticRetrieval("papers","",None)
    er=ElasticRetrieval("idx_az_annotated_pmc_2013_1","",None)

    text="method method MATCH method method sdfakjesf"
    match_start=text.find("MATCH")

    queries=ext.extract({"parameters":[(5,5)],
                         "match_start": match_start,
                         "match_end": match_start+5,
                         "doctext": text,
                         "method_name": "test"
                        })

##    q=er.rewriteQueryAsDSL(queries[0]["structured_query"], {"metadata.title":1})
##    print(q)

    q={"dsl_query":{'multi_match': {'fields': ['Obj^1',
                            'Res^1',
                            'Goa^1',
                            'Mot^1',
                            'Hyp^1',
                            'Met^1',
                            'Bac^1',
                            'Exp^1',
                            'Con^1',
                            'Obs^1',
                            'Mod^1'],
                 'operator': 'or',
                 'query': u'strongly imaginative via^2 associated coupled communication^2 is^4 repetitive influence mutations one stereotypies as show are point in^3 accounting debate around novo developmental^3 evidence dimensions for centres much separate linked delay genetic^2 difficulties genetically over parental interests activities play loci risk on some genes contribute early independent mediated although^2 asd^5 regression^2 heritable considerable susceptibility^2 processes interaction de third language older whether many age children deficits prior range determined evident social^2 usually narrow whole characterized effects ',
                 'type': 'best_fields'}}}

    doc_ids=['559005ea-9288-4459-a8ef-8ae72ed1dc0f']

##    hits=er.es.search(index="papers", doc_type="paper", body={"query":q}, _source="guid", request_timeout=QUERY_TIMEOUT,)
##    doc_ids=[hit["_id"] for hit in hits["hits"]["hits"]]
##    print(doc_ids)
##    global ES_TYPE_DOC
##    ES_TYPE_DOC="paper"

    formula=er.formulaFromExplanation(q, doc_ids[0])
    print(formula.formula)

def main():
    testExplanation()
    pass

if __name__ == '__main__':
    main()
