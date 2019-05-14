# classes that encapsulate all elasticsearch retrieval
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

from __future__ import absolute_import
import logging, sys

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, TransportError

import db.corpora as cp
from .base_retrieval import BaseRetrieval, SPECIAL_FIELDS_FOR_TESTS, MAX_RESULTS_RECALL
from .stored_formula import StoredFormula
from proc.structured_query import StructuredQuery
from six.moves import range
import time

import json

ES_TYPE_DOC = "doc"
QUERY_TIMEOUT = 500  # this is in seconds!


class ElasticRetrieval(BaseRetrieval):
    """
        Interfaces with the Elasticsearch API
    """

    def __init__(self, index_name, method, logger=None, use_default_similarity=True, max_results=None, es_instance=None,
                 save_terms=False, multi_match_type=None):
        self.index_name = index_name
        if es_instance:
            self.es = es_instance
        else:
            if cp.Corpus.__class__.__name__ == "ElasticCorpus":
                self.es = cp.Corpus.es
            else:
                self.es = Elasticsearch(timeout=QUERY_TIMEOUT)

        if not cp.Corpus.isIndexOpen(self.index_name):
            try:
                self.es.indices.open(self.index_name)
                time.sleep(10)
            except TransportError as e:
                print(e)

        if max_results:
            self.max_results = max_results
        else:
            self.max_results = MAX_RESULTS_RECALL

        self.method = method  # never used!
        self.logger = logger
        self.last_query = {}
        self.save_terms = save_terms
        self.default_field = "text"
        self.tie_breaker = 0
        if not multi_match_type:
            self.multi_match_type = "best_fields"
        else:
            self.multi_match_type = multi_match_type

    def rewriteQueryAsDSL1(self, structured_query, parameters):
        """
            Creates a multi_match DSL query for elasticsearch.

            :param structured_query: a StructuredQuery dict, optionally under the
                key "structured_query"
            :param parameters: dict of [field]=weight to replace in the query
        """
        if "structured_query" in structured_query:
            structured_query = structured_query["structured_query"]

        if not isinstance(structured_query, StructuredQuery):
            structured_query = StructuredQuery(structured_query)

        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query = structured_query

        lucene_query = ""

        for token in structured_query:
            # TODO proper computing of the boost formula. Different methods?
            t_boost = token.boost
            t_count = token.count

            if t_boost is None:
                print("NULL! ")
                print(token, token.boost, token.count)
                t_boost = 0
            if t_count is None:
                print("NULL! ")
                print(token, token.boost, token.count)
                t_count = 0

            boost = t_boost * t_count

            if boost == 0.0:
                continue

            bool_val = token.bool or ""

            token_text = token.token
            if " " in token_text:  # if token is a phrase
                token_text = "\"" + token_text + "\""

            lucene_query += "%s%s " % (bool_val, token_text)
            ##            if boost != 1:
            ##                lucene_query+="^%s" %str(boost)

            if boost != 1:
                token_str = token_text + " "
                lucene_query += bool_val + (token_str * int(boost - 1))

            lucene_query = lucene_query.strip()
            lucene_query += " "

        lucene_query = lucene_query.replace("  ", " ")

        fields = []
        for param in parameters:
            fields.append(param + "^" + str(parameters[param]))

        dsl_query = {
            "multi_match": {
                "query": lucene_query,
                "type": self.multi_match_type,
                "fields": fields,
                "operator": "or",
            }
        }

        ##        print(dsl_query)

        if self.tie_breaker:
            dsl_query["multi_match"]["tie_breaker"] = self.tie_breaker

        return dsl_query

    def rewriteQueryAsDSL2(self, structured_query, parameters):
        """
            Creates a multi_match DSL query for elasticsearch. Version 2

            :param structured_query: a StructuredQuery dict, optionally under the
                key "structured_query"
            :param parameters: dict of [field]=weight to replace in the query
        """
        if "structured_query" in structured_query:
            structured_query = structured_query["structured_query"]

        if not isinstance(structured_query, StructuredQuery):
            structured_query = StructuredQuery(structured_query)

        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query = structured_query

        lucene_query = ""

        for token in structured_query:
            boost = token.boost * token.count
            bool_val = token.bool or ""

            token_text = token.token
            if " " in token_text:  # if token is a phrase
                token_text = "\"" + token_text + "\""

            lucene_query += "%s%s " % (bool_val, token_text)

            if boost != 1:
                token_str = token_text + " "
                lucene_query += bool_val + (token_str * int(boost - 1))

            lucene_query = lucene_query.strip()
            lucene_query += " "

        elastic_query = {"bool": {"should": []}}

        fields = []
        for param in parameters:
            fields.append(param + "^" + str(parameters[param]))

        dsl_query = {
            "multi_match": {
                "query": lucene_query,
                "type": self.multi_match_type,
                "fields": fields,
                "operator": "or",
            }
        }

        ##        print(dsl_query)

        if self.tie_breaker:
            dsl_query["multi_match"]["tie_breaker"] = self.tie_breaker

        return dsl_query

    def rewriteQueryAsDSL(self, structured_query, parameters):
        """
            Creates a DSL query for elasticsearch. Version 3, uses individual "term" and "match" queries

            :param structured_query: a StructuredQuery dict, optionally under the
                key "structured_query"
            :param parameters: dict of [field]=weight to replace in the query
        """
        if isinstance(structured_query, dict) and "structured_query" in structured_query:
            structured_query = structured_query["structured_query"]

        if not isinstance(structured_query, StructuredQuery):
            structured_query = StructuredQuery(structured_query)

        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query = structured_query

        field_dicts = []

        for token in structured_query:
            # TODO proper computing of the boost formula. Different methods?
            boost = token.boost * token.count
            bool_val = token.bool or ""

            token_text = token.token
            # if " " in token_text:  # if token is a phrase
            #     token_text = "\"" + token_text + "\""

            if boost == 0.0:
                continue

            for field in parameters:
                if " " in token_text:
                    new_dict = {"match_phrase": {
                        field: {"query": token_text,
                                "boost": parameters[field] * boost},
                    }}

                else:
                    new_dict = {"term": {
                        field: {"value": token_text,
                                "boost": parameters[field] * boost},
                    }}

                field_dicts.append(new_dict)

        fields = []
        for param in parameters:
            fields.append(param + "^" + str(parameters[param]))

        dsl_query = {
            "bool": {
                "should": field_dicts
            }
        }

        return dsl_query

    def runQuery(self, structured_query, max_results=None):
        """
            Interfaces with the elasticsearch query API
        """
        if not structured_query or len(structured_query) == 0:
            return []

        if not max_results:
            max_results = self.max_results

        self.last_query = dict(structured_query)
        dsl_query = self.rewriteQueryAsDSL(structured_query["structured_query"], [self.default_field])

        res = self.es.search(
            body={"query": dsl_query},
            size=max_results,
            index=self.index_name,
            doc_type=ES_TYPE_DOC,
            request_timeout=QUERY_TIMEOUT,
        )

        structured_query["dsl_query"] = dsl_query
        hits = res["hits"]["hits"]
        ##        print("Found %d document(s) that matched query '%s':" % (res['hits']['total'], query))

        ##        if len(hits.scoreDocs) ==0:
        ##            print "Original query:",original_query
        ##            print "Query:", query
        result = []
        for hit in hits:
            metadata = hit["_source"]["metadata"]
            result.append((hit["_score"], metadata))
        return result

    def formulaFromExplanation(self, query, doc_id):
        """
            Runs .explain() for one query/doc pair, generates and returns a \
            StoredFormula instance from it

            :param query: StructuredQuery dict, with a "dsl_query" key
            :param doc_id: id of document to run .explain() for
            :returns:
        """
        explanation = None
        retries = 0
        while retries < 1:
            try:
                explanation = self.es.explain(
                    index=self.index_name,
                    doc_type=ES_TYPE_DOC,
                    body={"query": query["dsl_query"]},
                    id=doc_id,
                    request_timeout=QUERY_TIMEOUT,
                )
                break
            except Exception as e:
                ##                logging.error("Exception, retrying...")
                retries += 1

        if retries > 0:
            if retries == 1:
                logging.error("Retried {} times, failed to retrieve.".format(retries + 1))
            else:
                logging.warning("Retried %d times, retrieved successfuly." % (retries + 1))

        formula = StoredFormula()
        if explanation:
            formula.fromElasticExplanation(explanation, self.save_terms)
        return formula


class ElasticRetrievalBoost(ElasticRetrieval):
    """
        Use ElasticSearch for retrieval boosting different (AZ) fields differently
    """

    def __init__(self, index_name, method, logger=None, use_default_similarity=True, max_results=None, es_instance=None,
                 save_terms=False, multi_match_type=None):
        super(ElasticRetrievalBoost, self).__init__(index_name, method, logger, use_default_similarity, max_results,
                                                    es_instance, save_terms, multi_match_type)
        self.return_fields = ["guid"]

    def runQuery(self, structured_query, parameters=None, test_guid=None, max_results=None):
        """
            Run the query, return a list of tuples (score,metadata) of top docs

            :param structured_query: StructuredQuery or equivalent list
            :param parameters: dict with [key]=weight for searching different fields w/different weights
            :param test_guid: the GUID of the file that we've extracted the queries from
        """
        if not structured_query or len(structured_query) == 0:
            return []

        if not max_results:
            max_results = self.max_results

        self.last_query = structured_query

        query_text = self.rewriteQuery(structured_query, parameters, test_guid, include_field=False)
        dsl_query = self.rewriteQueryAsDSL(structured_query, parameters)

        # if structured_query["file_guid"] == 'b884c939-144c-4d95-9d30-097f8b83e1d3' and structured_query[
        #     "citation_id"] == 'cit9':
        #     print()
        #     print(json.dumps(dsl_query, indent=3))
        #     print()

        # simple_query = {
        #     "simple_query_string": {
        #         "query": query_text,
        #         # "analyzer": "snowball",
        #         "fields": [x for x in parameters.keys()],
        #         "default_operator": "or"
        #     }
        # }
        # print(query_text)
        retries = 0
        MAX_RETRIES = 2
        while not retries >= MAX_RETRIES:
            if query_text == "" or query_text is None:
                print("MAAAC! Empty query!")
                hits = []
                break
            else:
                ##            assert(False)
                try:
                    res = self.es.search(
                        body={"query": dsl_query},
                        # body={"query": simple_query},
                        size=max_results,
                        index=self.index_name,
                        doc_type=ES_TYPE_DOC,
                        _source=self.return_fields,
                        request_timeout=QUERY_TIMEOUT,
                    )

                    hits = res["hits"]["hits"]
                    structured_query["dsl_query"] = dsl_query
                    break
                except ConnectionError as e:
                    logging.exception("Error connecting to ES. Timeout?")
                    print("Exception:", sys.exc_info()[:2])
                    cp.Corpus.global_counters["query_error"] = cp.Corpus.global_counters.get("query_error", 0) + 1
                    print("Query error. Query len: ", len(query_text))
                    hits = []
                    retries += 1
                    if retries < MAX_RETRIES:
                        print("Retrying...")
                except TransportError as e:
                    if not cp.Corpus.isIndexOpen(self.index_name):
                        try:
                            self.es.indices.open(self.index_name)
                            time.sleep(10)
                        except TransportError as e:
                            print(e)

                        continue

                    logging.error("Error in query: " + str(dsl_query))
                    print("Exception:", sys.exc_info()[:2])
                    hits = []
                    retries += 1
                    if retries < MAX_RETRIES:
                        print("Retrying...")

        # explain the query
        if self.logger:
            self.logger.logReport(query_text + "\n")

            if self.logger.full_citation_id in self.logger.citations_extra_info:
                max_explanations = len(hits)
            else:
                max_explanations = 1

            for index in range(max_explanations):
                explanation = self.es.explain(
                    body=dsl_query,
                    size=max_results,
                    index=self.index_name,
                    doc_type=ES_TYPE_DOC
                )
                self.logger.logReport(explanation)

        result = []
        for hit in hits:
            ##            metadata= hit["_source"]["metadata"]
            ##            result.append((hit["_score"],{"guid":hit["_source"]["guid"]}))
            result.append((hit["_score"], hit["_source"]))

        if self.logger and self.logger.full_citation_id in self.logger.citations_extra_info:
            print(query_text, "\n", hits, "\n", result, "\n")

        return result


class ElasticRetrievalBoostTweaked(ElasticRetrievalBoost):

    def __init__(self, index_name, method, logger=None, use_default_similarity=True, max_results=None, es_instance=None,
                 save_terms=False, multi_match_type=None):
        super(ElasticRetrievalBoostTweaked, self).__init__(index_name, method, logger, use_default_similarity,
                                                           max_results,
                                                           es_instance, save_terms, multi_match_type)

    def rewriteQueryAsDSL(self, structured_query, parameters):
        """
            Creates a multi_match DSL query for elasticsearch.

            :param structured_query: a StructuredQuery dict, optionally under the
                key "structured_query"
            :param parameters: dict of [field]=weight to replace in the query
        """
        if "structured_query" in structured_query:
            structured_query = structured_query["structured_query"]

        if not isinstance(structured_query, StructuredQuery):
            structured_query = StructuredQuery(structured_query)

        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query = structured_query

        lucene_query = ""
        field_dicts = []

        for token in structured_query:
            # TODO proper computing of the boost formula. Different methods?
            boost = token.boost * token.count
            bool_val = token.bool or ""

            token_text = token.token
            if " " in token_text:  # if token is a phrase
                token_text = "\"" + token_text + "\""

            lucene_query += "%s%s " % (bool_val, token_text)

            if boost == 0.0:
                continue

            if boost != 1:
                token_str = token_text + " "
                lucene_query += bool_val + (token_str * int(boost - 1))

            lucene_query = lucene_query.strip()
            lucene_query += " "

            for field in parameters:
                new_dict = {"term": {
                    field: {"value": token_text,
                            "boost": parameters[field] * boost},
                }}
                if " " in token_text:
                    new_dict["term"]["field"]["type"] = "phrase"

                field_dicts.append(new_dict)

        lucene_query = lucene_query.replace("  ", " ")

        fields = []
        for param in parameters:
            fields.append(param + "^" + str(parameters[param]))

        # dsl_query = {
        #     "bool": {
        #         "should": field_dicts
        #     }
        # }

        dsl_query = {
            "bool": {
                "should": [
                    {"multi_match": {
                        "query": lucene_query,
                        "type": self.multi_match_type,
                        "fields": fields,
                        "operator": "or",
                    }},
                    {"match": {
                        # "_full_text":
                        "_full_ilc":
                        # "_all_text":
                            {"query": lucene_query,
                             "boost": 25}}},
                ]
            }
        }

        # dsl_query = {
        #     "bool": {
        #         "must": [
        #             {"multi_match": {
        #                 "query": lucene_query,
        #                 "type": self.multi_match_type,
        #                 "fields": fields,
        #                 "operator": "or",
        #             }}],
        #         "should": [{"match": {"_full_text":
        #                                   {"query":lucene_query,
        #                                    "boost": 26}}}]
        #     }
        # }

        # dsl_query = {
        #     "bool":{
        #         "must": [{"match": {"_full_text":
        #                                   {"query":lucene_query,
        #                                    "boost": 1 }}}]
        #     }
        # }

        ##        print(dsl_query)

        if self.tie_breaker:
            dsl_query["multi_match"]["tie_breaker"] = self.tie_breaker

        return dsl_query


from proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST, ILC_AZ_LIST, ILC_CORESC_LIST


class ElasticRetrievalBoostTweaked8(ElasticRetrievalBoost):

    def __init__(self, index_name, method, logger=None, use_default_similarity=True, max_results=None, es_instance=None,
                 save_terms=False, multi_match_type=None):
        super(ElasticRetrievalBoostTweaked8, self).__init__(index_name, method, logger, use_default_similarity,
                                                            max_results,
                                                            es_instance, save_terms, multi_match_type)

    def getQuery(self, structured_query):
        """

        :param structured_query:
        :param params:
        :return:
        """
        lucene_query = ""
        for token in structured_query:
            # TODO proper computing of the boost formula. Different methods?
            boost = token.boost * token.count
            bool_val = token.bool or ""

            token_text = token.token
            if " " in token_text:  # if token is a phrase
                token_text = "\"" + token_text + "\""

            lucene_query += "%s%s " % (bool_val, token_text)

            if boost == 0.0:
                continue

            if boost != 1:
                token_str = token_text + " "
                lucene_query += bool_val + (token_str * int(boost - 1))

            lucene_query = lucene_query.strip()
            lucene_query += " "

        lucene_query = lucene_query.replace("  ", " ")
        return lucene_query

    def getFieldsForParams(self, params):
        """

        :param params:
        :return:
        """
        fields = []
        for param in params:
            fields.append(param + "^" + str(params[param]))

        return fields

    def rewriteQueryAsDSL(self, structured_query, parameters):
        """
            Creates a multi_match DSL query for elasticsearch.

            :param structured_query: a StructuredQuery dict, optionally under the
                key "structured_query"
            :param parameters: dict of [field]=weight to replace in the query
        """
        if "structured_query" in structured_query:
            structured_query = structured_query["structured_query"]

        if not isinstance(structured_query, StructuredQuery):
            structured_query = StructuredQuery(structured_query)

        if not structured_query or len(structured_query) == 0:
            return None

        self.last_query = structured_query

        az_params = {k: parameters[k] for k in AZ_ZONES_LIST if k in parameters}
        ilc_az_params = {k: parameters[k] for k in ILC_AZ_LIST if k in parameters}
        coresc_params = {k: parameters[k] for k in CORESC_LIST if k in parameters}
        ilc_coresc_params = {k: parameters[k] for k in ILC_CORESC_LIST if k in parameters}

        if len(az_params) > len(coresc_params) or len(ilc_az_params) > len(ilc_coresc_params):
            internal_params = az_params
            external_params = ilc_az_params
        else:
            internal_params = coresc_params
            external_params = ilc_coresc_params

        lucene_query = self.getQuery(structured_query)

        dsl_query = {
            "bool": {
                "should": [
                    {"multi_match": {
                        "query": lucene_query,
                        "type": self.multi_match_type,
                        "fields": self.getFieldsForParams(external_params),
                        "operator": "or",
                    }},
                    {"multi_match": {
                        "query": lucene_query,
                        "type": self.multi_match_type,
                        "fields": self.getFieldsForParams(internal_params),
                        "operator": "or",
                    }},
                    {"match": {"_all_text":
                                   {"query": lucene_query,
                                    "boost": 50}}},
                ],

            }
        }

        # dsl_query = {
        #     "bool": {
        #         "must": [
        #             {"multi_match": {
        #                 "query": lucene_query,
        #                 "type": self.multi_match_type,
        #                 "fields": fields,
        #                 "operator": "or",
        #             }}],
        #         "should": [{"match": {"_full_text":
        #                                   {"query":lucene_query,
        #                                    "boost": 26}}}]
        #     }
        # }

        # dsl_query = {
        #     "bool":{
        #         "must": [{"match": {"_full_text":
        #                                   {"query":lucene_query,
        #                                    "boost": 1 }}}]
        #     }
        # }

        ##        print(dsl_query)

        if self.tie_breaker:
            dsl_query["multi_match"]["tie_breaker"] = self.tie_breaker

        return dsl_query


def main():
    pass


if __name__ == '__main__':
    main()
