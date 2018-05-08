# Celery task definitions
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function

import logging

import db.corpora as cp
import requests
from celery.utils.log import get_task_logger
from db.elastic_corpus import ElasticCorpus
from db.result_store import createResultStorers, ElasticResultStorer
from evaluation.keyword_functions import annotateKeywords
from evaluation.prebuild_functions import prebuildMulti
from evaluation.precompute_functions import addPrecomputeExplainFormulas
from evaluation.statistics_functions import computeAnnotationStatistics
from importing.importing_functions import (convertXMLAndAddToCorpus,
                                           updatePaperInCollectionReferences)
from retrieval.elastic_retrieval import ElasticRetrieval
from retrieval.index_functions import addBOWsToIndex
from ml.document_features import DocumentFeaturesAnnotator

from . import celery_app
from .celery_app import app

logger = get_task_logger(__name__)

RUN_LOCALLY = False


def checkCorpusConnection(local_corpus_dir="",
                          corpus_endpoint={"host": celery_app.MINERVA_ELASTICSEARCH_SERVER_IP,
                                           "port": celery_app.MINERVA_ELASTICSEARCH_SERVER_PORT}):
    """
        Connects this worker to the elasticsearch server. By default, uses
        values from celery_app.py
    """
    if not isinstance(cp.Corpus, ElasticCorpus):
        cp.useElasticCorpus()
        cp.Corpus.connectCorpus(local_corpus_dir, corpus_endpoint)


@app.task(ignore_result=True, bind=True)
def importXMLTask(self, file_path, corpus_id, import_id, collection_id, import_options, existing_guid):
    """
        Reads the input XML and saves a SciDoc
    """
    if RUN_LOCALLY:
        convertXMLAndAddToCorpus(
            file_path,
            corpus_id,
            import_id,
            collection_id,
            import_options,
            existing_guid=existing_guid)
    else:
        r = requests.get(celery_app.MINERVA_FILE_SERVER_URL + "/file/" + file_path)
        if not r.ok:
            logger.error("HTTP Error code %d" % r.status_code)
            if r.status_code == 500:
                raise self.retry(countdown=120)
            else:
                raise RuntimeError("HTTP Error code %d: %s" % (r.status_code, r.content))
        try:
            convertXMLAndAddToCorpus(
                file_path,
                corpus_id,
                import_id,
                collection_id,
                import_options,
                xml_string=r.content,
                existing_guid=existing_guid)
        except MemoryError:
            logging.exception("Exception: Out of memory in importXMLTask")
            raise self.retry(countdown=120, max_retries=4)
        except:
            # TODO what other exceptions?
            logging.exception("Exception in importXMLTask")
            raise self.retry(countdown=60, max_retries=2)


@app.task(ignore_result=True, bind=True)
def updateReferencesTask(self, doc_id, import_options):
    """
        Updates one paper's in-collection references, etc.
    """
    try:
        updatePaperInCollectionReferences(doc_id, import_options)
    except:
        logging.exception("Exception in updateReferencesTask")
        raise self.retry(countdown=120, max_retries=4)


@app.task(ignore_result=True, bind=True)
def prebuildBOWTask(self, method_name, parameters, function, guid, overwrite_existing_bows, filter_options, force_rebuild):
    """
        Builds the BOW for a single paper
    """
    try:
        prebuildMulti(method_name, parameters, function, None, None, guid, overwrite_existing_bows, filter_options, force_rebuild)
    except Exception as e:
        logging.exception("Error running prebuildMulti")
        self.retry(countdown=120, max_retries=4)


@app.task(ignore_result=True, bind=True)
def addToindexTask(self, guid, indexNames, index_max_year):
    """
        Adds one paper to the index for all indexes. If its BOW has not already
        been built, it builds it too.
    """
    logging.error("processing ", guid, indexNames, index_max_year)
    try:
        addBOWsToIndex(guid, indexNames, index_max_year)
    except Exception as e:
        logging.exception("Error running addBOWsToIndex")
        self.retry(countdown=120, max_retries=4)


@app.task(ignore_result=True, bind=True)
def precomputeFormulasTask(self, precomputed_query, doc_method, doc_list, index_name, exp_name, experiment_id,
                           max_results):
    """
        Runs one precomputed query, and the explain formulas and adds them to
        the DB.
    """
    try:
        model = ElasticRetrieval(index_name, doc_method, max_results=max_results, es_instance=cp.Corpus.es)
        writers = createResultStorers(exp_name)
        addPrecomputeExplainFormulas(precomputed_query, doc_method, doc_list, model, writers, experiment_id)
    except:
        logging.exception("Error running addPrecomputeExplainFormulas")
        self.retry(countdown=120, max_retries=4)


@app.task(ignore_result=True, bind=True)
def computeAnnotationStatisticsTask(self, guid):
    """
    """
    try:
        computeAnnotationStatistics(guid)
    except:
        logging.exception("Error running computeAnnotationStatisticsTask")
        self.retry(countdown=120, max_retries=4)


@app.task(ignore_result=True, bind=True)
def annotateKeywordsTask(self, precomputed_query,
                         doc_method,
                         doc_list,
                         index_name,
                         exp_name,
                         experiment_id,
                         context_extraction,
                         extraction_parameter,
                         keyword_selection_method,
                         keyword_selection_parameters,
                         max_results,
                         weights):
    """
        Runs one precomputed query, extracts explain formulas and from them
        picks best keywords for the citation/query, stores the keyword-annotated context
    """
    try:
        model = ElasticRetrieval(index_name, doc_method, max_results=max_results, es_instance=cp.Corpus.es)
        writers = {"ALL": ElasticResultStorer(self.exp["name"], "kw_data", endpoint=cp.Corpus.endpoint)}
        annotator=DocumentFeaturesAnnotator()
        annotateKeywords(precomputed_query, doc_method, doc_list, model, writers, experiment_id, context_extraction,
                         extraction_parameter, keyword_selection_method, keyword_selection_parameters, weights, annotator)
    except:
        logging.exception("Error running annotateKeywords")
        self.retry(countdown=120, max_retries=4)


checkCorpusConnection()
