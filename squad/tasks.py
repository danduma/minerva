# <purpose>
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

# AAC corpus importer
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import logging

import requests

import minerva.db.corpora as cp
from minerva.db.elastic_corpus import ElasticCorpus
from minerva.evaluation.elastic_retrieval import ElasticRetrieval

from minerva.importing.importing_functions import (convertXMLAndAddToCorpus,
    updatePaperInCollectionReferences)
from minerva.evaluation.prebuild_functions import prebuildMulti
from minerva.evaluation.precompute_functions import addPrecomputeExplainFormulas
from minerva.db.result_store import createResultStorers
from minerva.evaluation.index_functions import addBOWsToIndex

import celery_app
from celery_app import app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

RUN_LOCALLY=True

def checkCorpusConnection(local_corpus_dir="",
    corpus_endpoint={"host":celery_app.MINERVA_ELASTICSEARCH_SERVER_IP,
    "port":celery_app.MINERVA_ELASTICSEARCH_SERVER_PORT}):
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
        r=requests.get(celery_app.MINERVA_FILE_SERVER_URL+"/file/"+file_path)
        if not r.ok:
            logger.error("HTTP Error code %d" % r.status_code)
            if r.status_code==500:
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
            #TODO what other exceptions?
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
def prebuildBOWTask(self, method_name, parameters, function, guid, force_prebuild, rhetorical_annotations):
    """
        Builds the BOW for a single paper
    """
    try:
        #prebuildMulti(method_name, parameters, function, doc, doctext, guid, force_prebuild, rhetorical_annotations)
        prebuildMulti(method_name, parameters, function, None, None, guid, force_prebuild, rhetorical_annotations)
    except Exception as e:
        logging.exception("Error running prebuildMulti")
        self.retry(countdown=120, max_retries=4)

@app.task(ignore_result=True, bind=True)
def addToindexTask(self, guid, indexNames, index_max_year):
    """
        Adds one paper to the index for all indexes. If its BOW has not already
        been built, it builds it too.
    """
    try:
        addBOWsToIndex(guid, indexNames, index_max_year)
    except Exception as e:
        logging.exception("Error running addBOWsToIndex")
        self.retry(countdown=120, max_retries=4)

@app.task(ignore_result=True, bind=True)
def precomputeFormulasTask(self, precomputed_query, doc_method, doc_list, index_name, exp_name, experiment_id, max_results):
    """
        Runs one precomputed query, and the explain formulas and adds them to
        the DB.
    """
    try:
        model=ElasticRetrieval(index_name, doc_method, max_results=max_results, es_instance=cp.Corpus.es)
        writers=createResultStorers(exp_name)
        addPrecomputeExplainFormulas(precomputed_query, doc_method, doc_list, model, writers, experiment_id)
    except:
        logging.exception("Error running addPrecomputeExplainFormulas")
        self.retry(countdown=120, max_retries=4)

checkCorpusConnection()
