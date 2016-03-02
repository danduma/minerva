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

import requests

import minerva.db.corpora as cp
from minerva.db.elastic_corpus import ElasticCorpus
from minerva.importing.importing_functions import (convertXMLAndAddToCorpus,
    updatePaperInCollectionReferences)

import celery_app
from celery_app import app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

RUN_LOCALLY=False

def checkCorpusConnection(local_corpus_dir="",
    corpus_endpoint={"host":celery_app.MINERVA_ELASTICSEARCH_SERVER_IP,
    "port":celery_app.MINERVA_ELASTICSEARCH_SERVER_PORT}):
    """
    """
    if not isinstance(cp.Corpus, ElasticCorpus):
        cp.useElasticCorpus()
        cp.Corpus.connectCorpus(local_corpus_dir, corpus_endpoint)

@app.task(ignore_result=True)
def t_convertXMLAndAddToCorpus(file_path, corpus_id, import_id, collection_id, existing_guid=None):
    """
        Reads the input XML and saves a SciDoc
    """
    if RUN_LOCALLY:
        convertXMLAndAddToCorpus(
            file_path,
            corpus_id,
            import_id,
            collection_id,
            existing_guid=existing_guid)
    else:
        r=requests.get(celery_app.MINERVA_FILE_SERVER_URL+"/file/"+file_path)
        if not r.ok:
            logger.error("HTTP Error code %d" % r.status_code)
            if r.status_code==500:
                raise self.retry(countdown=120)
            else:
                raise RuntimeError("HTTP Error code %d: %s" % (r.status_code, r.content))

        convertXMLAndAddToCorpus(
            file_path,
            corpus_id,
            import_id,
            collection_id,
            xml_string=r.content,
            existing_guid=existing_guid)


@app.task(ignore_result=True)
def t_updatePaperInCollectionReferences(doc_id, import_options):
    """
        Updates one paper's in-collection references, etc.
    """
    updatePaperInCollectionReferences(doc_id, import_options)

checkCorpusConnection()