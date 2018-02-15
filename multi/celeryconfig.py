# <purpose>
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from kombu import Queue, Exchange

CELERY_TASK_SERIALIZER='json'
CELERY_ACCEPT_CONTENT=['json']  # Ignore other content
CELERY_RESULT_SERIALIZER='json'
CELERY_TIMEZONE='Europe/London'
CELERY_ENABLE_UTC=True
CELERY_TASK_RESULT_EXPIRES=3600

CELERY_QUEUES = (
    Queue('default', Exchange('default'), routing_key='default'),
    Queue('import_xml', Exchange('import_xml'), routing_key='import_xml'),
    Queue('update_references', Exchange('update_references'), routing_key='update_references'),
)

CELERY_ROUTES = {
    't_convertXMLAndAddToCorpus': {'queue': 'import_xml', 'routing_key': 'import_xml'},
    't_updatePaperInCollectionReferences': {'queue': 'update_references', 'routing_key': 'update_references'},
}