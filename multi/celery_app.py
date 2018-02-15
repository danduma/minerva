# Celery app module. A lot of config in here.
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

##from __future__ import absolute_import

from __future__ import absolute_import
from celery import Celery
from kombu import Queue, Exchange
from celery.bin import Option
from .config import *
##import multi.celeryconfig

# celery worker --app=multi.celery_app:app


app = Celery('multi',
             broker=MINERVA_AMQP_SERVER_URL,
             backend=MINERVA_AMQP_SERVER_URL,
             # include=['minerva.multi.tasks'])
             include=['multi.tasks'])

##app.config_from_object('celeryconfig')
# Optional configuration, see the application user guide.
app.conf.update(
    CELERY_TASK_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json'],  # Ignore other content
    CELERY_RESULT_SERIALIZER='json',
    CELERY_TIMEZONE='Europe/London',
    CELERY_ENABLE_UTC=True,
    CELERY_TASK_RESULT_EXPIRES=3600,

    CELERY_QUEUES = (
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('import_xml', Exchange('import_xml'), routing_key='import_xml'),
        Queue('update_references', Exchange('update_references'), routing_key='update_references'),
        Queue('prebuild_bows', Exchange('prebuild_bows'), routing_key='prebuild_bows'),
        Queue('add_to_index', Exchange('add_to_index'), routing_key='add_to_index'),
        Queue('precompute_formulas', Exchange('precompute_formulas'), routing_key='precompute_formulas'),
        Queue('compute_statistics', Exchange('compute_statistics'), routing_key='compute_statistics'),
        Queue('annotate_keywords', Exchange('annotate_keywords'), routing_key='annotate_keywords'),
    ),

    CELERY_ROUTES = {
        'importXMLTask': {'queue': 'import_xml', 'routing_key': 'import_xml'},
        'updateReferencesTask': {'queue': 'update_references', 'routing_key': 'update_references'},
        'prebuildBOWTask': {'queue': 'prebuild_bows', 'routing_key': 'prebuild_bows'},
        'addToIndexTask': {'queue': 'add_to_index', 'routing_key': 'add_to_index'},
        'precomputeFormulasTask': {'queue': 'precompute_formulas', 'routing_key': 'precompute_formulas'},
        'computeAnnotationStatisticsTask': {'queue': 'compute_statistics', 'routing_key': 'compute_statistics'},
        'annotateKeywordsTask': {'queue': 'annotate_keywords', 'routing_key': 'annotate_keywords'},
    }
)

##app.user_options['preload'].add(
##    Option('-LOC', '--run_locally', default=False,
##           help='Configuration template to use.'),
##)
##

if __name__ == '__main__':
    app.start()