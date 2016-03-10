# Celery app module. A lot of config in here.
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import

from celery import Celery
from kombu import Queue, Exchange
##import minerva.squad.celeryconfig

# celery worker --app=squad.celery_app:app

SERVER_IP="129.215.197.73"
##SERVER_IP="localhost"

MINERVA_FILE_SERVER_URL="http://%s:5599" % SERVER_IP
MINERVA_AMQP_SERVER_URL="amqp://minerva:minerva@%s:5672//" % SERVER_IP
MINERVA_ELASTICSEARCH_SERVER_IP="129.215.90.202"
MINERVA_ELASTICSEARCH_SERVER_PORT=9200
#MINERVA_RABBITMQ_ADMIN="http://%s:15672" % SERVER_IP
#MINERVA_FLOWER_ADMIN="http://%s:5555" % SERVER_IP

app = Celery('squad',
             broker=MINERVA_AMQP_SERVER_URL,
             backend=MINERVA_AMQP_SERVER_URL,
             include=['minerva.squad.tasks'])

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
    ),

    CELERY_ROUTES = {
        'importXMLTask': {'queue': 'import_xml', 'routing_key': 'import_xml'},
        'updateReferencesTask': {'queue': 'update_references', 'routing_key': 'update_references'},
        'prebuildBOWTask': {'queue': 'prebuild_bows', 'routing_key': 'prebuild_bows'},
        'addToIndexTask': {'queue': 'add_to_index', 'routing_key': 'add_to_index'},
        'precomputeFormulasTask': {'queue': 'precompute_formulas', 'routing_key': 'precompute_formulas'},
    }
)

if __name__ == '__main__':
    app.start()