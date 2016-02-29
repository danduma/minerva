# Celery app module
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import

from celery import Celery

# celery worker --app=squad.celery_app:app

SERVER_IP="129.215.91.3"
##SERVER_IP="localhost"

MINERVA_FILE_SERVER_URL="http://%s:5599" % SERVER_IP
MINERVA_AMQP_SERVER_URL="amqp://minerva:minerva@%s:5672//" % SERVER_IP
MINERVA_ELASTICSEARCH_SERVER_IP=SERVER_IP
MINERVA_ELASTICSEARCH_SERVER_PORT=9200
#MINERVA_RABBITMQ_ADMIN="http://%s:15672" % SERVER_IP
#MINERVA_FLOWER_ADMIN="http://%s:5555" % SERVER_IP

app = Celery('squad',
             broker=MINERVA_AMQP_SERVER_URL,
             backend=MINERVA_AMQP_SERVER_URL,
             include=['minerva.squad.tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    CELERY_TASK_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json'],  # Ignore other content
    CELERY_RESULT_SERIALIZER='json',
    CELERY_TIMEZONE='Europe/London',
    CELERY_ENABLE_UTC=True,
    CELERY_TASK_RESULT_EXPIRES=3600,
)

if __name__ == '__main__':
    app.start()