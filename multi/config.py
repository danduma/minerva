# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
import os
import six.moves.configparser

config = six.moves.configparser.ConfigParser()
config_file_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..","config.ini"))
config.readfp(open(config_file_path,"r"))

WORKSTATION_IP="129.215.197.73"
SERVER_IP="129.215.197.75"
##SERVER_IP="129.215.90.202"
##SERVER_IP="localhost"

ES_USER=config.get("Elasticsearch","user")
ES_PASS=config.get("Elasticsearch","password")
##MINERVA_FILE_SERVER_URL="http://%s:5599" % WORKSTATION_IP
MINERVA_FILE_SERVER_URL="http://%s:5599" % WORKSTATION_IP
MINERVA_AMQP_SERVER_URL="amqp://minerva:minerva@%s:5672//" % SERVER_IP
MINERVA_ELASTICSEARCH_SERVER_IP=SERVER_IP
MINERVA_ELASTICSEARCH_SERVER_PORT=9200
if ES_USER:
    MINERVA_ELASTICSEARCH_ENDPOINT="http://%s:%s@%s:%d/" % (ES_USER,
                                                            ES_PASS,
                                                            MINERVA_ELASTICSEARCH_SERVER_IP,
                                                            MINERVA_ELASTICSEARCH_SERVER_PORT)
##    MINERVA_ELASTICSEARCH_ENDPOINT={"host":MINERVA_ELASTICSEARCH_SERVER_IP,
##                                    "port":MINERVA_ELASTICSEARCH_SERVER_PORT,
##                                    "url_prefix":"http://%s:%s" % (ES_USER,  ES_PASS)}
else:
    MINERVA_ELASTICSEARCH_ENDPOINT={"host":MINERVA_ELASTICSEARCH_SERVER_IP,
                                    "port":MINERVA_ELASTICSEARCH_SERVER_PORT}


#MINERVA_RABBITMQ_ADMIN="http://%s:15672" % SERVER_IP
#MINERVA_FLOWER_ADMIN="http://%s:5555" % SERVER_IP



def main():
    pass

if __name__ == '__main__':
    main()
