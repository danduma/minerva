# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

##import ConfigParser
##config = ConfigParser.ConfigParser()
##config.readfp(open(os.path.dirname(__file__)))

WORKSTATION_IP="129.215.197.73"
SERVER_IP="129.215.197.75"
##SERVER_IP="129.215.90.202"
##SERVER_IP="localhost"

ES_USER="root"
ES_PASS="emperifollaramos"
##MINERVA_FILE_SERVER_URL="http://%s:5599" % WORKSTATION_IP
MINERVA_FILE_SERVER_URL="http://%s:5599" % WORKSTATION_IP
MINERVA_AMQP_SERVER_URL="amqp://minerva:minerva@%s:5672//" % SERVER_IP
MINERVA_ELASTICSEARCH_SERVER_IP=SERVER_IP
MINERVA_ELASTICSEARCH_SERVER_PORT=9200
##MINERVA_ELASTICSEARCH_ENDPOINT={"host":MINERVA_ELASTICSEARCH_SERVER_IP, "port":MINERVA_ELASTICSEARCH_SERVER_PORT}
MINERVA_ELASTICSEARCH_ENDPOINT="http://%s:%s@%s/" % (ES_USER, ES_PASS, SERVER_IP)

#MINERVA_RABBITMQ_ADMIN="http://%s:15672" % SERVER_IP
#MINERVA_FLOWER_ADMIN="http://%s:5555" % SERVER_IP



def main():
    pass

if __name__ == '__main__':
    main()
