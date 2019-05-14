# Config module: loads/keeps IP:port parameters for the elasticsearch server
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
import os
import json

config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))

WORKSTATION_IP = "129.215.197.73"
SERVER_IP = "129.215.197.75"

SERVER_CONFIGS = json.load(open(config_file_path, "r"))
MINERVA_ELASTICSEARCH_ENDPOINT = ""

MINERVA_FILE_SERVER_URL = "http://%s:5599" % WORKSTATION_IP
MINERVA_RPC_SERVER_URL = "amqp://minerva:minerva@%s:5672//" % SERVER_IP


def set_config(es_config="koko"):
    global MINERVA_ELASTICSEARCH_ENDPOINT

    bits = (SERVER_CONFIGS[es_config]["user"],
            SERVER_CONFIGS[es_config]["pass"],
            SERVER_CONFIGS[es_config]["ip"],
            SERVER_CONFIGS[es_config]["port"],
            )

    if SERVER_CONFIGS[es_config]["user"]:
        MINERVA_ELASTICSEARCH_ENDPOINT = "http://%s:%s@%s:%d/" % bits
    else:
        MINERVA_ELASTICSEARCH_ENDPOINT = "http://%s:%d/" % bits[2:]
    print("Using config [%s] (%s)" % (es_config, MINERVA_ELASTICSEARCH_ENDPOINT))

    return MINERVA_ELASTICSEARCH_ENDPOINT


def main():
    pass


if __name__ == '__main__':
    main()
