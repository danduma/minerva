#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      dd
#
# Created:     06/05/2014
# Copyright:   (c) dd 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from oaipmh.client import Client
from oaipmh.metadata import MetadataRegistry, oai_dc_reader

URL = ' http://www.pubmedcentral.nih.gov/oai/oai.cgi'

bla="set=pmc-open"

registry = MetadataRegistry()
registry.registerReader('oai_dc', oai_dc_reader)
client = Client(URL, registry)

for record in client.listRecords(metadataPrefix='oai_dc', set='pmc-open'):
    print(record)
