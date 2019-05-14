# Throwaway file for elastic maintenance: batch delete indeces and others
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import

if __name__ == "__main__" and __package__ is None:
    __package__ = "db"

import sys
import os

# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from db.ez_connect import ez_connect

ec = ez_connect("AAC", "aws-server")
from multi.config import MINERVA_ELASTICSEARCH_ENDPOINT

##
##ec.deleteIndex("wosp16_experiments_prr_*")
# ec.deleteIndex("*_lrec_experiments_*")
# ec.deleteIndex("acl_wosp_experiments_prr_*")
# ec.deleteIndex("idx_inlink_context_pmc_2013_*")
# ec.deleteIndex("wosp16_experiments_*")
# ec.deleteIndex("pmc_lrec_experiments2_prr_*")

# ec.deleteIndex("aac_full_text_kw_selection_kw_data")
# ec.deleteIndex("*")


##ec.deleteByQuery("cache", "_id:resolvable_*")
##ec.deleteByQuery("cache", "_id:bow_*")

##hits=ec.unlimitedQuery(
##        q="metadata.collection_id:AAC",
##        index=ES_INDEX_PAPERS,
##        doc_type=ES_TYPE_PAPER,
##        _source="_id",
##)
##print(len(hits))

##print(ec.loadSciDoc("f9c08f84-1e5e-4d57-80bc-b576fefa109f"))
##print(ec.SQLQuery("SELECT guid,metadata.filename FROM papers where metadata.year >2013"))
##print(ec.getMetadataByGUID("df8c8824-1784-46f1-b621-cc6e5aca0dad"))

##hits=ec.unlimitedQuery(
##        q="_id:\"26e00c42-d0fe-4116-abc8-47b0d454cfb5\"",
##        index="idx_az_annotated_pmc_2013_1",
##        _source="_id",
##)
##print(hits)
##print(len(hits))


def main():
    pass


if __name__ == '__main__':
    main()
