# Throwaway file for elastic maintenance: batch delete indeces and others
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from .elastic_corpus import ElasticCorpus

ec=ElasticCorpus()
from multi.config import MINERVA_ELASTICSEARCH_ENDPOINT
ec.connectCorpus("",endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
##
##ec.deleteIndex("wosp16_experiments_prr_*")
ec.deleteIndex("idx_az_annotated_pmc_2013_1")

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
