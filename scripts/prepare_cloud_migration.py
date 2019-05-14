from db.elastic_corpus import ElasticCorpus
import db.corpora as cp
from proc.general_utils import getRootDir
from multi.config import set_config

root_dir=getRootDir("aac")

cp.useElasticCorpus()
# cp.Corpus.connectCorpus(root_dir, endpoint=GCP_ENDPOINT)
cp.Corpus.connectCorpus(root_dir, endpoint=set_config("aws-server"))

print("")

# for index in ["scidocs", "papers", "venues", "cache", "authors", "links", "missing_references"]:
# for index in ["papers", "venues", "cache", "authors", "links", "missing_references"]:
# for index in ["papers", "venues", "cache", "authors", "links", "missing_references"]:
#     if cp.Corpus.es.indices.exists(index):
#         cp.Corpus.deleteIndex(index)

cp.Corpus.createAndInitializeDatabase()

# if cp.Corpus.es.indices.exists("scidocs"):
#     cp.Corpus.deleteIndex("scidocs")

# settings = {
#     "number_of_shards": 5,
#     "number_of_replicas": 1
# }
# properties = {
#     "scidoc": {"type": "string", "index": "no", "store": True, "doc_values": False},
#     "guid": {"type": "string", "index": "not_analyzed", "store": True},
#     "time_created": {"type": "date"},
#     "time_modified": {"type": "date"},
# }
# cp.Corpus.createTable("scidocs", settings, properties)

# if cp.Corpus.es.indices.exists("cache"):
#     cp.Corpus.deleteIndex("cache")
#
# settings = {
#     "number_of_shards": 3,
#     "number_of_replicas": 0
# }
# properties = {
#     "data": {"type": "string", "index": "no", "store": True, "doc_values": False},
#     "time_created": {"type": "date"},
#     "time_modified": {"type": "date"},
# }
# cp.Corpus.createTable("cache", settings, properties)


# if cp.Corpus.es.indices.exists("papers"):
#     cp.Corpus.deleteIndex("papers")
#
# settings = {
#     "number_of_shards": 5,
#     "number_of_replicas": 1
# }
# properties = {
#     "guid": {"type": "string", "index": "not_analyzed"},
#     "metadata": {"type": "nested", "include_in_parent": True},
#     "norm_title": {"type": "string", "index": "not_analyzed"},
#     "author_ids": {"type": "string", "index": "not_analyzed", "store": True},
#     "num_in_collection_references": {"type": "integer"},
#     "num_resolvable_citations": {"type": "integer"},
#     "num_inlinks": {"type": "integer"},
#     "collection_id": {"type": "string", "index": "not_analyzed", "store": True},
#     "import_id": {"type": "string", "index": "not_analyzed", "store": True},
#     "time_created": {"type": "date"},
#     "time_modified": {"type": "date"},
#     "has_scidoc": {"type": "boolean", "index": "not_analyzed", "store": True},
#     "flags": {"type": "string", "index": "not_analyzed", "store": True},
#     # This is all now accessed through the nested metadata
#     ##            "filename": {"type":"string", "index":"not_analyzed", "store":True},
#     ##            "corpus_id": {"type":"string", "index":"not_analyzed"},
#     ##            "title": {"type":"string", "store":True},##            "surnames": {"type":"string"},
#     ##            "year": {"type":"integer"},
#     ##            "in_collection_references": {"type":"string", "index":"not_analyzed", "store":True},
#     ##            "inlinks": {"type":"string", "index":"not_analyzed", "store":True},
# }
# cp.Corpus.createTable("papers", settings, properties)

settings = {
    "number_of_shards": 2,
    "number_of_replicas": 0
}
properties = {
    "guid": {"type": "string", "index": "not_analyzed"},
    "metadata": {"type": "nested", "include_in_parent": True},
    "norm_title": {"type": "string", "index": "not_analyzed"},
    "author_ids": {"type": "string", "index": "not_analyzed", "store": True},
    "num_in_collection_references": {"type": "integer"},
    "num_resolvable_citations": {"type": "integer"},
    "num_inlinks": {"type": "integer"},
    "collection_id": {"type": "string", "index": "not_analyzed", "store": True},
    "import_id": {"type": "string", "index": "not_analyzed", "store": True},
    "time_created": {"type": "date"},
    "time_modified": {"type": "date"},
    "has_scidoc": {"type": "boolean", "index": "not_analyzed", "store": True},
    "flags": {"type": "string", "index": "not_analyzed", "store": True},
    # This is all now accessed through the nested metadata
    ##            "filename": {"type":"string", "index":"not_analyzed", "store":True},
    ##            "corpus_id": {"type":"string", "index":"not_analyzed"},
    ##            "title": {"type":"string", "store":True},##            "surnames": {"type":"string"},
    ##            "year": {"type":"integer"},
    ##            "in_collection_references": {"type":"string", "index":"not_analyzed", "store":True},
    ##            "inlinks": {"type":"string", "index":"not_analyzed", "store":True},
}

# cp.Corpus.es.indices.create(
#     index="papers2",
#     body={"settings": settings,
#           "mappings": {"paper": {"properties": properties}}})
#
#

#