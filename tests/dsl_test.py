#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Masterman
#
# Created:     23/12/2015
# Copyright:   (c) Masterman 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

client = Elasticsearch()

s = Search(using=client, index="my-index") \
    .filter("term", category="search") \
    .query("match", title="python")   \
    .query(~Q("match", description="beta"))

s.aggs.bucket('per_tag', 'terms', field='tags') \
    .metric('max_lines', 'max', field='lines')

response = s.execute()

for hit in response:
    print(hit.meta.score, hit.title)

for tag in response.aggregations.per_tag.buckets:
    print(tag.key, tag.max_lines.value)

def main():
    pass

if __name__ == '__main__':
    main()
