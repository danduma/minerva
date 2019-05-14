#!/usr/bin/env python
#

import os
import argparse
import re
import elasticsearch
from elasticsearch import helpers
import json

# with open('cloudalytics_mappings.json', 'r') as infile:
#  MAPPING_STR = infile.read()

input_es_url = 'http://129.215.197.75:9200'
output_es_url = 'http://elastic:bvSHXy4k@35.234.156.47:9200'
timeout = 600
records_per_batch = 100

if re.search('^https?://', input_es_url):
    input_es = elasticsearch.Elasticsearch([input_es_url],
                                           timeout=timeout,
                                           sniff_on_start=True,
                                           sniff_on_connection_fail=True,
                                           sniffer_timeout=timeout)

if re.search('^https?://', output_es_url):
    output_es = elasticsearch.Elasticsearch([output_es_url],
                                            timeout=timeout,
                                            sniff_on_start=True,
                                            sniff_on_connection_fail=True,
                                            sniffer_timeout=timeout)
    output_i_client = elasticsearch.client.IndicesClient(output_es)


def get_hits(es, index_name, query_str):
    hits = 0
    try:
        res = es.search(index=index_name, body=query_str, filter_path=['hits.hits.total'])
        hits = res['hits']['total']
        print("[INPUT ] Got %d hits in %s for query: %s" % (hits,
                                                            index_name,
                                                            query_str))
    except (elasticsearch.exceptions.NotFoundError, elasticsearch.exceptions.AuthorizationException) as e:
        print("[INPUT ] No such index '%s' (or it's closed)" % index_name)

    return hits



#
# def copy_index_creation(input_es, input_index_name, output_es, output_i_client, output_index_name, query_str):
#     try:
#         res_output = output_es.search(index=output_index_name, body=query_str)
#         total_hits = res_output['hits']['total']
#         print("[OUTPUT] Got %d hits in %s" % (total_hits, output_index_name))
#     except elasticsearch.exceptions.NotFoundError as e:
#         print("[OUTPUT] No such index '%s'" % output_index_name)
#         print("[OUTPUT] Creating index '%s'" % output_index_name)
#         input_index_client = elasticsearch.client.IndicesClient(input_es)
#         old_settings = input_index_client.get_settings(input_index_name)
#         old_mappings = input_index_client.get_mapping(input_index_name)
#         new_settings = old_settings[input_index_name]
#         new_mappings = old_mappings[input_index_name]
#
#         new_settings.update(new_mappings)
#         if num_shards:
#             new_settings['settings']['index']['number_of_shards'] = args.num_shards
#         index_create_str = json.dumps(new_settings, sort_keys=True, indent=4, separators=(',', ': '))
#         # print index_create_str
#         output_i_client.create(index=output_index_name, body=index_create_str, timeout='240s', master_timeout='240s')


def copy_to_es(es, records, start_record_num, end_record_num, total_records, per_batch):
    print("Inserting records %d through %d of %s" % (start_record_num, end_record_num,
                                                     (str(total_records) if total_records > 0 else '???')))
    total_items = len(records)
    num_success, error_list = helpers.bulk(es, records, chunk_size=100)
    while num_success != total_items:
        print("[ERROR] %d of %d inserts succeeded!" % (num_success, per_batch))
        print("[ERROR] Errors:")
        print(error_list)
        print("Retrying...")
        num_success, error_list = helpers.bulk(es, records, chunk_size=100)


def copy_index(index_name, query_str, start_at=0):
    input_hits = 0

    input_hits = get_hits(input_es, index_name, query_str)

    if input_hits < 1:
        return

    i = 0

    input_data = helpers.scan(input_es,
                              index=index_name,
                              query=query_str,
                              size=records_per_batch)

    actions = []
    for record in input_data:
        i += 1
        actions.append(record)
        if i % records_per_batch == 0:
            copy_to_es(output_es, actions, i - records_per_batch + 1, i, input_hits, records_per_batch)

        actions = []

    # write out the last remaining records
    remaining_records = len(actions)
    if remaining_records > 0:
        copy_to_es(output_es, actions, i - remaining_records + 1, i, input_hits, remaining_records)


# main
def main():
    query_str = '{"query":{"match_all":{}}}'

    for index_name in ["scidocs"]:
        copy_index(index_name, query_str)

if __name__ == '__main__':
    main()
