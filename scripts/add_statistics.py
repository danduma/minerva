# Compute statistics for each file
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import os, sys, json

import minerva.db.corpora as cp
from minerva.proc.results_logging import ProgressIndicator
from minerva.evaluation.statistics_functions import computeAnnotationStatistics
from minerva.squad.tasks import computeAnnotationStatisticsTask


def add_statistics_to_all_files(use_celery=False, conditions=None, max_files=sys.maxint):
    """
        For each paper in the corpus, it computes and stores its statistics
    """
    print("Listing files...")
##    papers=cp.Corpus.listPapers(conditions, max_results=max_files)
    papers=cp.Corpus.listRecords(conditions, max_results=max_files, field="_id", table="scidocs")
    print("Computing statistics for each SciDoc")
    progress=ProgressIndicator(True,len(papers), print_out=False)
    for guid in papers:
        if use_celery:
            computeAnnotationStatisticsTask.apply_async(
                args=[guid],
                kwargs={},
                queue="compute_statistics"
                )
        else:
            computeAnnotationStatistics(guid)
            progress.showProgressReport("Computing statistics -- latest paper "+ guid)


def aggregate_statistics(conditions=None, max_files=sys.maxint):
    """
    """
    res={
        "csc_type_counts":{},
        "az_counts":{},
        "num_sentences":[],
        "num_sections":[],
        "num_paragraphs":[],
        "per_zone_citations":{},
        "num_files":0
        }

    print("Listing files...")

    papers=cp.Corpus.listRecords(conditions, max_results=max_files, table="scidocs", field="_id")
    print("Aggregating statistics for each SciDoc")
    progress=ProgressIndicator(True,len(papers), print_out=False)

    num_files=0
    for guid in papers:
##        try:
##            stats=cp.Corpus.getStatistics(guid)
##        except:
        computeAnnotationStatistics(guid)
        try:
            stats=cp.Corpus.getStatistics(guid)
        except:
            continue

        for key in ["csc_type_counts", "az_counts", "per_zone_citations"]:
            for key2 in stats[key]:
                res[key][key2] = res[key].get(key2,0) + stats[key][key2]

        for key in ["num_sentences", "num_sections", "num_paragraphs"]:
            res[key].append(stats[key])

        num_files+=1

        progress.showProgressReport("Aggregating statistics -- latest paper "+ guid)

    if num_files == 0:
        print("No files found in db!")
        return

    for key in ["num_sentences", "num_sections", "num_paragraphs"]:
        res[key.replace("num","avg")]=sum(res[key]) / float(num_files)

    res["num_files"] = num_files
    json.dump(res, file(os.path.join(cp.Corpus.paths.output,"stats.json"), "w"))


def main():
    from minerva.squad.config import MINERVA_ELASTICSEARCH_ENDPOINT
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\pmc_coresc", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)

    add_statistics_to_all_files(use_celery=True, max_files=1000)
##    aggregate_statistics(max_files=10)
    pass

if __name__ == '__main__':
    main()
