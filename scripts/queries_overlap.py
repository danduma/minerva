# Compute the overlap between sets of queries, export json with list of queries to ignore

import json
from six import string_types
from proc.general_utils import loadListFromTxtFile
import os


def get_queries_ids(queries_file):
    queries = json.load(open(queries_file))

    ids = []
    for q in queries:
        assert q["file_guid"]
        assert q["citation_id"]
        id = q["file_guid"] + "_" + q["citation_id"]
        ids.append(id)
    ids = set(ids)
    return ids


def find_queries_overlap(queries1, queries2):
    ids1 = get_queries_ids(queries1)
    ids2 = get_queries_ids(queries2)

    difference = ids1.symmetric_difference(ids2)
    overlap = ids1.intersection(ids2)

    print("len Q1:", float(len(ids1)))
    print("len Q2:", float(len(ids2)))
    print("Q1 in Q2:", len(overlap) / float(len(ids1)))
    pct_overlap = len(overlap) / (len(ids1) + len(ids2))
    # print("Q1 in Q2:", pct_overlap)

    return ids1, ids2


#     c3_query_ids=sorted(list(set(results["query_id"])))

def export_queries_ids(qfilenames, output_filename):
    guids = []
    for filename in qfilenames:
        queries = json.load(open(filename, "r"))
        for q in queries:
            if isinstance(q, dict):
                guids.append(q["file_guid"])
            elif isinstance(q, string_types):
                guids.append(q)
            else:
                raise ValueError(str(q) + " is neither dict nor string")

    guids = list(set(guids))

    f = open(output_filename, "w")
    if output_filename.endswith(".json"):
        json.dump(list(guids), f)
    elif output_filename.endswith(".txt"):
        for guid in guids:
            f.write(guid + "\n")


def loadGuidList(filename):
    if filename.endswith(".json"):
        return json.load(open(filename, "r"))
    elif filename.endswith(".txt"):
        return loadListFromTxtFile(filename)


def compareGUIDLists(file1, file2):
    guids1 = set(loadGuidList(file1))
    guids2 = set(loadGuidList(file2))

    diff = guids1.difference(guids2)
    overlap = guids1.intersection(guids2)
    print(os.path.basename(file1), ":", len(guids1))
    print(os.path.basename(file2), ":", len(guids2))
    print("diff:", len(diff))
    print("overlap:", len(overlap))

    print(sorted(list(guids1))[:3])
    print(sorted(list(guids2))[:3])


def statsOnTestFiles(files):
    from db.ez_connect import ez_connect
    corpus = ez_connect(None, "koko")

    for filename in files:
        years = {}
        guids = set(loadGuidList(filename))
        for guid in guids:
            meta = corpus.getMetadataByGUID(guid)
            year = meta["year"]
            if year not in years:
                years[year] = 0
            years[year] += 1

        print(os.path.basename(filename))
        print(years)


def main():
    # ids1, ids2 = find_queries_overlap(
    #     "/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac/precomputed_queries.json",
    #     "/Users/masterman/NLP/PhD/aac/experiments/acl_wosp_experiments/precomputed_queries.json")

    # export_queries_ids(["/Users/masterman/NLP/PhD/aac/experiments/acl_wosp_experiments/precomputed_queries.json",
    #                     "/Users/masterman/NLP/PhD/aac/experiments/aac_lrec_experiments/precomputed_queries.json"],
    #                    "/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac/ignore_guids_aac_c3.json")

    # export_queries_ids(
    #     ["/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter5_test_aac/precomputed_queries_new1k.json",
    #      # "/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter5_test_aac/precomputed_queries_old1k.json",
    #      ],
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/c3_test_guids.txt")

    # export_queries_ids(
    #     ["/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter5_test_aac/precomputed_queries_new1k.json",
    #      ],
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/c3_test_guids.txt")

    # export_queries_ids(
    #     ["/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter5_test_aac/precomputed_queries_new1k.json",
    #      "/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter5_test_aac/precomputed_queries_old1k.json", ],
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/ignore_test_guids.json")

    # ids1, ids2 = find_queries_overlap(
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/thesis_chapter3_experiment_pmc/precomputed_queries.json",
    #     # "/Users/masterman/NLP/PhD/pmc_coresc/experiments/wosp16_experiments/precomputed_queries.json",
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_lrec_experiments2/precomputed_queries.json"
    #     )
    # export_queries_ids(
    #     ["/Users/masterman/NLP/PhD/pmc_frozen_precomputed_queries.json"],
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/ignore_test_guids_frozen.txt")

    # export_queries_ids(
    #     ["/Users/masterman/NLP/PhD/pmc_coresc/experiments/thesis_chapter3_experiment_pmc/precomputed_queries.json"],
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/test_guids_c3_pmc.txt")

    # export_queries_ids(
    #     ["/Users/masterman/NLP/PhD/aac_frozen_precomputed_queries.json"],
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/ignore_test_guids_frozen.txt")
    #
    # export_queries_ids(
    #     ["/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac/precomputed_queries.json"],
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/test_guids_c3_aac.txt")

    # compareGUIDLists(
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/ignore_test_guids_frozen.txt",
    #     "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/ignore_test_guids.json"
    # )
    #
    compareGUIDLists(
        "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/test_guids_c3_aac_desktop.txt",
        "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/train_guids_aac_c6.txt"
    )


    #  compareGUIDLists(
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/ignore_test_guids_frozen.txt",
    #     # "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/test_guids_c3_pmc.txt",
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace_TEST/test_guids.txt",
    # )

    # compareGUIDLists(
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/test_guids_pmc_c3_1000.txt",
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/test_guids_c3_pmc.txt",
    # )

    # compareGUIDLists(
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/test_guids.txt",
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/test_guids_pmc_c3_1k.txt",
    # )
    #
    # compareGUIDLists(
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace_TEST/test_guids.txt",
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/train_guids_pmc_c6.txt",
    # )



    # compareGUIDLists(
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/thesis_chapter3_experiment_pmc/test_guids_c3_desktop.txt",
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/thesis_chapter3_experiment_pmc/test_guids_c3_macbook.txt",
    # )

    # statsOnTestFiles([
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/ignore_test_guids_frozen.txt",
    #     # "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace/test_guids_c3_pmc.txt",
    #     "/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace_TEST/test_guids.txt",
    #     # "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace/ignore_test_guids_frozen.txt",
    #     # "/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace_TEST/test_guids.json"
    # ])

    #


if __name__ == '__main__':
    main()
