from __future__ import print_function
from __future__ import absolute_import
from evaluation.experiment import Experiment
from proc.general_utils import getRootDir
import db.corpora as cp
from kw_evaluation_runs.aac_full_text_kw_exp import experiment, options
import json
from tqdm import tqdm
import pandas as pd
from db.ez_connect import ez_connect
import os

from scidoc.reference_formatting import formatListOfAuthors


def getAuthorNamesAsOneString(metadata):
    author_names = []
    for author in metadata.get("authors", []):
        author_name = "{} {}".format(author.get("given", ""), author.get("family", ""))
        author_name = author_name.strip()
        author_name = author_name.lower()
        if author_name == "":
            print("ERROR: Author name is blank", author)
        else:
            author_names.append(author_name)

    return author_names


def isSelfCitation(authors1, authors2):
    if len(authors1) > 0 and len(authors2) > 0 \
            and authors1[0].lower() == authors2[0].lower():
        return True
    return False


def getOverlappingAuthors(authors1, authors2):
    return list(set(authors1).intersection(set(authors2)))


def self_citing_comparison(metadata1, guid):
    authors1 = getAuthorNamesAsOneString(metadata1)
    metadata2 = cp.Corpus.getMetadataByGUID(guid)
    authors2 = getAuthorNamesAsOneString(metadata2)

    res = {"is_self_citation": isSelfCitation(authors1, authors2),
           "overlapping_authors": len(getOverlappingAuthors(authors1, authors2)),
           "authors1_len": len(authors1),
           "authors2_len": len(authors2),
           "guid_from": metadata1["guid"],
           "guid_to": guid,
           }
    return res


def save_csv_with_pandas(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(cp.Corpus.ROOT_DIR, filename))


def process_self_citations_corpus(collection_id):
    corpus = ez_connect(collection_id)

    results = []
    test_files = cp.Corpus.listPapers(experiment["test_files_condition"])
    test_files = set(test_files)
    all_files = cp.Corpus.listPapers()

    changes_results = []

    for guid in tqdm(all_files):
    # for guid in all_files:
        doc = cp.Corpus.loadSciDoc(guid)
        meta = cp.Corpus.getMetadataByGUID(guid)

        # print(doc.metadata["title"], metadata.get("collection_id"))
        res_cit, outlinks, missing_references = cp.Corpus.selectDocResolvableCitations(doc)

        num_self_references = 0
        num_refs_with_overlapping_authors = 0

        old = len(meta["outlinks"])
        new = len(outlinks)
        if new != old:
            print("{} changed: original {} now {}".format(guid, old, new))
            if new > old:
                diff = set(outlinks) - set(meta["outlinks"])
                prep = "ADD"
            elif old > new:
                prep = "REMOVE"
                diff = set(meta["outlinks"]) - set(outlinks)
            for diff_guid in diff:
                diff_meta = corpus.getMetadataByGUID(diff_guid)
                changes_results.append({
                    "op": prep,
                    "authors": formatListOfAuthors(diff_meta["authors"]),
                    "title": diff_meta["title"],
                    "year": diff_meta["year"],
                    "collection": diff_meta["collection_id"]
                })
                # print("{} : {} ({}) - {} - {}".format(prep,
                #                                       formatListOfAuthors(diff_meta["authors"]),
                #                                       diff_meta["year"],
                #                                       diff_meta["title"],
                #                                       diff_meta["collection_id"]))
            meta["outlinks"] = list(outlinks.keys())

        for link in outlinks:
            res = self_citing_comparison(doc.metadata, link)
            if res["is_self_citation"]:
                num_self_references += 1
            if res["overlapping_authors"] > 0:
                num_refs_with_overlapping_authors += 1

            res["test_file"] = (guid in test_files)
            results.append(res)

        meta["num_self_references"] = num_self_references
        meta["num_refs_with_overlapping_authors"] = num_refs_with_overlapping_authors
        corpus.setMetadata(meta)
        # print(json.dumps(res_cit, indent=3))

    save_csv_with_pandas("self_citations_{}.csv".format(collection_id.lower()), results)
    save_csv_with_pandas("changes_in_outlinks_{}.csv".format(collection_id.lower()), changes_results)


def main():
    # process_self_citations_corpus("AAC")
    process_self_citations_corpus("PMC_CSC")


if __name__ == '__main__':
    main()
