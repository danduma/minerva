import db.corpora as cp
from db.ez_connect import ez_connect
import os
from tqdm import tqdm
from copy import deepcopy
from db.elastic_corpus import index_equivalence

def fix_broken_fix(corpus, wrong_aac_pmc_files):
    ROOT_FIELDS = ["guid",
                   # "metadata",
                   "norm_title",
                   "num_in_collection_references",
                   "num_resolvable_citations",
                   "num_inlinks",
                   "time_modified",
                   # "outlinks",
                   "has_scidoc",
                   "statistics"]

    for guid in tqdm(wrong_aac_pmc_files):
        meta = corpus.getFullPaperData("guid", guid)
        if "collection_id" in meta:
            meta2 = deepcopy(meta)
            del meta2["metadata"] # meta2 IS the metadata
            meta3 = {}
            for r in ROOT_FIELDS:
                if r in meta2:
                    meta3[r] = meta2[r]
                elif r in meta:
                    meta3[r] = meta[r]

            meta3["metadata"] = meta2

            meta = meta3

        corpus.es.index(
            index=index_equivalence["papers"]["index"],
            doc_type=index_equivalence["papers"]["type"],
            op_type="index",
            id=meta["guid"],
            body=meta
        )

        if meta["metadata"]["collection_id"] != "PMC_CSC":
            print("Not yet fixed!", guid)
            meta["metadata"]["collection_id"] = "PMC_CSC"

            if "collection_id" in meta:
                del meta["collection_id"]

            corpus.setMetadata(guid, meta)
            # print(meta["title"],"\n")


def fix_once(corpus, wrong_aac_pmc_files):
    for guid in tqdm(wrong_aac_pmc_files):
        meta = corpus.getMetadataByGUID(guid)
        if meta["metadata"]["collection_id"] != "PMC_CSC":
            print("Not yet fixed!", guid)
            meta["metadata"]["collection_id"] = "PMC_CSC"

            if "collection_id" in meta:
                del meta["collection_id"]

            corpus.setMetadata(guid, meta)
            # print(meta["title"],"\n")


def main():
    corpus = ez_connect("AAC")
    with open(os.path.join(corpus.ROOT_DIR, "not_aac_but_pmc.txt")) as f:
        wrong_aac_pmc_files = [line.strip() for line in f.readlines()]

    # fix_once(corpus, wrong_aac_pmc_files)
    fix_broken_fix(corpus, wrong_aac_pmc_files)


if __name__ == '__main__':
    main()
