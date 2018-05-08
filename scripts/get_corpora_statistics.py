from db.ez_connect import ez_connect
import pandas as pd
import os
from tqdm import tqdm
from proc.nlp_functions import AZ_ZONES_LIST, CORESC_LIST
from scidoc.reference_formatting import formatListOfAuthors


def compile_corpus_metadata(collection):
    corpus = ez_connect(collection)
    guids = corpus.listPapers()

    num_papers = len(guids)
    empty_papers = 0
    abstract_only_papers = 0
    full_text_papers = 0

    results = []

    aac_to_pmc = []

    for guid in tqdm(guids):
        doc = corpus.loadSciDoc(guid)
        # if doc.metadata.get("pmc_id") or doc.metadata.get("journal_nlm-ta"):
        #     aac_to_pmc.append(guid)
        #     continue

        meta = corpus.getMetadataByGUID(guid)
        res = {"guid": guid,
               "title": doc.metadata["title"],
               "year": doc.metadata["year"],
               "authors": formatListOfAuthors(doc.metadata["authors"]),
               "num_sentences": len(doc.allsentences),
               "num_paragraphs": len(doc.allparagraphs),
               "num_sections": len(doc.allsections),
               "section_names": [s.get("header", "") for s in doc.allsections],
               "num_citations": len(doc.citations),
               "num_resolvable_citations": meta.get("num_resolvable_citations"),
               "num_in_collection_references": meta.get("num_in_collection_references"),
               "num_inlinks": len(meta.get("inlinks")),
               "num_outlinks": len(meta.get("outlinks")),
               "num_self_references":meta.get("num_self_references",0),
               "num_refs_with_overlapping_authors":meta.get("num_refs_with_overlapping_authors",0),
               }
        for az_type in AZ_ZONES_LIST:
            res[az_type] = 0
        for csc_type in CORESC_LIST:
            res[csc_type] = 0

        for sent in doc.allsentences:
            if sent.get("az") and sent.get("az") != "":
                res[sent["az"].upper()] += 1

            if sent.get("csc_type") and sent.get("csc_type") != "":
                res[sent["csc_type"].upper()] += 1

        results.append(res)

    print(aac_to_pmc)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(corpus.paths["output"], "corpus_metadata.csv"))


def main():
    all_stats = []

    # all_stats.append(get_corpus_metadata("AAC"))
    # all_stats.append(get_corpus_metadata("PMC_CSC"))
    # df = pd.DataFrame(get_corpus_metadata("AAC"))
    # df = df.transpose()
    # df.to_csv("aac_stats.csv")
    compile_corpus_metadata("AAC")


if __name__ == '__main__':
    main()
