from scidoc.citation_utils import annotatePlainTextCitationsInSentence
import db.corpora as cp
from proc.general_utils import getRootDir

def find_new_citations_in_aac():
    """
    Does another run through each AAC scidoc and tries to find citations that may have been missed

    :return:
    """
    from multi.celery_app import MINERVA_ELASTICSEARCH_ENDPOINT
    from tqdm import tqdm

    cp.useElasticCorpus()
    cp.Corpus.connectCorpus(getRootDir("aac"), endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
    cp.Corpus.setCorpusFilter("AAC")

    total_found = 0
    total_could_match = 0
    docs_with_new_ones = 0
    existing_citations = 0

    counter = tqdm(cp.Corpus.listPapers())

    for guid in counter:
        counter.set_description("{} docs_with_new_ones, {} total_found, {} total_could_match, {} existing_citations, ".format(docs_with_new_ones, total_found, total_could_match, existing_citations))
        doc = cp.Corpus.loadSciDoc(guid)

        for sent in doc.allsentences:
            existing_citations += len(sent.get("citations", []))
            new_citations, citations_found = annotatePlainTextCitationsInSentence(sent, doc)
            if len(citations_found) > 0:
                # print(len(new_citations),":",new_citations)
                total_found += len(citations_found)
                total_could_match += len(new_citations)
                docs_with_new_ones += 1
                # print("\n NEW CITATION:", sent["text"])
                # print(citations_found)
                # print()
            else:
                if len(sent.get("citations", [])) > 0:
                    # print("ALREADY ANNOTATED:", sent["text"], "\n")
                    pass

        cp.Corpus.saveSciDoc(doc)

    print("Total citations found: ", total_found)
    print("Total citations could match: ", total_could_match)
    print("Docs with new citations: ", docs_with_new_ones)
    print("Previously annotated citations: ", existing_citations)


if __name__ == '__main__':
    find_new_citations_in_aac()