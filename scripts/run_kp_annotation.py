from db.ez_connect import ez_connect
import db.corpora as cp
from proc.query_extraction import SentenceQueryExtractor
from proc.keyphrase_annotation import getMatchingKPsInContext, getFreqTextRankKeyPhrases, annotateDocWithKPs

from multi.tasks import annotateDocWithKPsTask

def testContextKPs(doc, best_kps):
    resolvable = cp.Corpus.loadOrGenerateResolvableCitations(doc)
    resolvable = resolvable["resolvable"]

    extractor = SentenceQueryExtractor()

    all_kps = set(best_kps)

    for citation in resolvable:

        params = {"parameters": ["2up_2down"],
                  "docfrom": doc,
                  "cit": citation["cit"],
                  "dict_key": "text",
                  "extract_only_plain_text": True,
                  "method_name": "inlink_context"
                  }
        contexts = extractor.extractMulti(params)
        for generated_context in contexts:
            print(generated_context["text"])
            intersection = getMatchingKPsInContext(generated_context["text"], all_kps)
            print(intersection)
            print()

            # context = addILCMetadata({"text": generated_context["text"]}, docfrom, doc_target)
            # all_contexts[generated_context["params"]].append(context)  # ["params"][0] is wleft


def test_main(corpus):
    corpus = ez_connect("AAC", "koko")
    guids = corpus.listPapers(max_results=10, sort="metadata.num_in_collection_references:desc")
    for guid in guids[3:4]:
        doc = corpus.loadSciDoc(guid)
        # text = doc.formatTextForExtraction(doc.getFullDocumentText())
        print(doc.metadata["title"])
        # print(getNgramFrequency())
        # print(getTextRankKPs(text))
        # print_out_different_weight_comb(text,
        #                                 window_size=2,
        #                                 mu=2,
        #                                 max_kp=100)
        # print(getNgramsFromSections(doc))
        best_kps = []
        best_kps = getFreqTextRankKeyPhrases(doc)
        print([" ".join(k) for k in best_kps])
        print()
        testContextKPs(doc, best_kps)
        # print_ngram_freq(text, 3)
        # testFrequencyKeyphrases(text)


def annotateSciDocsWithKPs(corpus, use_celery=False):
    from tqdm import tqdm

    # guids = corpus.listPapers("metadata.num_in_collection_references:>0 AND metadata.year:>2013", sort="metadata.num_in_collection_references:desc",
    #                           max_results=10000000)
    guids = corpus.listPapers()
    # for guid in tqdm(guids[10230:]):
    for guid in tqdm(reversed(guids)):
        if use_celery:
            annotateDocWithKPsTask.apply_async(
                args=[guid],
                kwargs={},
                queue="annotate_keyphrases"
                )
        else:
            annotateDocWithKPs(guid)


def main():
    corpus = ez_connect("AAC", "koko")
    # corpus = ez_connect("PMC_CSC", "koko")
    # test_main(corpus)
    annotateSciDocsWithKPs(corpus, use_celery=False)


if __name__ == '__main__':
    main()