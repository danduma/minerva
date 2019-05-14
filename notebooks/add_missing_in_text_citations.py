from proc.nlp_functions import AUTHOR_MARKER
import db.corpora as cp
from proc.general_utils import getRootDir
from scidoc.reference_formatting import formatAPACitationAuthors


def add_missing_citations(guid):
    doc = cp.Corpus.loadSciDoc(guid)
    inline_ref_mentions = []
    for ref in doc.references:
        # inline_ref_mentions.extend(ref["surnames"])
        # print(formatAPACitationAuthors(ref))
        inline_ref_mentions.append(formatAPACitationAuthors(ref))
        # names_from_institution=ref["institution"].replace()
        # all_text.extend()

    inline_ref_mentions = set(inline_ref_mentions)
    # print(surnames)

    for sent in doc.allsentences:
        text = sent["text"]
        for s in inline_ref_mentions:
            text = text.replace(s, AUTHOR_MARKER)

        sent["text"] = text
        print(text)


def main():
    from multi.config import MINERVA_ELASTICSEARCH_ENDPOINT
    import random
    cp.useElasticCorpus()
    cp.Corpus.setCorpusFilter("AAC")
    cp.Corpus.connectCorpus(getRootDir("aac"), endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)

    guids=cp.Corpus.listPapers(max_results=1000)
    # for guid in [random.choice(guids) for _ in range(10)]:
    for guid in ["e84fa104-ba83-49a9-9425-9cec9815d897"]:
        doc=cp.Corpus.loadSciDoc(guid)
        print(guid,"\n\n")
        print(doc.formatTextForExtraction(doc.getFullDocumentText()))

    # remove_author_names("33b50793-ccbb-4a6f-8ad1-615d7a47b73d")

if __name__ == '__main__':
    main()
