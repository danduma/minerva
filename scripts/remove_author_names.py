from proc.nlp_functions import AUTHOR_MARKER
import db.corpora as cp
from proc.general_utils import getRootDir
from scidoc.reference_formatting import formatAPACitationAuthors


def remove_author_names(guid):
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
    from db.ez_connect import ez_connect
    corpus = ez_connect("AAC", "koko")

    # guids = corpus.listPapers(max_results=1000)
    # for guid in [random.choice(guids) for _ in range(10)]:
    # for guid in ["e84fa104-ba83-49a9-9425-9cec9815d897"]:
    for guid in ["7120018d-fe25-4e1d-a889-93a76f8c068d"]:
        doc = cp.Corpus.loadSciDoc(guid)
        print(guid, "\n\n")
        print(doc.formatTextForExtraction(doc.getFullDocumentText()))

    # remove_author_names("33b50793-ccbb-4a6f-8ad1-615d7a47b73d")


if __name__ == '__main__':
    main()
