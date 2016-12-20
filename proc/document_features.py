# DocumentFeaturesAnnotator: annotates all features in a document to be used for
# keyword extraction
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import base64
import spacy
en_nlp = spacy.load('en')

class DocumentFeaturesAnnotator(object):
    """
        Pre-annotates all features in a scidoc to be used for keyword extraction
    """
    def __init__(self):
        pass

    def annotate_section(self, section):
        """
        """

    def annotate_sentence(self, s):
        """
        """

    def select_sentences(self, doc, window_size=3):
        """
            Selects which sentences to annotate
        """

        to_annotate=[0 for s in range(len(doc.allsentences))]
        for index, s in enumerate(doc.allsentences):
            if len(s.get("citations",[])) > 0:
                for cnt in range (max(index-2,0),min(index+2,len(to_annotate))+1):
                    to_annotate[cnt] = True

    def annotate(self, doc):
        """
            Adds all features to a scidoc
        """
        for sent in doc.allsentences:
            parse = en_nlp(sent["text"])
            sent["parse"]=base64.encodestring(parse.to_bytes())


def main():
    pass

if __name__ == '__main__':
    main()
