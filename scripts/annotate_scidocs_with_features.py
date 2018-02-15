# Annotate the features used to extract keywords later on
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function
import db.corpora as cp
from scidoc.scidoc import SciDoc
from ml.document_features import DocumentFeaturesAnnotator, en_nlp

import spacy
import json
from nltk import Tree
import os.path


ES_SERVER="koko.inf.ed.ac.uk"

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def find_root(docu):
    for token in docu:
        if token.head is token:
            return token

def annotate_one(guid):
    """
        Just for testing: annotate a sigle doc
    """
    def cache_save(cached_filename, doc):
        """
        """
        json.dump(doc.data,open(cached_filename, "w"))

    annotator=DocumentFeaturesAnnotator()
##    doc=cp.Corpus.loadSciDoc(guid)
    cached_filename="simple_cache/%s.json" % guid
    if os.path.isfile(cached_filename):
        try:
            doc=SciDoc(cached_filename)
        except:
            print(("Corrupt file: %s" % cached_filename))
            doc=cp.Corpus.loadSciDoc(guid)
            cache_save(cached_filename, doc)
    else:
        doc=cp.Corpus.loadSciDoc(guid)
        cache_save(cached_filename, doc)

    annotator.annotate(doc)

##    parse=en_nlp(u"")
##    parse.from_bytes(base64.decodestring(doc.allsentences[0]["parse"]))
    print("done")


def annotateSciDocs(doc_list):
    """
    """
    annotator=DocumentFeaturesAnnotator()


def main():
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("", endpoint={"host":ES_SERVER, "port":9200})
    annotate_one("c3eaadb3-0d3c-4d76-b485-83e8cb2af70f")
    pass

if __name__ == '__main__':
    main()
