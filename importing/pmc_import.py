# PMC corpus importer
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

from __future__ import absolute_import
import os, json, re

from .corpus_import import CorpusImporter
from . import corpus_import
import db.corpora as cp
from scidoc.xmlformats.read_jatsxml import JATSXMLReader
from proc.nlp_functions import tokenizeText, basic_stopwords
from string import punctuation

corpus_import.FILES_TO_PROCESS_FROM=0

def import_pmc_corpus():
    """
        Do the importing of the PMC corpus
    """

    def getPMC_corpus_id(filename):
        """
            Returns the PMC id for a file
        """
##        return os.path.split(filename)[1].replace("-paper.xml","").lower()
        return ""

    importer=CorpusImporter(reader=JATSXMLReader())
    importer.collection_id="PMC"
    importer.import_id="initial"
##    importer.generate_corpus_id=getACL_corpus_id

##    cp.useLocalCorpus()
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\pmc")

    importer.importCorpus("g:\\nlp\\phd\\pmc\\inputXML",file_mask="*.nxml")
    importer.updateInCollectionReferences(cp.Corpus.listPapers(), {})



def main():
    import_pmc_corpus()
    pass

if __name__ == '__main__':
    main()
