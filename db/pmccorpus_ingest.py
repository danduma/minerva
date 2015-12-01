# main preprocessing and indexing of the PMC corpus
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from general_utils import *

import corpus_ingest

from xmlformats.jatsxml import *
from scidoc import SciDoc

import corpora

# ------------------------------------------------------------------------------
#   MAIN functions
# ------------------------------------------------------------------------------

def main():
    corpus_ingest.FILE_CONVERSION_FUNCTION=loadJATS_XML
    corpus_ingest.ingestCorpus("g:\\nlp\\phd\\pmc\\inputXML","g:\\nlp\\phd\\pmc\\", file_mask="*.nxml", start_at=148600)
    # latest start: 368758
    pass

if __name__ == '__main__':
    main()
