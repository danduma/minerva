# AAC corpus importer
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import os, json, re

import corpus_import
from corpus_import import CorpusImporter
import minerva.db.corpora as cp
from minerva.scidoc.xmlformats.read_sapienta_jatsxml import SapientaJATSXMLReader

def import_sapienta_pmc_corpus():
    """
        Do the importing of the Sapienta-annotated PMC corpus
    """

    def getPMC_CSC_corpus_id(filename):
        """
            Returns the ACL id for a file
        """
        fn=os.path.split(filename)[1]
        match=re.search(r"(\d+)_(?:annotated|done)\.xml",fn.replace("-paper.xml","").lower(),flags=re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return fn

    importer=CorpusImporter(reader=SapientaJATSXMLReader())
    importer.collection_id="PMC_CSC"
    importer.import_id="initial"
    importer.generate_corpus_id=getPMC_CSC_corpus_id

##    cp.useLocalCorpus()
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\pmc_coresc")

    options={
##        "list_missing_references":True, # default: False
        "convert_and_import_docs":False, # default: True
    }

##    corpus_import.FILES_TO_PROCESS_FROM=4500
##    corpus_import.FILES_TO_PROCESS_TO=500

##    importer.restartCollectionImport(options)

    importer.use_celery = True
    importer.importCorpus("g:\\nlp\\phd\\pmc_coresc\\inputXML",file_mask="*.xml", import_options=options)


def fix_authors_full_corpus():
    """
        Fixes authors in each metadata entry having a "papers" key which they
        shouldn't
    """
    from minerva.proc.results_logging import ProgressIndicator
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\pmc_coresc")
    guids=cp.Corpus.listPapers()
    progress=ProgressIndicator(True, len(guids), True)
    for guid in guids:
        doc_meta=cp.Corpus.getMetadataByGUID(guid)
        new_authors=[]
        for old_author in doc_meta.authors:
            del old_author["papers"]
        cp.Corpus.updatePaper(doc_meta)
        progress.showProgressReport("Removing redundant author information")


def main():
    import_sapienta_pmc_corpus()

    pass

if __name__ == '__main__':
    main()
