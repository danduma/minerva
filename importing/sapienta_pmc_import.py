# Sapienta-annotated PMC corpus importer
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

def import_sapienta_pmc_corpus():
    """
        Do the importing of the Sapienta-annotated PMC corpus
    """
    from minerva.multi.config import MINERVA_ELASTICSEARCH_ENDPOINT
    importer=CorpusImporter(reader=SapientaJATSXMLReader())
    importer.collection_id="PMC_CSC"
    importer.import_id="initial"
    importer.generate_corpus_id=getPMC_CSC_corpus_id

##    cp.useLocalCorpus()
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\pmc_coresc", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)

    options={
##        "reload_xml_if_doc_in_collection": True, # default: False
##        "list_missing_references":True, # default: False
        "convert_and_import_docs":False, # default: True
##        "update_doc_references":False, # default: True
        "force_generate_resolvable_citations":True, # default: False

        "force_collection_id":"PMC_CSC", # this sets the collection_id again after updating references. Used for fixes
        "force_import_id":"fixed",
    }

##    corpus_import.FILES_TO_PROCESS_FROM=4500
##    corpus_import.FILES_TO_PROCESS_TO=10

##    importer.restartCollectionImport(options)

    importer.use_celery = True
    importer.importCorpus("g:\\nlp\\phd\\pmc_coresc\\inputXML", file_mask="*.xml", import_options=options)


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

def fix_broken_scidocs():
    """
        Iterates through the papers already in the collection. Tries to load
        their scidoc. If KeyError occurs, it loads the XML again
    """
    cp.useElasticCorpus()
    import minerva.multi.celery_app as celery_app
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\pmc_coresc",
            endpoint={"host":celery_app.MINERVA_ELASTICSEARCH_SERVER_IP,
            "port":celery_app.MINERVA_ELASTICSEARCH_SERVER_PORT})
    importer=CorpusImporter("PMC_CSC","initial", use_celery=True)
    importer.generate_corpus_id=getPMC_CSC_corpus_id
    importer.reloadSciDocsOnly("metadata.collection_id:\"PMC_CSC\"",
        "g:\\nlp\\phd\\pmc_coresc\\inputXML", "*.xml")

def set_collection_id(sql, new_id, guids=None):
    """
        Select a bunch of papers by a sql query,
    """
    if not guids:
        guids=cp.Corpus.SQLQuery(sql)

    for guid in guids:
        print(guid)
        meta=cp.Corpus.getMetadataByGUID(guid)
        meta["collection_id"]=new_id
        cp.Corpus.updatePaper(meta)
    print ("Total files %d" % len(guids))

def main():
##    import_sapienta_pmc_corpus()
##    fix_broken_scidocs()
##    set_collection_id("SELECT guid FROM papers where metadata.collection_id <> \"PMC_CSC\"", "AAC")
##    set_collection_id("SELECT guid FROM papers where metadata.collection_id = \"AAC\" and metadata.year > 2011", "PMC_CSC")

    guids=["cdee4ecf-59f0-4a19-bded-9b0f727e13ae", "0eebf17f-5aaa-414e-b3a3-6cfa62fee701", ]
    set_collection_id("", "PMC_CSC", guids)

    pass

if __name__ == '__main__':
    main()
