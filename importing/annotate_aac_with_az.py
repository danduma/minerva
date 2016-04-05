# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
from __future__ import print_function

import json, os, sys

import minerva.db.corpora as cp
##from minerva.scidoc.xmlformats.read_paperxml import PaperXMLReader
from minerva.scidoc.xmlformats.write_azprime_scixml import AZPrimeWriter
##from minerva.scidoc.xmlformats.read_scixml import SciXMLReader

from corpus_import import CorpusImporter
import corpus_import

from aac_import import AANReferenceMatcher, getACL_corpus_id, import_aac_corpus
from minerva.proc.general_utils import writeFileText
from minerva.scidoc.render_content import SciDocRenderer

from minerva.proc.results_logging import ProgressIndicator


def connectToCorpus():
    """
    """
    from minerva.squad.config import MINERVA_ELASTICSEARCH_ENDPOINT
##    cp.useLocalCorpus()
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\aac", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
    cp.Corpus.setCorpusFilter("AAC")

def exportSciXML():
    """
    """
    papers=cp.Corpus.listPapers(max_results=sys.maxint)

    writer=AZPrimeWriter()
    writer.save_pos_tags=True
##    papers=papers[3894:]
    progress=ProgressIndicator(True, len(papers),False)
    print("Exporting SciXML files")
    for guid in papers:
        doc=cp.Corpus.loadSciDoc(guid)
        writer.write(doc, os.path.join(cp.Corpus.paths.output, doc.metadata["guid"]+".pos.xml"))
        cp.Corpus.saveSciDoc(doc)
        progress.showProgressReport("Exporting -- %s" % guid)


def ownAZannot(export_annots=False):
    """
    """
    from minerva.az.az_cfc_classification import AZannotator

    annot=AZannotator("trained_az_classifier.pickle")

    papers=cp.Corpus.listPapers(max_results=sys.maxint)

    writer=AZPrimeWriter()
    writer.save_pos_tags=True
##    papers=papers[:1]
    progress=ProgressIndicator(True, len(papers),False)

    print("Producing annotations for SciDocs...")
    for guid in papers:
        doc=cp.Corpus.loadSciDoc(guid)
        annot.annotateDoc(doc)
        if export_annots:
            output_filename=os.path.join(cp.Corpus.paths.output, doc.metadata["guid"]+".annot.txt")
            output_file=open(output_filename,"w")
            for sentence in doc.allsentences:
                output_file.write(sentence.get("az","")+"\n")
            output_file.close()
        else:
            cp.Corpus.saveSciDoc(doc)

        progress.showProgressReport("Annotating -- %s" % guid)


def loadAZLabels():
    """
    """
    papers=cp.Corpus.listPapers(max_results=10)

    writer=AZPrimeWriter()
    for guid in papers[:1]:
        filename=os.path.join(cp.Corpus.paths.output, doc.metadata["guid"]+".annot.txt")
        if os.path.exists(filename):
            doc=cp.Corpus.loadSciDoc(guid)
            f=file(filename, "r")
            lines=f.readlines()
            for index,sent in enumerate(doc.allsentences):
                sent["az"]=lines[index]
            cp.Corpus.saveSciDoc(doc)
        else:
            print("Cannot find annotation file for guid %s" % guid)

def annotateCorpus():
    """
    """
    pass


def main():
    connectToCorpus()
##    exportSciXML()
    ownAZannot()
    pass


if __name__ == '__main__':
    main()
