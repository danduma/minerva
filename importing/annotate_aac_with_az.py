# Uses either external output from AZprime or own classifier to label whole
# corpus with AZ
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
    from minerva.multi.config import MINERVA_ELASTICSEARCH_ENDPOINT
##    cp.useLocalCorpus()
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\aac", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
    cp.Corpus.setCorpusFilter("AAC")

def exportSciXML():
    """
        Exports all scidocs with the selected collection_id to AZPrime XML in the output dir of the corpus
    """
    papers=cp.Corpus.listPapers(max_results=sys.maxint)

    writer=AZPrimeWriter()
    writer.save_pos_tags=True
##    papers=papers[3894:]
    progress=ProgressIndicator(True, len(papers),False)
    print("Exporting SciXML files")
    for guid in papers:
        doc=cp.Corpus.loadSciDoc(guid)
        if len(doc.allsentences) < 1:
            continue
        writer.write(doc, os.path.join(cp.Corpus.paths.output, doc.metadata["guid"]+".pos.xml"))
        cp.Corpus.saveSciDoc(doc)
        progress.showProgressReport("Exporting -- %s" % guid)


def ownAZannot(export_annots=False):
    """
        Annotates each sentence using own classifier
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


def loadAZLabels(annot_dir=""):
    """
        Loads generated AZ labels from AZPrime output
    """
    if annot_dir=="":
        annot_dir=cp.Corpus.paths.output

    papers=cp.Corpus.listPapers()

    print("Loading AZPrime labels...")
    progress=ProgressIndicator(True, len(papers),False)

    for guid in papers:
        filename=os.path.join(annot_dir, guid+".pred.txt")
        if os.path.exists(filename):

            doc=cp.Corpus.loadSciDoc(guid)
            f=file(filename, "r")
            lines=f.readlines()
            allsentences=[s for s in doc.allsentences if s.get("type","") == "s"]

            if len(lines) != len(allsentences):
                print("Number of tags mismatch! %d != %d -- %s" % (len(lines), len(allsentences), guid))
                lines=["" for n in range(len(allsentences))]
##            else:
##                print("No mismatch! %d != %d -- %s" % (len(lines), len(doc.allsentences), guid))

            for index,sent in enumerate(allsentences):
                sent["az"]=lines[index].strip()
            cp.Corpus.saveSciDoc(doc)
        else:
            print("Cannot find annotation file for guid %s" % guid)

        progress.showProgressReport("Loading labels -- %s" % guid)

def testLabels():
    """
    """
    guid="f7921eed-89bc-4f38-a794-7c9a5878a7ee"
    writer=AZPrimeWriter()
    writer.save_pos_tags=True

    doc=cp.Corpus.loadSciDoc(guid)

    writer.write(doc, os.path.join(cp.Corpus.paths.output, doc.metadata["guid"]+".pos.xml"))


def main():
    connectToCorpus()
##    exportSciXML()
##    testLabels()
    loadAZLabels(r"G:\NLP\PhD\aac\azprime_output")
##    ownAZannot()
    pass


if __name__ == '__main__':
    main()
