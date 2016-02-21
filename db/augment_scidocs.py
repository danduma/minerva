# Automate the tagging of each sentence in each document with its AZ/CoreSC
#
# Copyright (C) 2015 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


from general_utils import *
from scidoc import SciDoc

import corpora as cp


def augmentSciDocWithAZ(guid):
    """
        Annotates a single file in the corpus with AZ and CFC
    """
    if len(cp.Corpus.annotators)==0:
        cp.Corpus.loadAnnotators()
    doc=cp.Corpus.loadSciDoc(guid)
    cp.Corpus.annotateDoc(doc)
    cp.Corpus.saveSciDoc(doc)

def augmentAllSciDocsWithAZ():
    """
        Goes through the files in the cp.Corpus, annotating all of them with AZ and CFC
    """
    guids=cp.Corpus.listAllPapers()
    for guid in guids:
        print guid
        augmentSciDocWithAZ(guid)

def augmentSciDocWithSapienta(doc,sapienta_filename):
    """
        Read Sapienta XML output file, augment a SciDoc with the CoreSC data
        just by matching the XML strings
    """
    rxcmp=re.compile(r'<s\ssid="(\d+)"><CoreSc1\sadvantage="(.*?)"\sconceptID="(.*?)"\snovelty="(.*?)"\stype="(.*?)"',re.IGNORECASE)
    f=codecs.open(sapienta_filename,"r",errors="ignore")
    text=f.read()
    for match in rxcmp.finditer(text):
        s_id="s"+str(match.group(1))
        if s_id in doc.element_by_id:
            s=doc.element_by_id[s_id]
            s["csc_type"]=match.group(5)
            s["csc_adv"]="" if match.group(2)=="None" else match.group(2)
            s["csc_nov"]="" if match.group(4)=="None" else match.group(4)

def augmentAllSciDocsWithSapienta(sapienta_output_dir):
    """
        Goes through all the SciDocs in a the cp.Corpus. If there is a matching file
        with GUID+"_annotated.xml" in the sapienta_output_dir then it calls
        augmentSciDocWithSapienta()
    """
    sapienta_output_dir=ensureTrailingBackslash(sapienta_output_dir)
    guids=cp.Corpus.listAllPapers()
    for guid in guids:
        print guid
        doc=cp.Corpus.loadSciDoc(guid)
        filename=doc.metadata["filename"]
        sapienta_output_dir=ensureTrailingBackslash(sapienta_output_dir)
        sapienta_filename=sapienta_output_dir+getFileName(filename)+"_annotated.xml"

        if exists(sapienta_filename):
            augmentSciDocWithSapienta(doc,sapienta_filename)
            cp.Corpus.saveSciDoc(doc)
    pass

def main():
##    augmentAllSciDocsWithSapienta(r"C:\NLP\PhD\bob\daniel_converted_out")
##    augmentAllSciDocsWithAZ()
##    augmentSciDocWithAZ("P95-1026")

    pass


if __name__ == '__main__':
    main()
