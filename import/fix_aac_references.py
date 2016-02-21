# Automate the tagging of each sentence in each document with its AZ/CoreSC
#
# Copyright (C) 2015 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


from minerva.proc.general_utils import loadFileText, writeFileText
import re, os, fnmatch, json
import corpora as cp


def selectRefListSection(original_text,  pmc_file, pmc_id):
    """
        Returns the indexes to select the text in between <ref-list> tags
    """
    refs_start=re.search("<ref-list>", original_text, re.IGNORECASE)
    if not refs_start:
        print("File %s (pmcid: %s) has no original <ref-list> section" % (pmc_file, pmcid))
        return None

    refs_end=re.search("</ref-list>", original_text[refs_start.end():], re.IGNORECASE)
    if not refs_end:
        print("File %s (pmcid: %s) no </ref-list> tag" % (pmc_file, pmcid))
        return None

    return refs_start.end(), refs_start.end()+refs_end.start()

def fixPaperReferences(annotated_file, pmc_file, original_text=None):
    """
        Replaces the <ref-list> section in `annotated_file` with that from
        `pmc_file`

        Checking that they actually the same file is done outside.
    """
    annotated_text=loadFileText(annotated_file)
    if not original_text:
        original_text=loadFileText(pmc_file)

    orig_start, orig_end=selectRefListSection(original_text, pmc_file, pmc_id)
    original_refs=original_text[orig_start:orig_end]

    annot_start, annot_end=selectRefListSection(original_text)
    original_refs=original_text[orig_start:orig_end]

    annotated_text=annotated_text[:annot_start]+original_text[orig_start:orig_end]+annotated_text[annot_end:]
    writeFileText(annotated_text, annotated_file)

def getPaperPMCID(filename):
    """
        Loads JATS file, returns its pmcid and the loaded text, or None if pmcid not found
    """
    original_text=loadFileText(filename)
    pmcid=re.search(r"<article-id pub-id-type=\"pmcid\">(.*?)</article-id>", original_text, re.IGNORECASE)
    if not pmcid:
        print("File %s has no original pmcid " % filename)
        return None
    return pmcid.group(1), original_text

def listAllFilesWithID(annotated_path_mask):
    """
        Creates an annotated_files dict with relative paths from the start_dir
    """
    annotated_files={}

    start_dir=os.path.dirname(annotated_path_mask)
    file_mask=os.path.basename(annotated_path_mask)

    for dirpath, dirnames, filenames in os.walk(start_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename,file_mask):
                    fn=os.path.join(dirpath,filename)
                    id=re.search(r"(\d+)[\_\.]", filename)
                    if not id:
                        print("Can't get pmcid from file name: %s" % filename)
                    annotated_files[id.group(1)]=fn

    print "Total files:",len(annotated_files)
    return annotated_files


def fixAllPapers(annotated_path_mask, pmc_path_mask):
    """
        Walks through the original papers path and for each tries to find the
        annotated paper and fixes the references
    """
    annotated_path_dir=os.path.dirname(annotated_path_mask)
    ids_file=os.path.join(annotated_path_dir,"all_files.json")
    if not os.path.exists(ids_file):
        annotated_files=listAllFilesWithID(annotated_path_mask)
        json.dump(annotated_files,file(ids_file, "w"))
    else:
        annotated_files=json.load(file(ids_file, "r"))

    file_mask=os.path.basename(annotated_path_mask)

    for dirpath, dirnames, filenames in os.walk(annotated_path_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename,file_mask):
                fn=os.path.join(dirpath,filename)
                pmcid, original_text=getPaperPMCID(fn)
                if pmcid:
                    if pmcid in annotated_files:
                        fixPaperReferences(annotated_files[pmcid], fn, original_text)

def main():
    drive="g"
    fixAllPapers(drive+r":\NLP\PhD\pmc_coresc\inputXML\*.xml",drive+r":\NLP\PhD\pmc\inputXML\*.nxml")
    pass


if __name__ == '__main__':
    main()
