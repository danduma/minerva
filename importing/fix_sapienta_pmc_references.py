# Automate the tagging of each sentence in each document with its AZ/CoreSC
#
# Copyright (C) 2015 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


from minerva.proc.general_utils import loadFileText, writeFileText
import re, os, fnmatch, json
import minerva.db.corpora as cp


def selectRefListSection(original_text, pmc_file, pmc_id):
    """
        Returns the indexes to select the text in between <ref-list> tags
    """
    refs_start=re.search(r"<ref-list.*?>", original_text, flags=re.IGNORECASE)
    if not refs_start:
        print("File %s (pmcid: %s) has no <ref-list> section" % (pmc_file, pmc_id))
        raise ValueError

    refs_end=re.search("</ref-list>", original_text[refs_start.end():], re.IGNORECASE)
    if not refs_end:
        print("File %s (pmcid: %s) no </ref-list> tag" % (pmc_file, pmc_id))
        raise ValueError

    return refs_start.end(), refs_start.end()+refs_end.start()

def fixPaperReferences(annotated_file, pmc_file, pmc_id, original_text=None):
    """
        Replaces the <ref-list> section in `annotated_file` with that from
        `pmc_file`

        Checking that they actually the same file is done outside.
    """
    annotated_text=loadFileText(annotated_file)
    if not original_text:
        original_text=loadFileText(pmc_file)

    try:
        orig_start, orig_end=selectRefListSection(original_text,  pmc_file, pmc_id)
    except ValueError:
        return

    original_refs=original_text[orig_start:orig_end]

    try:
        annot_start, annot_end=selectRefListSection(annotated_text, annotated_file, getFilePMCID(annotated_file))
    except ValueError:
        return

    new_annotated_text=annotated_text[:annot_start]+original_text[orig_start:orig_end]+annotated_text[annot_end:]
    writeFileText(new_annotated_text, annotated_file)

def getPaperPMCID(filename):
    """
        Loads JATS file, returns its pmcid and the loaded text, or None if pmcid not found
    """
    original_text=loadFileText(filename)
    pmcid=re.search(r"<article-id pub-id-type=\"(?:pmcid|pmc)\">(.*?)</article-id>", original_text, re.IGNORECASE)
    if not pmcid:
        print("File %s has no original pmcid " % filename)
        return None, None
    return pmcid.group(1), original_text

def getFilePMCID(filename):
    """
        Returns the PMCID(numeric) from the filename
    """
    id=re.search(r"(\d+)[\_\.]", filename)
    if not id:
        return None
    return id.group(1)


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
                    id=getFilePMCID(filename)
                    if not id:
                        print("Can't get pmcid from file name: %s" % filename)
                    annotated_files[id]=fn

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



    file_point=r"g:\NLP\PhD\pmc_coresc\inputXML\data\scratch\mpx245\epmc\output\Out_PMC3834866_PMC3851806.xml.gz.gz\3843143_annotated.xml"

    reached_point=True

    file_mask=os.path.basename(pmc_path_mask)
    pmc_path_dir=os.path.dirname(pmc_path_mask)

    for dirpath, dirnames, filenames in os.walk(pmc_path_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename,file_mask):
                fn=os.path.join(dirpath,filename)
                if not reached_point and fn != file_point:
                    continue
                reached_point=True

                pmc_id, original_text=getPaperPMCID(fn)
                if pmc_id:
                    if pmc_id in annotated_files:
                        print fn
                        fixPaperReferences(annotated_files[pmc_id], fn, pmc_id, original_text)
##                        return

def main():
    drive="g"
    fixAllPapers(drive+r":\NLP\PhD\pmc_coresc\inputXML\*.xml",drive+r":\NLP\PhD\pmc\inputXML\*.nxml")
##    fixPaperReferences(r"G:\NLP\PhD\pmc_coresc\inputXML\Out_PMC549041_PMC1240567.xml.gz.gz\549043_done.xml",r"G:\NLP\PhD\pmc\inputXML\articles.A-B\BMC_Psychiatry\BMC_Psychiatry_2005_Feb_6_5_9.nxml",549043)
    pass


if __name__ == '__main__':
    main()
