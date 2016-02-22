# Corpus ingesting functions
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import json, sys, os, datetime, fnmatch
##from logging import info,warning,error
from copy import deepcopy

##from minerva.proc.general_utils import *
from minerva.proc.general_utils import (ensureTrailingBackslash, loadFileList,
saveFileList, normalizeTitle, reportTimeLeft)

import minerva.db.corpora as cp
from minerva.scidoc.xmlformats.read_auto import AutoXMLReader

exception_file=None

FILES_TO_PROCESS_FROM=0
FILES_TO_PROCESS_TO=1000000000

class CorpusImporter(object):
    """
        Allows the importing of a coherent corpus procedurally in one go
    """

    def __init__(self, collection_id="", import_id="", reader=None):
        """
        """

        def getDefault_corpus_id(filename):
            """
                Default corpus_id generator
            """
            return os.path.split(filename)[1].replace(".xml","").lower()

        self.metadata_index=None
        self.files_to_ignore=[]
        self.file_mask="*.xml"
        self.reader=reader
        self.collection_id=collection_id
        self.import_id=import_id
        self.generate_corpus_id=getDefault_corpus_id
        self.num_processes=1

    def makeNoteOfMissingReferences(self, missing_references):
        """
            Notes missing references in the database
        """
        #TODO test makeNoteOfMissingReferences
        for ref in missing_references:
            cp.Corpus.addMissingPaper(copyDictExceptKeys(ref,["xml"]))

    def convertDoc(self, filename, corpus_id):
        """
            Reads the input XML and saves a SciDoc
        """
##        if cp.Corpus.getMetadataByField("corpus_id", corpus_id):
##            # Doc is already in collection
##            return

        if not self.reader:
            self.reader=AutoXMLReader()

        doc=self.reader.readFile(filename)
        doc.metadata["norm_title"]=normalizeTitle(doc.metadata["title"])
        doc.metadata["guid"]=cp.Corpus.generateGUID(doc.metadata)
        doc.metadata["corpus_id"]=corpus_id
        cp.Corpus.saveSciDoc(doc)
        return doc

    def addConvertedSciDocToDB(self, doc):
        """
            Extends metadata from doc and adds to database
        """
        meta=deepcopy(doc["metadata"])

        if meta.get("corpus_id","")=="":
            meta["corpus_id"]=meta["pm_id"] if meta.has_key("pm_id") else ""

        meta["norm_title"]=normalizeTitle(meta["title"])
        meta["numref"]=str(len(doc["references"]))
        meta["outlinks"]=[]
        meta["inlinks"]=[]
        meta["num_citations"]=len(doc["citations"])

        # this is for later processing and adding to database
        meta["num_in_collection_references"]=0
        meta["num_references"]=len(doc["references"])
        meta["num_resolvable_citations"]=0
        meta["num_citations"]=0
        meta["import_id"]=self.import_id
        meta["collection_id"]=self.collection_id
        cp.Corpus.addPaper(meta)

    def convertAllFilesAndAddToDB(self, ALL_INPUT_FILES, inputdir):
        """
            Loads each XML file, saves it as a SciDoc JSON file, adds its metadata to
            the database
        """
        time0=datetime.datetime.now()

        ALL_FILES_IN_DB=cp.Corpus.listAllPapers(field="filename") or []

        count=0

        dot_every_xfiles=len(ALL_INPUT_FILES) / 2000
        if dot_every_xfiles == 0:
            dot_every_xfiles=1
        print (len(ALL_INPUT_FILES),"files (progress . = "+str(dot_every_xfiles)+" files):")

        # main loop over all files
        for fn in ALL_INPUT_FILES[FILES_TO_PROCESS_FROM:FILES_TO_PROCESS_TO]:
            filename=cp.Corpus.paths.inputXML+fn
            print("Processing",filename)
            corpus_id=self.generate_corpus_id(fn)
    ##        match=cp.Corpus.getPaperByFilename(os.path.basename(fn))
    ##        if not match:
##            try:
            if fn not in ALL_FILES_IN_DB:
##                assert "A94-1033-paper.xml" not in fn
                count+=1
                try:
                    doc=self.convertDoc(os.path.join(cp.Corpus.paths.inputXML,fn),corpus_id)
                except ValueError:
                    print("ERROR: Couldn't convert %s" % fn)
                    continue

                self.addConvertedSciDocToDB(doc)
                if count % dot_every_xfiles ==0:
    ##                print count,"/",len(ALL_INPUT_FILES)
                    reportTimeLeft(count,len(ALL_INPUT_FILES), time0," -- importing")

##            except:
##                print "File: ",fn
##                print "EXCEPTION: ",sys.exc_info()[:2]
##                exception_file.write("File: "+fn+"\n")
##                exception_file.write("Exception: "+sys.exc_info()[:2].__repr__()+"\n\n")

    def updateInCollectionReferences(self, ALL_GUIDS, options={}):
        """
            For every guid, it matches its in-collection references, and its
            resolvable citations

            Args:
                ALL_GUIDS: list of guids
        """
        print("Finding resolvable references, populating database...")
        time0=datetime.datetime.now()
        time1=datetime.datetime.now()
        dot_every_xfiles=100
        count=0
        for doc_id in ALL_GUIDS[FILES_TO_PROCESS_FROM:FILES_TO_PROCESS_TO]:
            time1=datetime.datetime.now()
            doc_meta=cp.Corpus.getMetadataByGUID(doc_id)
    ##        print "Processing in-collection references for ", doc_meta["filename"]

            doc_file=cp.Corpus.loadSciDoc(doc_id)
            if not doc_file:
                print("Cannot load",doc_meta["filename"])
                continue

            tin_can=cp.Corpus.loadOrGenerateResolvableCitations(doc_file)

            resolvable=tin_can["resolvable"]
            in_collection_references=tin_can["outlinks"]
            missing_references=tin_can.get("missing_references",[])
            doc_meta["num_in_collection_references"]=len(in_collection_references)
            doc_meta["num_references"]=len(doc_file["references"])
            doc_meta["num_resolvable_citations"]=len(resolvable)
            doc_meta["num_citations"]=len(doc_file["citations"])

            for ref in in_collection_references:
                match_meta=cp.Corpus.getPaperByGUID(ref)
                if match_meta:
                    if match_meta["guid"] not in doc_meta["outlinks"]:
                        doc_meta["outlinks"].append(match_meta["guid"])
                    if doc_meta["guid"] not in match_meta["inlinks"]:
                        match_meta["inlinks"].append(doc_meta["guid"])
                        cp.Corpus.updatePaper(match_meta)
                else:
                    # do something if no match?
                    pass

            cp.Corpus.updatePaper(doc_meta)
            if options.get("list_missing_references", False):
                makeNoteOfMissingReferences(missing_references)

    ##        global_missing_references.extend(missing_references)
            count+=1
            if count % dot_every_xfiles ==0:
    ##            print "This just took ",datetime.datetime.now()-time1, "s"
                reportTimeLeft(count,len(ALL_GUIDS), time0," -- latest paper "+doc_meta["filename"])
                time1=datetime.datetime.now()

    def listAllFiles(self, start_dir, file_mask):
        """
            Creates an ALL_FILES list with relative paths from the start_dir, saves it
        """
        ALL_FILES=[]

        for dirpath, dirnames, filenames in os.walk(start_dir):
            for filename in filenames:
                if fnmatch.fnmatch(filename,file_mask) and filename not in cp.Corpus.FILES_TO_IGNORE:
                        fn=os.path.join(dirpath,filename)
                        fn=fn.replace(start_dir,"")
                        ALL_FILES.append(fn)

##        saveFileList(ALL_FILES,cp.Corpus.paths.fileDB+"db\\ALL_INPUT_FILES.txt")

        print("Total files:",len(ALL_FILES))
        return ALL_FILES

    def selectRandomInputFiles(self, howmany, file_mask="*.xml"):
        """
            Of all input files, it picks number of random ones.

            Tries to open ALL_INPUT_FILES.txt in filedb/db directory. If not
            present, it makes sure all directories exist, calls listAllFiles()
            to generate the file.

        """
        try:
            all_files_file=file(os.path.join(cp.Corpus.paths.fileDB,"ALL_INPUT_FILES.txt"), "r")
            all_files=all_files_file.readlines()
        except:
            cp.Corpus.createDefaultDirs()
            all_files=self.listAllFiles(cp.Corpus.paths.inputXML, file_mask)

        result=[]

        for cnt in range(howmany):
            result.append(all_files[random.randint(0,len(all_files))].strip("\n"))
        return result


    def importCorpus(self, root_input_dir, file_mask="*.xml", options={}, start_at=0, maxfiles=10000000000):
        """
            Does all that is necessary for the initial import of the corpus
        """
        inputdir=ensureTrailingBackslash(root_input_dir)
##        outputdir=ensureTrailingBackslash(root_corpus_dir)

        global_missing_references=[]

        print("Starting ingestion of corpus...")
        print("Creating database...")
##        cp.Corpus.connectCorpus(outputdir, initializing_corpus=True)

        cp.Corpus.createAndInitializeDatabase()
        cp.Corpus.connectToDB()

        cp.Corpus.metadata_index=self.metadata_index
        cp.Corpus.FILES_TO_IGNORE=self.files_to_ignore

##        global exception_file
##        exception_file=open(os.path.join(cp.Corpus.paths.fileDB_db,"exception_list.txt"),mode="a")

        all_input_files_fn=os.path.join(cp.Corpus.paths.fileDB,"all_input_files.txt")
        ALL_INPUT_FILES=loadFileList(all_input_files_fn)
        if not ALL_INPUT_FILES:
            print("Listing all files...")
            ALL_INPUT_FILES=self.listAllFiles(inputdir,file_mask)
##            saveFileList(ALL_INPUT_FILES,all_input_files_fn)

        print("Converting input files to SciDoc format and loading metadata...")
        self.convertAllFilesAndAddToDB(ALL_INPUT_FILES, inputdir)

        ALL_GUIDS=cp.Corpus.listAllPapers()
        self.updateInCollectionReferences(ALL_GUIDS, options)


def simpleTest():
    """
    """
    cp.Corpus.globalDBconn.close()
    f=open(os.join(cp.Corpus.paths.fileDB_db,"missing_references.json"),"w")
    json.dump(global_missing_references,f)
    f.close()


def main():

    pass

if __name__ == '__main__':
    main()
