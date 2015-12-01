# Corpus ingesting functions
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import json, sys

from minerva.util.general_utils import *
from minerva.db.corpora import Corpus
from minerva.xmlformats.read_auto import AutoXMLReader
from copy import deepcopy

exception_file=None

def generateCSV(filename, index):
    """
        Write to a CSV all the important info about the database
    """

    f=codecs.open(filename,"wb","utf-8", errors="replace")
    line="Filename\tAuthors\tSurnames\tYear\tTitle\tNumRef\tInlinks\tIn-collection_references\tError\tNotes\n"
    f.write(line)

    for doc in index:
        line="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cs(doc["filename"]),
        doc["authors"],
        doc["surnames"],
        cs(doc["year"]),
        cs(doc["title"]),
        cs(str(doc["numref"])),
        cs(str(len(doc["inlinks"]))),
        cs(str(len(doc["outlinks"]))),
        cs(doc["error"]),
        cs(doc["notes"])
        )
##            line=unicode(line,"utf-8","replace")
        f.write(line)

    f.close()


class CorpusImporter(object):
    """
        Allows the importing of a coherent corpus procedurally in one go
    """

    def __init__(self, corpus_id="", reader=None):
        """
        """
        self.metadata_index=None
        self.files_to_ignore=[]
        self.file_mask="*.xml"
        self.reader=reader
        self.corpus_id=corpus_id
        self.import_id=""

    def makeNoteOfMissingReferences(self, missing_references):
        """
            Notes missing references in the database
        """
        #TODO test this
        for ref in missing_references:
            Corpus.addMissingPaper(copyDictExceptKeys(ref,["xml"]))

    def convertDoc(self, filename, corpus_id):
        """
            Reads the input XML and saves a SciDoc
        """
        if Corpus.getMetadataByField("corpus_id", corpus_id):
            # Doc is already in collection
            return

        if not self.reader:
            self.reader=AutoXMLReader()

        doc=self.reader.readFile(filename)
        doc.metadata["norm_title"]=normalizeTitle(doc.metadata["title"])
        doc.metadata["guid"]=Corpus.generateGUID(doc.metadata)
        Corpus.saveSciDoc(doc)
        return doc

    def addConvertedSciDocToDB(self, doc):
        """
            Extends metadata from doc and adds to database
        """
        meta=deepcopy(doc["metadata"])
##        meta["filepath"]=fn # original path of file, relative to /inputXML folder
        meta["corpus_id"]=meta["pm_id"] if meta.has_key("pm_id") else ""
        meta["norm_title"]=normalizeTitle(meta["title"])
        meta["numref"]=str(len(doc["references"]))
        meta["outlinks"]=[]
        meta["inlinks"]=[]
        meta["num_citations"]=len(doc["citations"])
        # this is for later processing and adding to database
        meta["num_in_collection_references"]=0
        meta["num_references"]=len(doc["references"])
        meta["num_matchable_citations"]=0
        meta["num_citations"]=0
        Corpus.addPaper(meta)

    def convertAllFilesAndAddToDB(self, ALL_INPUT_FILES, inputdir):
        """
            Loads each XML file, saves it as a SciDoc JSON file, adds its metadata to
            the database
        """
        time0=datetime.datetime.now()

        ALL_FILES_IN_DB=Corpus.listAllFilenames() or []

        count=0

        dot_every_xfiles=len(ALL_INPUT_FILES) / 2000
        if dot_every_xfiles == 0:
            dot_every_xfiles=1
        print len(ALL_INPUT_FILES),"files (progress . = "+str(dot_every_xfiles)+" files):"

        for fn in ALL_INPUT_FILES:
            filename=Corpus.paths.inputXML+fn
    ##        print "Processing",filename
    ##        match=Corpus.getPaperByFilename(os.path.basename(fn))
    ##        if not match:
##            try:
            if fn not in ALL_FILES_IN_DB:
                count+=1
                doc=self.convertDoc(os.path.join(Corpus.paths.inputXML,fn),self.corpus_id)
                self.addConvertedSciDocToDB(doc)

                if count % dot_every_xfiles ==0:
    ##                print count,"/",len(ALL_INPUT_FILES)
                    reportTimeLeft(count,len(ALL_INPUT_FILES), time0," -- importing")

##            except:
##                print "File: ",fn
##                print "EXCEPTION: ",sys.exc_info()[:2]
##                exception_file.write("File: "+fn+"\n")
##                exception_file.write("Exception: "+sys.exc_info()[:2].__repr__()+"\n\n")

    def importCorpus(self, root_input_dir, root_corpus_dir, options={}, start_at=0, maxfiles=10000000000):
        """
            Does all that is necessary for the initial import of the corpus
        """
        inputdir=ensureTrailingBackslash(root_input_dir)
        outputdir=ensureTrailingBackslash(root_corpus_dir)

        global_missing_references=[]

        print "Starting ingestion of corpus..."
        print "Creating database..."
        Corpus.connectCorpus(outputdir, initializing_corpus=True)

        Corpus.createFilesDB()
        Corpus.connectToDB()

        Corpus.metadata_index=self.metadata_index
        Corpus.FILES_TO_IGNORE=self.files_to_ignore

        global exception_file
        exception_file=open(os.path.join(Corpus.paths.fileDB_db,"exception_list.txt"),mode="a")

        all_input_files_fn=os.path.join(Corpus.paths.fileDB_db,"all_input_files.txt")
        ALL_INPUT_FILES=loadFileList(all_input_files_fn)
        if not ALL_INPUT_FILES:
            print "Listing all files..."
            ALL_INPUT_FILES=listAllFiles(inputdir,file_mask)
            saveFileList(ALL_INPUT_FILES,all_input_files_fn)

        print "Converting input files to SciDoc format and loading metadata..."
        self.convertAllFilesAndAddToDB(ALL_INPUT_FILES[start_at:maxfiles], inputdir)

        ALL_FILES=Corpus.listAllPapers()[start_at:start_at+maxfiles]
        print "Finding matchable references, populating database..."
        time0=datetime.datetime.now()
        time1=datetime.datetime.now()
        dot_every_xfiles=100
        count=0
        for doc_id in ALL_FILES:
            time1=datetime.datetime.now()
            doc_meta=Corpus.getMetadataByGUID(doc_id)
    ##        print "Processing in-collection references for ", doc_meta["filename"]

            doc_file=Corpus.loadSciDoc(doc_id)
            if not doc_file:
                print "Cannot load",doc_meta["filename"]
                continue

            matchable, in_collection_references, missing_references=Corpus.loadOrGenerateMatchableCitations(doc_file)

            doc_meta["num_in_collection_references"]=len(in_collection_references)
            doc_meta["num_references"]=len(doc_file["references"])
            doc_meta["num_matchable_citations"]=len(matchable)
            doc_meta["num_citations"]=len(doc_file["citations"])

            for ref in in_collection_references:
                match_meta=Corpus.getPaperByGUID(ref)
                if match_meta:
                    if match_meta["guid"] not in doc_meta["outlinks"]:
                        doc_meta["outlinks"].append(match_meta["guid"])
                    if doc_meta["guid"] not in match_meta["inlinks"]:
                        match_meta["inlinks"].append(doc_meta["guid"])
                        Corpus.updatePaper(match_meta)
                else:
                    # do something if no match?
                    pass

            Corpus.updatePaper(doc_meta)
            if options.get("list_missing_references", False):
                makeNoteOfMissingReferences(missing_references)

    ##        global_missing_references.extend(missing_references)
            count+=1
            if count % dot_every_xfiles ==0:
    ##            print "This just took ",datetime.datetime.now()-time1, "s"
                reportTimeLeft(count,len(ALL_FILES), time0," -- latest paper "+doc_meta["filename"])
                time1=datetime.datetime.now()

def simpleTest():
    """
    """
    Corpus.globalDBconn.close()
    f=open(os.join(Corpus.paths.fileDB_db,"missing_references.json"),"w")
    json.dump(global_missing_references,f)
    f.close()

def import_aac_corpus():
    """
    """
    importer=CorpusImporter()
    importer.corpus_id="AAC"
    importer.import_id="initial"
    importer.importCorpus("g:\\nlp\\phd\\aac\\inputxml\\","g:\\nlp\\phd\\aac",maxfiles=5)

def main():
    import_aac_corpus()
    pass

if __name__ == '__main__':
    main()
