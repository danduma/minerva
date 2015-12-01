# Corpus ingesting functions
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import json, sys

from minerva.util.general_utils import *
import minerva.db.corpora as corpora

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


def makeNoteOfMissingReferences(missing_references):
    """

    """
    # do what with these? save somewhere
    for ref in missing_references:
        corpora.Corpus.addMissingPaper(copyDictExceptKeys(ref,["xml"]))


def convertAllFilesAndAddToDB(ALL_INPUT_FILES, inputdir):
    """
        Loads each XML file, saves it as a SciDoc JSON file, adds its metadata to
        the database
    """
    time0=datetime.datetime.now()

    ALL_FILES_IN_DB=corpora.Corpus.listAllFilenames() or []

    count=0

    dot_every_xfiles=len(ALL_INPUT_FILES) / 2000
    print len(ALL_INPUT_FILES),"files (progress . = "+str(dot_every_xfiles)+" files):"

    for fn in ALL_INPUT_FILES:
        filename=corpora.Corpus.dir_inputXML+fn
##        print "Processing",filename
##        match=corpora.Corpus.getPaperByFilename(os.path.basename(fn))
##        if not match:
        try:
            if fn not in ALL_FILES_IN_DB:
                doc=corpora.Corpus.loadSciDoc(corpora.Corpus.getFileUID(filename))
                if not doc:
                    doc=FILE_CONVERSION_FUNCTION(filename)
                    corpora.Corpus.saveSciDoc(doc)

                newdoc={}
                meta=doc["metadata"]
                newdoc["filename"]=meta["filename"] # original file name
                newdoc["filepath"]=fn # original path of file, relative to /inputXML folder
                newdoc["guid"]=meta["guid"]
                newdoc["doi"]=meta["doi"] if meta.has_key("doi") else ""
                newdoc["corpus_id"]=meta["pm_id"] if meta.has_key("pm_id") else ""
                newdoc["authors"]=meta["authors"]
                newdoc["surnames"]=meta["surnames"]
                newdoc["year"]=meta["year"]
                newdoc["title"]=meta["title"]
                newdoc["norm_title"]=normalizeTitle(meta["title"])
                newdoc["numref"]=str(len(doc["references"]))
        ##        newdoc["error"]=meta.get("error","")
        ##        newdoc["notes"]=meta.get("notes","")
    ##            newdoc["references"]=doc["references"]
                newdoc["outlinks"]=[]
                newdoc["inlinks"]=[]
                newdoc["num_citations"]=len(doc["citations"])
                # this is for later processing and adding to database
                newdoc["num_in_collection_references"]=0
                newdoc["num_references"]=len(doc["references"])
                newdoc["num_matchable_citations"]=0
                newdoc["num_citations"]=0

                corpora.Corpus.addPaper(newdoc)

                count+=1
                if count % dot_every_xfiles ==0:
    ##                print count,"/",len(ALL_INPUT_FILES)
                    reportTimeLeft(count,len(ALL_INPUT_FILES), time0," -- importing")

        except:
            print "File: ",fn
            print "EXCEPTION: ",sys.exc_info()[:2]
            exception_file.write("File: "+fn+"\n")
            exception_file.write("Exception: "+sys.exc_info()[:2].__repr__()+"\n\n")

def ingestCorpus(inputdir, outputdir, metadata_index=None, files_to_ignore=[], file_mask="*.xml", options={}, start_at=0, maxfiles=10000000):
    """
        Does all that is necessary for the first ingestion of the (split and cleaned)
        ACL bob corpus

    """
    inputdir=ensureTrailingBackslash(inputdir)
    outputdir=ensureTrailingBackslash(outputdir)

    global_missing_references=[]

    print "Starting ingestion of corpus..."
    print "Creating database..."
    corpora.Corpus=corpora.CorpusClass(outputdir, initializing_corpus=True)

    corpora.Corpus.createFilesDB()
    corpora.Corpus.connectToDB()

    corpora.Corpus.metadata_index=metadata_index
    corpora.Corpus.FILES_TO_IGNORE=files_to_ignore

    global exception_file
    exception_file=open(corpora.Corpus.dir_fileDB_db+"exception_list.txt",mode="a")

    all_input_files_fn=corpora.Corpus.dir_fileDB_db+"all_input_files.txt"
    ALL_INPUT_FILES=loadFileList(all_input_files_fn)
    if not ALL_INPUT_FILES:
        print "Listing all files..."
        ALL_INPUT_FILES=listAllFiles(inputdir,file_mask)
        saveFileList(ALL_INPUT_FILES,all_input_files_fn)

    print "Converting input files to SciDoc format and loading metadata..."
    convertAllFilesAndAddToDB(ALL_INPUT_FILES[start_at:maxfiles], inputdir)

    ALL_FILES=corpora.Corpus.listAllPapers()[start_at:start_at+maxfiles]
    print "Finding matchable references, populating database..."
    time0=datetime.datetime.now()
    time1=datetime.datetime.now()
    dot_every_xfiles=100
    count=0
    for doc_id in ALL_FILES:
        time1=datetime.datetime.now()
        doc_meta=corpora.Corpus.getPaperByGUID(doc_id)
##        print "Processing in-collection references for ", doc_meta["filename"]

        doc_file=corpora.Corpus.loadSciDoc(doc_id)
        if not doc_file:
            print "Cannot load",doc_meta["filename"]
            continue

        matchable, in_collection_references, missing_references=corpora.Corpus.loadOrGenerateMatchableCitations(doc_file)

        doc_meta["num_in_collection_references"]=len(in_collection_references)
        doc_meta["num_references"]=len(doc_file["references"])
        doc_meta["num_matchable_citations"]=len(matchable)
        doc_meta["num_citations"]=len(doc_file["citations"])

        for ref in in_collection_references:
            match_meta=corpora.Corpus.getPaperByGUID(ref)
            if match_meta:
                if match_meta["guid"] not in doc_meta["outlinks"]:
                    doc_meta["outlinks"].append(match_meta["guid"])
                if doc_meta["guid"] not in match_meta["inlinks"]:
                    match_meta["inlinks"].append(doc_meta["guid"])
                    corpora.Corpus.updatePaper(match_meta)
            else:
                # do something if no match?
                pass

        corpora.Corpus.updatePaper(doc_meta)
##        if options.has_key("list_missing_references"):
##            makeNoteOfMissingReferences(missing_references)

##        global_missing_references.extend(missing_references)
        count+=1
        if count % dot_every_xfiles ==0:
##            print "This just took ",datetime.datetime.now()-time1, "s"
            reportTimeLeft(count,len(ALL_FILES), time0," -- latest paper "+doc_meta["filename"])
            time1=datetime.datetime.now()


##    corpora.Corpus.globalDBconn.close()
##    f=open(corpora.Corpus.fileDB_db_dir+"missing_references.json","w")
##    json.dump(global_missing_references,f)
##    f.close()

def main():
    pass

if __name__ == '__main__':
    main()
