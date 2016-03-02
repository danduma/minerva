# Corpus ingesting functions
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import json, sys, os, datetime, fnmatch, random
import logging

from minerva.proc.general_utils import (ensureTrailingBackslash, loadFileList,
saveFileList, ensureDirExists)
from minerva.proc.results_logging import ProgressIndicator

import minerva.db.corpora as cp

from importing_functions import (convertXMLAndAddToCorpus, updatePaperInCollectionReferences)
from minerva.squad.tasks import (t_convertXMLAndAddToCorpus, t_updatePaperInCollectionReferences)


exception_file=None

FILES_TO_PROCESS_FROM=0
FILES_TO_PROCESS_TO=sys.maxint


class CorpusImporter(object):
    """
        Allows the importing of a coherent corpus procedurally in one go
    """

    def __init__(self, collection_id="", import_id="", reader=None, use_celery=False):
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
        self.use_celery=use_celery

##    def convertDoc(self, filename, corpus_id):
##        """
##            Reads the input XML and saves a SciDoc
##        """
####        if cp.Corpus.getMetadataByField("metadata.corpus_id", corpus_id):
####            # Doc is already in collection
####            return
##
##        if not self.reader:
##            self.reader=AutoXMLReader()
##
##        doc=self.reader.readFile(filename)
##        doc.metadata["norm_title"]=normalizeTitle(doc.metadata["title"])
##        if doc.metadata.get("guid", "") == "":
##            doc.metadata["guid"]=cp.Corpus.generateGUID(doc.metadata)
##        assert corpus_id != ""
##        doc.metadata["corpus_id"]=corpus_id
##        cp.Corpus.saveSciDoc(doc)
##        return doc

    def convertAllFilesAndAddToDB(self, ALL_INPUT_FILES, inputdir):
        """
            Loads each XML file, saves it as a SciDoc JSON file, adds its metadata to
            the database
        """
        progress=ProgressIndicator(True, self.num_files_to_process, dot_every_xitems=20)

##        ALL_FILES_IN_DB=cp.Corpus.listPapers(field="filename") or []

        if self.use_celery:
            tasks=[]
            for fn in ALL_INPUT_FILES[FILES_TO_PROCESS_FROM:FILES_TO_PROCESS_TO]:
                corpus_id=self.generate_corpus_id(fn)
                match=cp.Corpus.getMetadataByField("metadata.filename",os.path.basename(fn))
                if not match:
                    tasks.append(t_convertXMLAndAddToCorpus.apply_async(args=[
                            os.path.join(inputdir,fn),
                            corpus_id,
                            self.import_id,
                            self.collection_id,
                            ],
                            queue="import_xml"
                            ))
        else:
            # main loop over all files
            for fn in ALL_INPUT_FILES[FILES_TO_PROCESS_FROM:FILES_TO_PROCESS_TO]:
                filename=cp.Corpus.paths.inputXML+fn
                corpus_id=self.generate_corpus_id(fn)

                match=cp.Corpus.getMetadataByField("metadata.filename",os.path.basename(fn))
                if not match:
                    try:
                        doc=convertXMLAndAddToCorpus(
                            os.path.join(inputdir,fn),
                            corpus_id,
                            self.import_id,
                            self.collection_id,
                            )
                    except ValueError:
                        logging.exception("ERROR: Couldn't convert %s" % fn)
                        continue

                    progress.showProgressReport("Importing -- latest file %s" % fn)



    def updateInCollectionReferences(self, ALL_GUIDS, import_options={}):
        """
            For every guid, it matches its in-collection references, and its
            resolvable citations

            Args:
                ALL_GUIDS: list of guids
        """
        print("Finding resolvable references, populating database...")
        progress=ProgressIndicator(True, len(ALL_GUIDS), dot_every_xitems=100)

        tasks=[]

        for doc_id in ALL_GUIDS[FILES_TO_PROCESS_FROM:FILES_TO_PROCESS_TO]:
            if self.use_celery:
                tasks.append(t_updatePaperInCollectionReferences.apply_async(
                    args=[doc_id, import_options],
                    kwargs={},
                    queue="update_references"
                    ))
            else:
                doc_meta=updatePaperInCollectionReferences(doc_id, import_options)
                filename=doc_meta["filename"] if doc_meta else "<ERROR>"
                progress.showProgressReport("Updating references -- latest paper "+filename)

    def listAllFiles(self, start_dir, file_mask):
        """
            Creates an ALL_FILES list with relative paths from the start_dir
        """
        ALL_FILES=[]

        for dirpath, dirnames, filenames in os.walk(start_dir):
            for filename in filenames:
                if fnmatch.fnmatch(filename,file_mask) and filename not in cp.Corpus.FILES_TO_IGNORE:
                        fn=os.path.join(dirpath,filename)
                        fn=fn.replace(start_dir,"")
                        ALL_FILES.append(fn)

        print("Total files:",len(ALL_FILES))
        return ALL_FILES

    def loadListOrListAllFiles(self, inputdir, file_mask):
        """
            Either loads the existing file list or lists the contents of the
            input directory.
        """
        all_input_files_fn=os.path.join(cp.Corpus.paths.fileDB,"all_input_files.txt")
        ALL_INPUT_FILES=loadFileList(all_input_files_fn)
        if not ALL_INPUT_FILES:
            print("Listing all files...")
            ALL_INPUT_FILES=self.listAllFiles(inputdir,file_mask)
            ensureDirExists(cp.Corpus.paths.fileDB)
            saveFileList(ALL_INPUT_FILES,all_input_files_fn)

        return ALL_INPUT_FILES

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


    def restartCollectionImport(self, import_options):
        """
            This is to be used during testing, when a bug is found during
            an import or it is interrupted and it has to be restarted. It
            essentially flushes all the added papers, scidocs and cache during
            the import.
        """
        if import_options.get("convert_and_import_docs",True) and FILES_TO_PROCESS_FROM == 0:
            imported_guids=cp.Corpus.listPapers("metadata.collection_id:\"%s\"" % self.collection_id)
            resolvable_bags=["resolvable_"+id for id in imported_guids]
##            print("Imported papers:",imported_guids)
            cp.Corpus.bulkDelete(imported_guids, "scidocs")
            cp.Corpus.bulkDelete(imported_guids, "papers")
            cp.Corpus.bulkDelete(resolvable_bags, "cache")

    def importCorpus(self, root_input_dir, file_mask="*.xml", import_options={}, maxfiles=10000000000):
        """
            Does all that is necessary for the initial import of the corpus
        """
        inputdir=ensureTrailingBackslash(root_input_dir)

        print("Starting ingestion of corpus...")
        print("Creating database...")

        cp.Corpus.createAndInitializeDatabase()
        cp.Corpus.connectToDB()

        cp.Corpus.metadata_index=self.metadata_index
        cp.Corpus.FILES_TO_IGNORE=self.files_to_ignore


        self.start_time=datetime.datetime.now()
        ALL_INPUT_FILES=self.loadListOrListAllFiles(inputdir,file_mask)

        self.num_files_to_process=min(len(ALL_INPUT_FILES),FILES_TO_PROCESS_TO-FILES_TO_PROCESS_FROM)
        print("Converting input files to SciDoc format and loading metadata...")
        if import_options.get("convert_and_import_docs",True):
            self.convertAllFilesAndAddToDB(ALL_INPUT_FILES, inputdir)

        print("Updating in-collection links...")
        ALL_GUIDS=cp.Corpus.listPapers("metadata.collection_id:\"%s\"" % self.collection_id)
        self.updateInCollectionReferences(ALL_GUIDS, import_options)
        self.end_time=datetime.datetime.now()
        print("All done. Processed %d files. Took %s" % (self.num_files_to_process, str(self.end_time-self.start_time)))

    def reloadSciDocsOnly(self, conditions, inputdir, file_mask):
        """
            This reloads the SciDocs for a subset of all files in the corpus
        """
##        filenames=cp.Corpus.SQLQuery("SELECT guid,metadata.filename FROM papers where %s limit 10000" % conditions)
        filenames=cp.Corpus.unlimitedQuery(
            index="papers",
            doc_type="paper",
            _source=["metadata.filename","guid"],
            q=conditions
            )

        ALL_INPUT_FILES=self.loadListOrListAllFiles(inputdir,file_mask)
        files_to_process=[]
        files_hash={}
        for input_file in ALL_INPUT_FILES:
            corpus_id=self.generate_corpus_id(input_file)
            files_hash[corpus_id]=input_file

        for file_name in filenames:
            corpus_id=self.generate_corpus_id(file_name["_source"]["metadata"]["filename"])
            assert(corpus_id in files_hash)
            files_to_process.append([files_hash[corpus_id],file_name["_source"]["guid"]])

        tasks=[]
        progress=ProgressIndicator(True,len(files_to_process))

        for fn in files_to_process:
            corpus_id=self.generate_corpus_id(fn[0])
            if self.use_celery:
                tasks.append(t_convertXMLAndAddToCorpus.apply_async(args=[
                        os.path.join(cp.Corpus.paths.inputXML,fn[0]),
                        corpus_id,
                        self.import_id,
                        self.collection_id,
                        ],
                        kwargs={"existing_guid":fn[1]},
                        queue="import_xml"
                        ))
            else:
                try:
                    doc=convertXMLAndAddToCorpus(
                        os.path.join(cp.Corpus.paths.inputXML,fn[0]),
                        corpus_id,
                        self.import_id,
                        self.collection_id,
                        existing_guid=fn[1]
                        )
                except ValueError:
                    logging.exception("ERROR: Couldn't convert %s" % fn)
                    continue

                progress.showProgressReport("Importing -- latest file %s" % fn)

def simpleTest():
    """
    """
    cp.Corpus.globalDBconn.close()
    f=open(os.path.join(cp.Corpus.paths.fileDB_db,"missing_references.json"),"w")
    json.dump(global_missing_references,f)
    f.close()


def main():

    pass

if __name__ == '__main__':
    main()
