# Corpus ingesting functions
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import sys, os, datetime, fnmatch, random
import logging

from minerva.proc.general_utils import (ensureTrailingBackslash, loadFileList,
    saveFileList, ensureDirExists)
from minerva.proc.results_logging import ProgressIndicator

import minerva.db.corpora as cp

from importing_functions import (convertXMLAndAddToCorpus, updatePaperInCollectionReferences)
from minerva.squad.tasks import (importXMLTask, updateReferencesTask)

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

    def convertAllFilesAndAddToDB(self, ALL_INPUT_FILES, inputdir, import_options):
        """
            Loads each XML file, saves it as a SciDoc JSON file, adds its metadata to
            the database
        """
        progress=ProgressIndicator(True, self.num_files_to_process, dot_every_xitems=20)
        tasks=[]

        for fn in ALL_INPUT_FILES[FILES_TO_PROCESS_FROM:FILES_TO_PROCESS_TO]:
            corpus_id=self.generate_corpus_id(fn)
            match=cp.Corpus.getMetadataByField("metadata.filename",os.path.basename(fn))
            if not match or import_options.get("reload_xml_if_doc_in_collection",False):
                if self.use_celery:
                        match_id=match["guid"] if match else None
                        tasks.append(importXMLTask.apply_async(
                          args=[
                                os.path.join(inputdir,fn),
                                corpus_id,
                                self.import_id,
                                self.collection_id,
                                import_options,
                                match_id
                                ],
                                queue="import_xml"
                                ))
                else:
                    # main loop over all files
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
                                import_options
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
                tasks.append(updateReferencesTask.apply_async(
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
        if import_options.get("convert_and_import_docs",True):
            print("Converting input files to SciDoc format and loading metadata...")
            self.convertAllFilesAndAddToDB(ALL_INPUT_FILES, inputdir, import_options)

        if import_options.get("update_doc_references",True):
            print("Updating in-collection links...")

            ALL_GUIDS=cp.Corpus.SQLQuery("SELECT guid FROM papers where metadata.original_citation_style = \"AFI\" or metadata.original_citation_style = null or metadata.pmc_id <> null limit 20000000")
##            ALL_GUIDS=cp.Corpus.listPapers("metadata.collection_id:\"%s\"" % self.collection_id)
##            assert False
            self.updateInCollectionReferences(ALL_GUIDS, import_options)

        self.end_time=datetime.datetime.now()
        print("All done. Processed %d files. Took %s" % (self.num_files_to_process, str(self.end_time-self.start_time)))

    def reloadSciDocsOnly(self, conditions, inputdir, file_mask):
        """
            Iterates through the papers already in the collection given the
            condition. Tries to load their scidoc. If KeyError occurs, it loads
            the XML again
        """
##        filenames=cp.Corpus.SQLQuery("SELECT guid,metadata.filename FROM papers where %s limit 10000" % conditions)
        in_collection=[item["_source"] for item in cp.Corpus.unlimitedQuery(
            index="papers",
            doc_type="paper",
            _source=["metadata.corpus_id","metadata.filename","guid"],
            q=conditions
            )]

        print("Fixing broken SciDocs")
        print("Listing all loaded papers...")
        ALL_INPUT_FILES=self.loadListOrListAllFiles(inputdir,file_mask)
        files_to_process=[]
        files_hash={}
        for input_file in ALL_INPUT_FILES:
            corpus_id=self.generate_corpus_id(input_file)
            files_hash[corpus_id]=input_file

        print("Iterating over all papers trying to load them...")
        tasks=[]
        import_options={"reload_xml_if_doc_in_collection": True,}
        progress=ProgressIndicator(True,len(in_collection))
        for item in in_collection:
            corpus_id=self.generate_corpus_id(item["metadata"]["filename"])
            assert corpus_id==item["metadata"]["corpus_id"]
            try:
                doc=cp.Corpus.loadSciDoc(item["guid"])
            except KeyError:
                print("File %s is broken" % item["guid"])
                if self.use_celery:
                    tasks.append(importXMLTask.apply_async(args=[
                            os.path.join(cp.Corpus.paths.inputXML,files_hash[corpus_id]),
                            corpus_id,
                            self.import_id,
                            self.collection_id,
                            import_options
                            ],
                            kwargs={"existing_guid":item["guid"]},
                            queue="import_xml"
                            ))
                else:
                    files_to_process.append([files_hash[corpus_id],item["guid"]])

            progress.showProgressReport("Checking papers")

        if self.use_celery:
            return

        print("Processing all %s broken files..." % len(files_to_process))
        progress=ProgressIndicator(True,len(files_to_process))

        for fn in files_to_process:
            corpus_id=self.generate_corpus_id(fn[0])
            try:
                doc=convertXMLAndAddToCorpus(
                    os.path.join(cp.Corpus.paths.inputXML,fn[0]),
                    corpus_id,
                    self.import_id,
                    self.collection_id,
                    import_options,
                    existing_guid=fn[1],
                    )
            except ValueError:
                logging.exception("ERROR: Couldn't convert %s" % fn)
                continue

            progress.showProgressReport("Importing -- latest file %s" % fn)

# To inspect queues
##from celery.task.control import inspect
##i = inspect('scrapper_start')
##i.active()  #  get a list of active tasks
##i.registered() # get a list of tasks registered
##i.scheduled # get a list of tasks waiting
##i.reserved() #tasks that has been received, but waiting to be executed

def main():

    pass

if __name__ == '__main__':
    main()
