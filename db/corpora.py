# Corpus class. Deals with a database of scientific papers.
#
# Copyright:   (c) Daniel Duma 2014
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

#TODO: Split this file into server/client, use proper external DB

import os, sys, json, glob, cPickle, sqlite3, codecs, uuid, fnmatch
from minerva.util.general_utils import *
import minerva.scidoc as scidoc

class CorpusClass:
    """
        Class that deals with corpus access
    """
    def __init__(self):
        """
            Basic creator
        """
        self.ALL_FILES=[]
        self.TEST_FILES=[]
        self.FILES_TO_IGNORE=[]
        self.metadata_index=None

        self.setPaths("")

        # annotate files with AZ, CSC, etc.
        self.annotators={}

    def setPaths(self, root_dir):
        """
        """
        self.ROOT_DIR=root_dir
        self.paths=AttributeDict()
        self.paths.fileDB=self.ROOT_DIR+"fileDB"+os.sep
        self.paths.fileDB_db=self.paths.fileDB+"db"+os.sep
        self.paths.inputXML=self.ROOT_DIR+"inputXML"+os.sep
        self.paths.pickledDocs=self.ROOT_DIR+"fileDB\\pickled"+os.sep
        self.paths.jsonDocs=self.ROOT_DIR+"fileDB\\jsonDocs"+os.sep
        self.paths.prebuiltBOWs=self.ROOT_DIR+"fileDB\\BOWs"+os.sep
        self.paths.fileLuceneIndex=self.ROOT_DIR+"fileDB\\LuceneIndeces"+os.sep
        self.paths.fullLuceneIndex=self.ROOT_DIR+"fileDB\\LuceneFullIndex"+os.sep
        self.paths.output=self.ROOT_DIR+"output"+os.sep
        self.paths.experiments=self.ROOT_DIR+"experiments"+os.sep
        self.indexDB_path=self.paths.fileDB+"db\\index.db"


    def connectCorpus(self, base_directory, initializing_corpus=False,suppress_error=False):
        """
            If DB has been created, connect to it. If not, initialize it first.

            Args:
                base_directory: root dir of this corpus
                initializing_corpus: if True, create DB and directories
                suppress_error: if true, db doesn't complain if it's connected already
        """
        self.setPaths(ensureTrailingBackslash(base_directory))

        if initializing_corpus:
            self.createFilesDB()
        self.connectToDB(suppress_error)

    def loadAnnotators(self,annotators=["AZ","CFC"]):
        """
            Loads annotators (classifiers), either as parameter or by default
        """
        for annotator in annotators:
            if annotator=="AZ":
                from az_cfc_classification import AZannotator
                self.annotators[annotator]=AZannotator("trained_az_classifier.pickle")
            if annotator=="CFC":
                from az_cfc_classification import CFCannotator
                self.annotators[annotator]=CFCannotator("trained_cfc_classifier.pickle")

    def annotateDoc(self,doc,annotate_with=None):
        """
            Shortcut to get Coprus object to annotate file with one or more annotators
        """
        if not annotate_with:
            annotate_with=self.annotators.keys()

        for annotator in annotate_with:
            assert(self.annotators.has_key(annotator))
            self.annotators[annotator].annotateDoc(doc)

    def makeAllFilesList(self):
        """
            Method to override by descendant classes
        """
        self.ALL_FILES=[]


    def listAllFiles(self, start_dir,file_mask):
        """
            Creates an ALL_FILES list with relative paths from the start_dir, saves it
        """
        ALL_FILES=[]

        for dirpath, dirnames, filenames in os.walk(start_dir):
            for filename in filenames:
                if fnmatch.fnmatch(filename,file_mask) and filename not in self.FILES_TO_IGNORE:
                        fn=os.path.join(dirpath,filename)
                        fn=fn.replace(start_dir,"")
                        ALL_FILES.append(fn)

        saveFileList(ALL_FILES,self.paths.fileDB+"db\\ALL_INPUT_FILES.txt")

        print "Total files:",len(ALL_FILES)
        return ALL_FILES


    def selectRandomInputFiles(self, howmany, file_mask="*.xml"):
        """
            Of all input files, it picks number of random ones.

            Tries to open ALL_INPUT_FILES.txt in filedb/db directory. If not
            present, it makes sure all directories exist, calls listAllFiles()
            to generate the file.

        """
        try:
            all_files_file=file(os.path.join(self.paths.fileDB_db,"ALL_INPUT_FILES.txt"), "r")
            all_files=all_files_file.readlines()
        except:
            self.createDefaultDirs()
            all_files=self.listAllFiles(self.paths.inputXML, file_mask)

        result=[]

        for cnt in range(howmany):
            result.append(all_files[random.randint(0,len(all_files))].strip("\n"))
        return result


    def loadPickleIfAvailable(self,filename):
        if os.path.exists(filename):
            return loadPickle(filename)
        else:
            return None

    def prebuiltFileName(self, fileUID, method, parameter):
        """
            Returns filename for prebuilt BOW in JSON format
        """
        return self.getFileUID(fileUID)+"_"+method+str(parameter)+".json"

    def prebuiltFilePath(self, fileUID, method, parameter):
        """
            Returns filename for prebuilt BOW in JSON format
        """
        return self.paths.prebuiltBOWs+self.getFileUID(fileUID)+os.sep+self.prebuiltFileName(fileUID, method, parameter)

    def getLuceneIndexPath(self, guid, index_filename, full_corpus=False):
        """
            Returns the path to the Lucene index for a test file in
            the corpus

            if full_corpus is True, this is the general index for that method
                else
            when using Citation Resolution (resolving only from the
            references at the bottom of a paper) it is the specific index for
            that file guid
        """
        guid=guid.lower()
        if full_corpus:
            return self.paths.fullLuceneIndex+index_filename
        else:
            return self.paths.fileLuceneIndex+guid+os.sep+index_filename

    def getFileUID(self, filename):
        """
            Given a filename, returns the unique identifier of that file in the collection
        """
        fileName, fileExtension = os.path.splitext(filename)
        return os.path.basename(fileName).lower()

    def generateGUID(self, metadata):
        """
            Generates a UUID version 5. It is based on the authors' surnames and
            title of the document

            >>> print CorpusClass().generateGUID({"surnames":["Jones","Smith"], "year": 2005, "norm_title": "a simple title"})
            296da00b-5fb9-5800-8b5f-f50ef951afc4
        """
        if metadata.get("norm_title","") == "" or metadata.get("surnames",[]) == []:
            raise ValueError("No norm_title or no surnames!")

        uuid_string = " ".join(metadata["surnames"]).lower().strip()
        uuid_string += metadata["norm_title"]
        uuid_string = uuid_string.encode("ascii")
        hex_string=str(uuid.uuid5(uuid.NAMESPACE_DNS,uuid_string))
        return hex_string

    def prebuiltFileExists(self, fileUID, method, parameter):
        return os.path.isfile(self.prebuiltFilePath(fileUID,method,parameter))

    def pickledFileExists(self, filename):
        """
            True if pickled file is found
        """
        return os.path.isfile(self.paths.pickledDocs+os.sep+filename)

    def jsonFileExists(self, filename):
        """
            True if JSON file is found
        """
        return os.path.isfile(self.paths.prebuiltBOWs+os.sep+filename)

    def experimentExists(self, filename):
        """
            True if pickled file is found
        """
        return os.path.isfile(self.paths.experiments+filename+".json")

    def selectBOWParametersToPrebuild(self, filename, method, parameters):
        """
            Returns a list of parameters to build BOWs for, which may be empty,
            by removing all those that already have a prebuiltBOW

            Delete the files if you want them prebuilt again
        """
        newparam=[param for param in parameters if not self.prebuiltFileExists(filename, method, param)]
        return newparam

    def savePrebuiltBOW(self,filename, method, parameter, bow):
        """
            Provides a nifty interface to cache the BOWs for incoming link contexts
        """
        path=self.prebuiltFilePath(filename,method,parameter)
        self.savePrecomputedDataJson(path,bow)

    def loadPrebuiltBOW(self,filename, method, parameter):
        """
            Provides a nifty interface to cache the BOWs for incoming link contexts
        """
        path=self.prebuiltFilePath(filename,method,parameter)
        return self.loadPrecomputedDataJson(path)

    def matchableCitationsFileName(self,guid):
        guid=guid.lower()
        return self.paths.prebuiltBOWs+guid+"_resolvable.json"

    def saveResolvableCitations(self,guid,tin_can):
        guid=guid.lower()
        self.savePrecomputedDataJson(self.matchableCitationsFileName(guid),tin_can)

    def loadResolvableCitations(self,guid):
        guid=guid.lower()
        return self.loadPrecomputedDataJson(self.matchableCitationsFileName(guid))

    def getResolvableCitationsCache(self, guid, doc):
        """
            Loads if available, generates otherwise
        """
        guid=guid.lower()
        if os.path.exists(self.matchableCitationsFileName(guid)):
            tin_can=self.loadResolvableCitations(guid) # load the citations in the document that are resolvable
            if tin_can:
                return tin_can
        else:
            try:
                data=doc.selectResolvableCitations()
                tin_can={"resolvable":data[0],"outlinks":data[1]}
                self.saveResolvableCitations(guid,tin_can)
                return tin_can
            except:
                print "Exception in getResolvableCitationsCache() while saving data"
                return None

    def savePrecomputedDataJson(self,path, bow):
        """
            Save anything as JSON
        """
        ensureDirExists(os.path.dirname(path))
        lines=json.dumps(bow,indent=3)
        try:
            f=codecs.open(path, "w","utf-8")
            f.write(lines)
            f.close()
        except:
            print "Error saving JSON", path, "Exception in savePrecomputedDataJson():",sys.exc_info()[:2]

    def loadPrecomputedDataJson(self,path):
        """
            Load precomputed JSON
        """
        try:
            f=open(path, "rb")
##            print "Loading saved JSON ",path
            res=json.load(f)
            f.close()
            return res
        except:
            print "Exception in loadPrecomputedDataJson():", sys.exc_info()[:2], path
            return None


    def savePickledXML(self, filename, xml):
        """
        """
        name=self.getFileUID(filename)+"_pickled.pic"
        try:
            f=open(self.paths.pickledDocs+name, "wb")
            cPickle.dump(xml,f,0)
            f.close()
        except:
            print "ERROR PICKLING XML",filename,"Exception:", sys.exc_info()[:2]

    def loadPickledXML(self,filename):
        """
            Load a previously pickled SciXML
        """

        name=self.getFileUID(filename)+"_pickled.pic"
        try:
            f=open(self.paths.pickledDocs+name, "rb")
            doc=cPickle.load(f)
            f.close()
        except:
            print "Exception in loadPickledXML():", sys.exc_info()[:2]
            doc=None

        return doc

    def loadSciDoc(self,guid):
        """
            If a SciDocJSON file exists for guid, it returns it, otherwise None
        """
        guid=guid.lower()
        filename=os.path.join(self.paths.jsonDocs,guid+".json")
        if os.path.exists(filename):
            return scidoc.SciDoc(filename)
        else:
            return None

    def loadExperiment(self,exp_name):
        """
            Return a dict with the details for running an experiment
        """
        paths.name=self.paths.experiments+exp_name
        filename=paths.name+".json"
        assert(os.path.isfile(filename))
        exp=json.load(open(filename,"r"))
        exp["exp_dir"]=paths.name+os.sep
        return exp

    def saveSciDoc(self,doc):
        """
            Saves the document as JSON using the SciDoc's saveToFile method,
            name is generated from its GUID + .json
        """
        filename=os.path.join(self.paths.jsonDocs,doc["metadata"]["guid"]+".json")
        doc.saveToFile(filename)

    def connectToDB(self, suppress_error=False):
    	if os.path.exists(self.indexDB_path):
    		self.globalDBconn=sqlite3.connect(self.indexDB_path)
        else:
            if not suppress_error:
                print "ERROR: Couldn't connect to DB",self.indexDB_path
            self.globalDBconn=None

    def createDefaultDirs(self):
        """
            Creates all necessary dirs
        """
        ensureDirExists(self.paths.inputXML)
        ensureDirExists(self.paths.fileDB)
        ensureDirExists(self.paths.fileDB_db)
        ensureDirExists(self.paths.fileLuceneIndex)
        ensureDirExists(self.paths.fullLuceneIndex)
        ensureDirExists(self.paths.jsonDocs)
        ensureDirExists(self.paths.pickledDocs)
        ensureDirExists(self.paths.prebuiltBOWs)

    def createFilesDB(self):
        """
            Ensures that the directory structure is in place and creates
            the SQLite database and tables
        """
        self.createDefaultDirs()
    	if not os.path.exists(self.indexDB_path):
    		conn=sqlite3.connect(self.indexDB_path)
    		c=conn.cursor()
            #        id int primary key,
    		c.execute("""create table papers (
            GUID text primary key,
            DOI text,
            corpus_id text,
            filename text,
            norm_title text,
            surnames text,
            year text,
            num_in_collection_references int,
            num_references int,
            num_matchable_citations int,
            num_citations int,
            num_inlinks int,
            import_id text,
            collection_id text,
            metadata text)""")
    		conn.commit()

    		c.execute("""create table authors (
            GUID text primary key,
            full_name,
            surname text,
            metadata text)""")
    		conn.commit()

    		c.execute("""create table links (
            GUID_from text,
            GUID_to text,
            authors_from text,
            authors_to text,
            year_from text,
            year_to text,
            numcitations int)""")
    		conn.commit()

    		c.execute("""create table missing_papers (
            norm_title text,
            reference_counts int,
            metadata text)""")
    		conn.commit()

    		c.close()
    		conn.close()

    def getMetadataByFilename(self,filename):
        return getMetadataByField("filename",filename)

    def getMetadataByGUID(self,guid):
        guid=guid.lower()
        return self.getMetadataByField("guid",guid)

    def getMetadataByField(self,field,value):
        if self.globalDBconn:
            c=self.globalDBconn.cursor()
            c.execute("select metadata from papers where "+field+" = ?",(value,))
            rows=c.fetchall()
            c.close()
            if len(rows) > 0:
                try:
                    return json.loads(rows[0][0])
                except ValueError:
                    c=self.globalDBconn.cursor()
                    c.execute("DELETE FROM papers WHERE "+field+" = ?",(value,))
                    c.close()
                    self.globalDBconn.commit()
                    return None
            else:
                return None

    def updatePaper(self,metadata):
        if self.globalDBconn:
            c=self.globalDBconn.cursor()
            c.execute("""update papers set
            metadata=?,
            num_in_collection_references=?,
            num_matchable_citations=?,
            num_citations=?,
            num_references=?,
            num_inlinks=?
            where guid = ?""",
            (json.dumps(metadata),
            metadata["num_in_collection_references"],
            metadata["num_matchable_citations"],
            metadata["num_citations"],
            metadata["num_references"],
            unicode(len(metadata["inlinks"])),
            unicode(metadata["guid"])))
            self.globalDBconn.commit()
            c.close()

    def listPapers(self,conditions):
        """
            Return a list of GUIDs in papers table where [conditions]
        """
        return self.runSingleValueQuery("select guid from papers where "+conditions)


    def listAllPapers(self):
        """
            Return a list of all GUIDs of all papers in papers table
        """
        return self.runSingleValueQuery("select guid from papers")

    def listAllFilenames(self):
        """
            Return a list of all filenames of all papers in papers table
        """
        return self.runSingleValueQuery("select filename from papers")

    def runSingleValueQuery(self,query):
        if self.globalDBconn:
            c=self.globalDBconn.cursor()
            c.execute(query)
            rows=c.fetchall()
            c.close()
            if len(rows) > 0:
                return [row[0] for row in rows]
            else:
                return None
        else:
            return None

    def addAuthor(self, author):
        """
            Make sure author is in database
        """
##        if not author.get("guid",None):
##            author["guid"]=self.generateGUID()
        # TODO implement this
        pass

    def addPaper(self, metadata, check_existing=True):
        if self.globalDBconn:
            c=self.globalDBconn.cursor()

            if check_existing:
                doc_meta=self.getMetadataByGUID(metadata["guid"])
                if doc_meta:
                    self.updatePaper(metadata)
                    return

            for author in metadata["authors"]:
                self.addAuthor(author)

            c.execute(u"""insert into papers (
            GUID,
            DOI,
            corpus_id,
            filename,
            norm_title,
            surnames,
            year,
            num_in_collection_references,
            num_references,
            num_matchable_citations,
            num_citations,
            num_inlinks,
            import_id,
            collection_id,
            metadata
            )
            values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (unicode(metadata["guid"]),
            unicode(metadata["doi"]),
            unicode(metadata["corpus_id"]),
            unicode(metadata["filename"]),
##            unicode(normalizeTitle(metadata["title"])),
            unicode(metadata["norm_title"]),
            unicode(metadata["surnames"]),
            unicode(metadata["year"]),
            metadata["num_in_collection_references"],
            metadata["num_references"],
            metadata["num_matchable_citations"],
            metadata["num_citations"],
            unicode(len(metadata["inlinks"])),
            "", # import_id
            "", # collection_id
            json.dumps(metadata)))
            self.globalDBconn.commit()
            c.close()


    def addLink(self,GUID_from,GUID_to,authors_from,authors_to,year_from,year_to,numcitations):
        if self.globalDBconn:
            c=self.globalDBconn.cursor()
            c.execute(u"""insert into links (
            GUID_from,
            GUID_to,
            authors_from,
            authors_to,
            year_from,
            year_to,
            numcitations
            )
            values (?,?,?,?,?,?,?,?,?,?)""",
            (metadata["guid"],
            metadata["filename"],
            normalizeTitle(metadata["title"]),
            metadata["surnames"],
            metadata["year"],
            metadata["num_in_collection_references"],
            metadata["num_references"],
            metadata["num_matchable_citations"],
            metadata["num_citations"],
            metadata))
            self.globalDBconn.commit()
            c.close()

    def addMissingPaper(self, metadata):
        if self.globalDBconn:
            c=self.globalDBconn.cursor()
            metadata["norm_title"]=normalizeTitle(metadata["title"])
            c.execute("select reference_counts from missing_papers where norm_title = ?",(metadata["norm_title"],))
            rows=c.fetchall()
            if len(rows) > 0:
                c.execute("update missing_papers set reference_counts=reference_counts+1 where norm_title=?",(metadata["norm_title"],))
                self.globalDBconn.commit()
                c.close()
                return

            c.execute("""insert into missing_papers (
            norm_title,
            reference_counts,
            metadata)
            values (?,?,?)""",
            (unicode(normalizeTitle(metadata["title"])),
            0,
            unicode(json.dumps(metadata))))
            self.globalDBconn.commit()
            c.close()

    def updateAllPapersMetadataOutlinks(self):
        """
            Goes through the papers table, updating the outlinks in the metadata
            field with the actual in-collection references.

            Returns a list of guids of documents that have experienced changes,
            so that indeces can be rebuilt accordingly if neccesary
        """

        res=[]
        for paper_guid in self.listAllPapers():
            metadata=self.getMetadataByGUID(paper_guid)
            old_outlinks=metadata["outlinks"]
            doc=self.loadSciDoc(metadata["guid"])
            new_outlinks=self.listDocInCollectionReferences(doc)

            if set(new_outlinks) != set(old_outlinks):
                res.append(paper_guid)
            metadata["outlinks"]=new_outlinks
            self.updatePaper(metadata)

        return res

    def listDocInCollectionReferences(self,doc):
        """
            Returns a list of guids of all the in-collection references of a doc
        """
        res=[]
        for ref in doc["references"]:
            match=self.matchReferenceInIndex(ref)
            if match:
                res.append(match["guid"])
        return res

    def tagAllReferencesAsInCollectionOrNot(self, doc):
        """
            For each reference in a doc, it adds a boolean "in_collection"
        """
        for ref in doc["references"]:
            match=self.matchReferenceInIndex(ref)
            ref["in_collection"]=True if match else False

    def matchReferenceInIndex(self, ref):
        """
            Returns the matching document in the hashed index based on the title and surnames of authors
        """
        if self.globalDBconn is not None:

            # try matching by ID first
            for id_type in [("doi","doi"),("pmid","corpus_id")]:
                if ref.has_key(id_type[0]) and ref[id_type[0]] != "":
                    doc_meta=self.getMetadataByField(id_type[1],ref[id_type[0]])
                    if doc_meta:
                        return doc_meta

            # if it can't be matched by id
            hash=normalizeTitle(ref["title"])

            c=self.globalDBconn.cursor()
            if not isinstance(hash, unicode):
                hash=unicode(hash, errors="ignore")
            c.execute(u"select metadata from papers where norm_title=?",(hash,))
            rows=c.fetchall()
            c.close()
##            res=None
            for row in rows:
                doc_meta=json.loads(row[0]) # load metadata dict
                if len(doc_meta["surnames"]) > 0:
                    for a1 in doc_meta["surnames"]:
                        for a2 in ref["surnames"]:
                            if a1 and a2 and a1.lower() == a2.lower():
                                # essentially, if ANY surname matches
        ##                        print "SUCCESS matching:",hash
                                return doc_meta
            return None

        else:
            raise Exception

    def luceneFileIndexPath(self,fileUID,method,parameter):
        """
            For every citation resolution context, that is, for every file,
            one index for every method and parameter to compare
        """
        return self.paths.fileLuceneIndex+getFileUID(fileUID)+os.sep+method+os.sep+str(parameter)

    def loadOrGenerateResolvableCitations(self, doc_file):
        """
            Tries to open pre-pickled matchable citations and outlinks,
            generates and saves them if not found
        """
        missing_references=[]
        if not self.jsonFileExists(self.matchableCitationsFileName(doc_file["metadata"]["guid"])):
            matchable, outlinks, missing_references=doc_file.selectResolvableCitations()
            tin_can={"matchable":matchable,"outlinks":outlinks}
            self.saveResolvableCitations(doc_file["metadata"]["guid"],tin_can)
        else:
            tin_can=self.loadResolvableCitations(doc_file["metadata"]["guid"])
            if not tin_can:
                matchable, outlinks, missing_references=doc_file.selectResolvableCitations()
            else:
                matchable=tin_can["matchable"]
                outlinks=tin_can["outlinks"]

        return [matchable,outlinks,missing_references]

    def listIncollectionReferencesOfList(self, guid_list):
        """
            Will return all in-collection references of all the files in the list
        """
        res=[]
        for guid in guid_list:
            res.extend(self.getMetadataByGUID(guid)["outlinks"])
        res=list(set(res))
        return res

    def selectDocResolvableCitations(self, doc):
        """
            Returns a list of {"cit","match"} with the citations in the text and
            their matching document in the index

            WARNING! Also removes inline citations that aren't resolvable
        """
        res=[]
        unique={}
        sents_with_multicitations=[]
        missing_references=[]

    ##    print "total citations:", len(doc["citations"])
        for ref in doc["references"]:
            match=self.matchReferenceInIndex(ref)
            if match:
                ref["resolved_link"]=match["guid"]
                for cit_id in ref["citations"]:
                    res.append({"cit":doc.citation_by_id[cit_id],"match_guid":match["guid"]})
                    unique[match["guid"]]=unique.get(match["guid"],0)+1
            else:
                missing_references.append(ref)
                #remove citation from sentence
                for cit_id in ref["citations"]:
                    cit=doc.citation_by_id[cit_id]
                    sent=doc.element_by_id[cit["parent_s"]]
                    sent["text"]=re.sub(r"<cit\sid=.?"+str(cit["id"])+".{0,3}/>","",sent["text"], 0,re.IGNORECASE)
                    sents_with_multicitations.append(sent)

        for sent in sents_with_multicitations:
            # deal with many citations within characters of each other: make them know they are a cluster
            doc.countMultiCitations(sent)

        return [res, unique, missing_references]


##    def selectResolvableCitations_old(self):
##        """
##            Returns a list of {"cit","match"} with the citations in the text and
##            their matching document in the index
##
##            WARNING! Also removes inline citations that aren't resolvable
##        """
##        res=[]
##        unique={}
##        sents_with_multicitations=[]
##        missing_references=[]
##
##    ##    print "total citations:", len(self["citations"])
##        for cit in self["citations"]:
##            if cit["ref_id"] != 0:
##                ref=self.matchReferenceById(cit["ref_id"])
##                if ref:
##                    match=Corpus.matchReferenceInIndex(ref)
##                    if match:
##                        ref["resolved_link"]=match["guid"]
##                        res.append({"cit":cit,"match_guid":match["guid"]})
##                        unique[match["guid"]]=unique.get(match["guid"],0)+1
##                    else:
##                        missing_references.append(ref)
##                        #remove citation from sentence
##                        sent=self.element_by_id[cit["parent_s"]]
##                        sent["text"]=re.sub(r"<cit\sid=.?"+str(cit["id"])+".{0,3}/>","",sent["text"], 0,re.IGNORECASE)
##                        sents_with_multicitations.append(sent)
##                else:
##                    print "no match by id!", cit["id"],"->",cit["ref_id"]
##            else:
##                print "cit link == 0! ", cit["id"]
##
##        for sent in sents_with_multicitations:
##            self.countMultiCitations(sent) # deal with many citations within characters of each other: make them know they are a cluster TODO cluster them
##
##        return [res, unique, missing_references]


    def createDBindeces(self):
        """
            Call this after importing the metadata into the corpus and before
            matching in-collection references, it will speed up search massively
        """

        c=self.globalDBconn.cursor()
        c.execute(u"CREATE UNIQUE INDEX `papers_guid` ON `papers` (`GUID` ASC);")
        c.execute(u"CREATE  INDEX `papers_doi` ON `papers` (`DOI` ASC);")
        c.execute(u"CREATE  INDEX `papers_corpus_id` ON `papers` (`corpus_id` ASC);")
        c.execute(u"CREATE  INDEX `papers_norm_title` ON `papers` (`norm_title` ASC);")
        c.execute(u"CREATE  INDEX `papers_num_in_collection_references` ON `papers` (`num_in_collection_references` DESC);")
        c.execute(u"CREATE  INDEX `papers_matchable_citations` ON `papers` (`num_matchable_citations` DESC);")
        c.execute(u"CREATE  INDEX `papers_inlinks` ON `papers` (`num_inlinks` DESC);")
        c.close()
        self.globalDBconn.commit()

Corpus=CorpusClass()

DOCTEST = True

if __name__ == '__main__':

    if DOCTEST:
        import doctest
        doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)