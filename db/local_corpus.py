# LocalCorpus: uses SQLite for storage of paper metadata, .json files on disk
# for SciDocs and all cached BOWs
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function

import os, sys, json, glob, sqlite3, codecs, uuid, unicodedata
from minerva.proc.general_utils import AttributeDict, ensureTrailingBackslash, ensureDirExists
from minerva.scidoc.scidoc import SciDoc

from base_corpus import BaseCorpus

class LocalCorpus(BaseCorpus):
    """
        Class that deals with corpus access
    """
    def __init__(self):
        """
            Basic creator
        """
        super(self.__class__, self).__init__()
        self.ALL_FILES=[]
        self.TEST_FILES=[]
        self.FILES_TO_IGNORE=[]
        self.metadata_index=None

        self.setPaths("")
        self.globalDBconn=None

        # annotate files with AZ, CSC, etc.
        self.annotators={}
        # pretty-print saved SciDoc files?
        self.saveDocIndent=None

    def setPaths(self, root_dir):
        """
        """
        self.ROOT_DIR=root_dir
        self.paths=AttributeDict()
        self.paths.fileDB=self.ROOT_DIR+"fileDB"+os.sep
        self.paths.fileDB_db=self.paths.fileDB+"db"+os.sep
        self.paths.inputXML=self.ROOT_DIR+"inputXML"+os.sep
        self.paths.jsonDocs=self.paths.fileDB+"jsonDocs"+os.sep
        self.paths.cachedJson=self.paths.fileDB+"BOWs"+os.sep
        self.paths.fileLuceneIndex=self.paths.fileDB+"LuceneIndeces"+os.sep
        self.paths.fullLuceneIndex=self.paths.fileDB+"LuceneFullIndex"+os.sep
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
            self.createAndInitializeDatabase()
        self.connectToDB(suppress_error)

    def connectedToDB(self):
        """
            returns True if connected to DB, False otherwise
        """
        return (self.globalDBconn is not None)

    def getRetrievalIndexPath(self, guid, index_filename, full_corpus=False):
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
            return os.path.join(self.paths.fullLuceneIndex,index_filename)
        else:
            return os.path.join(self.paths.fileLuceneIndex+guid,index_filename)

    def connectToDB(self, suppress_error=False):
        """
            Connects to SQLite DB
        """
    	if os.path.exists(self.indexDB_path):
    		self.globalDBconn=sqlite3.connect(self.indexDB_path)
        else:
            if not suppress_error:
                print("ERROR: Couldn't connect to DB",self.indexDB_path)
            self.globalDBconn=None

    def createAndInitializeDatabase(self):
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
            num_resolvable_citations int,
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

    def createDefaultDirs(self):
        """
            Creates all necessary dirs
        """
        for path in self.paths:
            ensureDirExists(self.paths[path])

    def getMetadataByGUID(self,guid):
        """
        """
        guid=guid.lower()
        return self.getMetadataByField("guid",guid)

    def getMetadataByField(self,field,value):
        """
            Gets a single paper's metadata given a field and its value
        """
        assert self.connectedToDB(), "Not connected to DB!"

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

    def listFieldByField(self,field1,field2,value):
        """
            Returns a list: for each paper, field1 if field2==value
        """
        assert self.connectedToDB(), "Not connected to DB!"

        c=self.globalDBconn.cursor()
        c.execute(u"select ? from papers where ?=?",(field1, field2, value,))
        rows=c.fetchall()
        c.close()

    def updatePaper(self,metadata):
        """
            Updates an existing record in the db
        """
        assert(self.globalDBconn)

        c=self.globalDBconn.cursor()
        c.execute("""update papers set
        metadata=?,
        num_in_collection_references=?,
        num_resolvable_citations=?,
        num_citations=?,
        num_references=?,
        num_inlinks=?
        where guid = ?""",
        (json.dumps(metadata),
        metadata["num_in_collection_references"],
        metadata["num_resolvable_citations"],
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
        query=self.filterQuery(conditions)
        return self.runSingleValueQuery("select guid from papers where "+query)

    def listAllPapers(self):
        """
            Return a list of all GUIDs of all papers in papers table
        """
        return self.runSingleValueQuery("select guid from papers")


    def cachedDataIDString(self, type, guid, params=None):
        """
            Returns a string that can be used as either a path or resource ID
        """
        res=super(self.__class__, self).cachedDataIDString(type, guid, params)
        res+=".json"
        if type=="resolvable_citations":
            return os.path.join(self.paths.cachedJson,res)
        if type=="prebuilt_bow":
            return os.path.join(self.paths.cachedJson+guid,res)
        return

    def cachedJsonExists(self, type, guid, params=None):
        """
        """
        return os.path.isfile(self.cachedDataIDString(type, guid, params))

    def saveCachedJson(self, path, data):
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
            print("Error saving JSON", path, "Exception in saveCachedJson():",sys.exc_info()[:2])

    def loadCachedJson(self,path):
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
            print("Exception in loadCachedJson():", sys.exc_info()[:2], path)
            return None

    def loadSciDoc(self,guid):
        """
            If a SciDocJSON file exists for guid, it returns it, otherwise None
        """
        guid=guid.lower()
        filename=os.path.join(self.paths.jsonDocs,guid+".json")
        if os.path.exists(filename):
            return SciDoc(filename)
        else:
            return None

    def saveSciDoc(self,doc):
        """
            Saves the document as JSON using the SciDoc's saveToFile method,
            name is generated from its GUID + .json
        """
        filename=os.path.join(self.paths.jsonDocs,doc["metadata"]["guid"]+".json")
        doc.saveToFile(filename, indent=self.saveDocIndent)

    def runSingleValueQuery(self,query):
        assert self.globalDBconn is not None, "Not connected to DB!"

        c=self.globalDBconn.cursor()
        c.execute(query)
        rows=c.fetchall()
        c.close()
        if len(rows) > 0:
            return [row[0] for row in rows]
        else:
            return None

    def addAuthor(self, author):
        """
            Make sure author is in database
        """
##        if not author.get("guid",None):
##            author["guid"]=self.generateGUID()
        # TODO implement this
        raise NotImplementedError
        pass

    def addPaper(self, metadata, check_existing=True, ):
        """
            Adds a paper's metadata to the database
        """
        assert self.globalDBconn is not None, "Not connected to DB!"

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
        num_resolvable_citations,
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
        metadata["num_resolvable_citations"],
        metadata["num_citations"],
        unicode(len(metadata["inlinks"])),
        metadata["import_id"], # import_id
        metadata["collection_id"], # collection_id
        json.dumps(metadata)))
        self.globalDBconn.commit()
        c.close()

    def addLink(self,GUID_from,GUID_to,authors_from,authors_to,year_from,year_to,numcitations):
        assert self.globalDBconn is not None, "Not connected to DB!"

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
        metadata["num_resolvable_citations"],
        metadata["num_citations"],
        metadata))
        self.globalDBconn.commit()
        c.close()

    def addMissingPaper(self, metadata):
        """
            Inserts known data about an unkown paper
        """
        assert self.globalDBconn is not None, "Not connected to DB!"

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
        c.execute(u"CREATE  INDEX `papers_matchable_citations` ON `papers` (`num_resolvable_citations` DESC);")
        c.execute(u"CREATE  INDEX `papers_inlinks` ON `papers` (`num_inlinks` DESC);")
        c.close()
        self.globalDBconn.commit()

    def setCorpusFilter(self, collection_id=None, import_id=None, date=None):
        """
            Sets the filter query to limit all queries to a collection (corpus)
            or an import date

            :param collection: identifier of corpus, e.g. "ACL" or "PMC". This is set at import time.
            :type collection:basestring
            :param import: identifier of import, e.g. "initial"
            :type import:basestring
            :param date: comparison with a date, e.g. ">[date]", "<[date]",
            :type collection:basestring
        """
        query_items=[]
        if collection_id:
            query_items.append("collection_id=%s" % collection_id)
        if import_id:
            query_items.append("import=%s" % import_id)
        if date:
            query_items.append("time_created%s" % date)

        self.query_filter=" AND ".join(query_items)+" AND "


DOCTEST = True

if __name__ == '__main__':

    if DOCTEST:
        import doctest
        doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
