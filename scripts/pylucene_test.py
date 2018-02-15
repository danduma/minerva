from __future__ import absolute_import
from __future__ import print_function
import sys
import lucene
import json

from lucene import (SimpleFSDirectory, System, File,
    Document, Field, StandardAnalyzer, IndexWriter, IndexSearcher, QueryParser)

if __name__ == "__main__":
    lucene.initVM()
    fullIndexDir = r"c:\NLP\PhD\bob\fileDB\LuceneFullIndex"

    print("lucene version is:", lucene.VERSION)

    fullIndex = SimpleFSDirectory(File(fullIndexDir))
    analyzer = StandardAnalyzer(lucene.Version.LUCENE_CURRENT)
    writer = IndexWriter(fullIndex, analyzer, True, IndexWriter.MaxFieldLength(20000000))
##    writer = IndexWriter(store, analyzer, True, IndexWriter.MaxFieldLength(512))

    print("Currently there are %d documents in the index..." % writer.numDocs())

##    print  "Reading lines from sys.stdin..."
    lines=["bla bla bla bla bla","Erase una vez que se era", "En un lugar de La Mancha de cuyo nombre no quiero acordarme, no ha mucho que vivia un hidalgo de los de lanza en ristre", "Manchame mancha mancha que te mancha la mancha"]

    for l in lines:
        doc = Document()
        doc.add(Field("text", l, Field.Store.YES, Field.Index.ANALYZED))
        metadata={"asdfa":"asdfa"}
        json_metadata=json.dumps(metadata)
        doc.add(Field("metadata", json_metadata, Field.Store.YES, Field.Index.NO))
        writer.addDocument(doc)

    print("Indexed lines from stdin (%d documents in index)" % (writer.numDocs()))
    print("About to optimize index of %d documents..." % writer.numDocs())
    writer.optimize()
    print("...done optimizing index of %d documents" % writer.numDocs())
    print("Closing index of %d documents..." % writer.numDocs())
    print("...done closing index of %d documents" % writer.numDocs())
    writer.close()

    # RETRIEVAL

    dir = SimpleFSDirectory(File(fullIndexDir))
    analyzer = StandardAnalyzer(lucene.Version.LUCENE_CURRENT)
    searcher = IndexSearcher(dir)

    query = QueryParser(lucene.Version.LUCENE_CURRENT, "text", analyzer).parse(u"¿Dónde está La Mancha?")
    MAX = 1000
    hits = searcher.search(query, MAX)

    print("Found %d document(s) that matched query '%s':" % (hits.totalHits, query))

    for hit in hits.scoreDocs:
        print(hit.score, hit.doc, hit.toString())
        doc = searcher.doc(hit.doc)
        print(doc.get("text").encode("utf-8"))
        print(doc.get("metadata").encode("utf-8"))
