# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT
import json, os

import minerva.db.corpora as cp
from minerva.scidoc.xmlformats.read_paperxml import PaperXMLReader
from minerva.scidoc.xmlformats.write_scixml import SciXMLWriter
from minerva.scidoc.xmlformats.read_scixml import SciXMLWriter


from corpus_import import CorpusImporter
import corpus_import

from aac_import import AANReferenceMatcher, getACL_corpus_id, import_aac_corpus
from proc.general_utils import writeFileText
from scidoc.render_content import SciDocRenderer


def basicTest():
    """
    """
    importer=CorpusImporter(reader=PaperXMLReader())
    importer.collection_id="AAC_AZ"
    importer.import_id="initial"
    importer.generate_corpus_id=getACL_corpus_id

##    cp.useLocalCorpus()
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\aac")

def generateSideBySide():
    """
    """
    from minerva.squad.celery_app import MINERVA_ELASTICSEARCH_ENDPOINT
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus("g:\\nlp\\phd\\aac", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)

    from minerva.scidoc.xmlformats.read_auto import AutoXMLReader

    reader=AutoXMLReader()
    output_dir=os.path.join(cp.Corpus.ROOT_DIR,"conversion_visualization\\")

    file_list=[]
    for filename in doc_list:
        print("Converting %s" % filename)
        input_file=cp.Corpus.paths.inputXML+filename
        output_file=output_dir+"%s_1.html" % os.path.basename(filename)

        doc=reader.readFile(input_file)
        try:
            json.dumps(doc.data)
        except:
            print("Not JSON Serializable!!!!")

        html=SciDocRenderer(doc).prettyPrintDocumentHTML(True,True,True, True)
        writeFileText(html,output_file)

        WriteSciXML

        output_file2=output_file.replace("_1.html","_2.html")
        writeFileText(html,output_file2)
        file_list.append([os.path.basename(output_file),os.path.basename(output_file2)])

    file_list_json="file_data=%s;" % json.dumps(file_list)
    writeFileText(file_list_json,output_dir+"file_data.json")


def main():
    pass

if __name__ == '__main__':
    main()
