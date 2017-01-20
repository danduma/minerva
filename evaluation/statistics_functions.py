# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import sys

from minerva.proc.nlp_functions import CORESC_LIST, AZ_ZONES_LIST
import minerva.db.corpora as cp


def getAnnotationStatistics(doc):
    """
        Given a scidoc, computes

        :param doc: scidoc
        :type doc: scidoc
    """
    csc_counts={x:0 for x in CORESC_LIST}
    az_counts={x:0 for x in AZ_ZONES_LIST}
    citation_zone_counts={}
    zone_list=[]

    for sentence in doc.allsentences:
        if sentence.get("csc_type","") != "":
            csc_counts[sentence["csc_type"]] += 1
            zone_list.append(sentence["csc_type"])
        elif sentence.get("az","") != "":
            az_counts[sentence["az"]] += 1
            zone_list.append(sentence["az"])

    for citation in doc.citations:
        parent=citation.get("parent_s",None)
        if not parent:
            parent=citation.get("parent",None)
        if not parent:
            continue

        if sentence.get("csc_type","") != "":
            csc_counts[sentence["csc_type"]] += 1
            zone_list.append(sentence["csc_type"])
        elif sentence.get("az","") != "":
            az_counts[sentence["az"]] += 1
            zone_list.append(sentence["az"])

        s_type=doc.element_by_id[parent].get("csc_type", "")
        if s_type == "":
            s_type=doc.element_by_id[parent].get("az", "")
        if s_type != "":
            citation_zone_counts[s_type]=citation_zone_counts.get(s_type,0)+1

    return {
            "csc_type_counts":csc_counts,
            "az_counts":az_counts,
            "zone_list":zone_list,
            "num_sentences":len(doc.allsentences),
            "num_sections":len(doc.allsections),
            "num_paragraphs":len(doc.allparagraphs),
            "per_zone_citations":citation_zone_counts
            }

from minerva.importing.importing_functions import addSciDocToDB


def computeAnnotationStatistics(guid):
    """
        Store for each scidoc statistics on the PMC annotation

        :param guid: the uuid of the paper to compute statistics for
    """
##    try:
    doc=cp.Corpus.loadSciDoc(guid)
    stats=getAnnotationStatistics(doc)

    # TODO remove this when fixed: this is to fix having broken the corpus
    # !TODO ALSO CHANGE ElasticCorpus.addPaper : check_existing=True
    addSciDocToDB(doc,doc.metadata.get("import_id",""), doc.metadata.get("collection_id",""))
##    cp.Corpus.updatePaper(doc.metadata, op_type="index", has_scidoc=True)

##    meta=cp.Corpus.getMetadataByGUID(guid)
##    print meta
    # remove until here
    cp.Corpus.setStatistics(guid, stats)


##    except:
##        print("Error computing statistics for %s :", sys.exc_info()[:2])

def main():
    import json
##    cp.useElasticCorpus()
    from minerva.squad.config import MINERVA_ELASTICSEARCH_ENDPOINT
    cp.Corpus.connectCorpus("",endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
##    doc=cp.Corpus.loadSciDoc("957e1fcf-d5b4-41dc-af32-7db08f1d2ded")
##    print getAnnotationStatistics(doc)
    computeAnnotationStatistics("957e1fcf-d5b4-41dc-af32-7db08f1d2ded")
    print json.dumps(cp.Corpus.getStatistics("957e1fcf-d5b4-41dc-af32-7db08f1d2ded"),indent=3)
    pass

if __name__ == '__main__':
    main()
