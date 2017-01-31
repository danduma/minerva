# <description>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import minerva.db.corpora as cp

def connectToElastic():
    """
        Does the basics of connecting to the ES endpoint
    """
    cp.useElasticCorpus()
    from minerva.multi.config import MINERVA_ELASTICSEARCH_ENDPOINT
    cp.Corpus.connectCorpus(r"g:\nlp\phd\pmc_coresc", endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)

def testExplanation():
    """
    """
    from minerva.proc.query_extraction import WindowQueryExtractor
    from minerva.retrieval.elastic_retrieval import ElasticRetrieval

    ext=WindowQueryExtractor()
    er=ElasticRetrieval("idx_az_annotated_pmc_2013_1","",None)

    text="method method MATCH method method sdfakjesf"
    match_start=text.find("MATCH")

    queries=ext.extract({"parameters":[(5,5)],
                         "match_start": match_start,
                         "match_end": match_start+5,
                         "doctext": text,
                         "method_name": "test"
                        })

##    q=er.rewriteQueryAsDSL(queries[0]["structured_query"], {"metadata.title":1})
##    print(q)

    q={"dsl_query":{'multi_match': {'fields': ['Obj^1',
                            'Res^1',
                            'Goa^1',
                            'Mot^1',
                            'Hyp^1',
                            'Met^1',
                            'Bac^1',
                            'Exp^1',
                            'Con^1',
                            'Obs^1',
                            'Mod^1'],
                 'operator': 'or',
                 'query': u'strongly imaginative via associated coupled communication is repetitive influence mutations one stereotypies as show are point in accounting debate around novo developmental evidence dimensions for centres much separate linked delay genetic difficulties genetically over parental interests activities play loci risk on some genes contribute early independent mediated although asd regression heritable considerable susceptibility processes interaction de third language older whether many age children deficits prior range determined evident social usually narrow whole characterized effects ',
                 'type': 'best_fields'}}}

##    q={"dsl_query":
##        {"dis_max":}
##        {'match': {'content': ['Obj^1',
##                            'Res^1',
##                            'Goa^1',
##                            'Mot^1',
##                            'Hyp^1',
##                            'Met^1',
##                            'Bac^1',
##                            'Exp^1',
##                            'Con^1',
##                            'Obs^1',
##                            'Mod^1'],
##                 'operator': 'or',
##                 'query': u'strongly imaginative via associated coupled communication is repetitive influence mutations one stereotypies as show are point in accounting debate around novo developmental evidence dimensions for centres much separate linked delay genetic difficulties genetically over parental interests activities play loci risk on some genes contribute early independent mediated although asd regression heritable considerable susceptibility processes interaction de third language older whether many age children deficits prior range determined evident social usually narrow whole characterized effects ',
##                 'type': 'best_fields'}}}


    doc_ids=['559005ea-9288-4459-a8ef-8ae72ed1dc0f']

##    hits=er.es.search(index="papers", doc_type="paper", body={"query":q}, _source="guid", request_timeout=QUERY_TIMEOUT,)
##    doc_ids=[hit["_id"] for hit in hits["hits"]["hits"]]
##    print(doc_ids)
##    global ES_TYPE_DOC
##    ES_TYPE_DOC="paper"

    formula=er.formulaFromExplanation(q, doc_ids[0])
    print(formula.formula)
##    print(formula.computeScore(None, None, None))
    print(formula.computeScore(None, None, {"susceptibility":20}))

    from minerva.evaluation.best_keyword_selection import getFormulaTermWeights
    print(get({"match_guid":doc_ids[0],"formulas":[{"guid":doc_ids[0],"formula":formula.formula}]}))


def main():
    connectToElastic()
    testExplanation()
    pass

if __name__ == '__main__':
    main()
