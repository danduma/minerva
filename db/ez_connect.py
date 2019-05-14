# convenience functions to skip over connecting the corpus

# from multi.celery_app import set_config
import multi.celery_app as celery_app
import db.corpora as cp
from proc.general_utils import getRootDir


def ez_connect(corpus="AAC", es_config=None):
    """
    Simplifies connecting to the Corpus

    :param corpus:
    :return: corpus instance
    """
    # global MINERVA_ELASTICSEARCH_ENDPOINT
    root_dir = ""
    if corpus == "AAC":
        root_dir = getRootDir("aac")
    elif corpus == "PMC_CSC":
        root_dir = getRootDir("pmc_coresc")
    elif corpus is None:
        root_dir = ""
    else:
        raise ValueError("Unknown corpus")

    cp.useElasticCorpus()

    if es_config:
        celery_app.MINERVA_ELASTICSEARCH_ENDPOINT = celery_app.set_config(es_config)

    cp.Corpus.connectCorpus(root_dir, endpoint=celery_app.MINERVA_ELASTICSEARCH_ENDPOINT)

    if corpus:
        cp.Corpus.setCorpusFilter(corpus)
    return cp.Corpus
