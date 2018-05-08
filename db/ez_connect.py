# convenience functions to skip over connecting the corpus

from multi.celery_app import MINERVA_ELASTICSEARCH_ENDPOINT
import db.corpora as cp
from proc.general_utils import getRootDir


def ez_connect(corpus="AAC"):
    """
    Simplifies connecting to the Corpus

    :param corpus:
    :return: corpus instance
    """
    root_dir=""
    if corpus=="AAC":
        root_dir=getRootDir("aac")
    elif corpus=="PMC_CSC":
        root_dir = getRootDir("pmc_coresc")

    cp.useElasticCorpus()
    cp.Corpus.connectCorpus(root_dir, endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
    cp.Corpus.setCorpusFilter(corpus)
    return cp.Corpus