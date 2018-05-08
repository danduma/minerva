from multi.config import MINERVA_ELASTICSEARCH_ENDPOINT
from evaluation.experiment import Experiment
from proc.general_utils import getRootDir
from proc.nlp_functions import getFirstNumberFromString

import db.corpora as cp

from tqdm import tqdm
import json

num_removed_sent = 0
num_papers_removed_sent = 0


def mergeSentences(id1, id2, doc):
    """
    Sentence with id2 becomes a part of sentence with id1
    :param id1:
    :param id2:
    :return:
    """
    global num_removed_sent

    s1 = doc.element_by_id[id1]
    s2 = doc.element_by_id[id2]

    if len(s2["text"]) > len(s1["text"]):
        s1["az"]=s2.get("az","")
    s1["text"] = s1["text"] + " " + s2["text"]
    if "citations" not in s1:
        s1["citations"] = []
    if "citations" not in s2:
        s2["citations"] = []

    s1["citations"].extend(s2["citations"])
    s1["pos_tagged"] = s1.get("pos_tagged","") + " " + s2.get("pos_tagged", "")
    for cit_id in s2["citations"]:
        doc.citation_by_id[cit_id]["parent_s"] = id1

    doc.element_by_id[s2["parent"]]["content"].remove(id2)
    doc.data["content"].remove(s2)
    doc.updateContentLists()
    # print("removed sentence ", id2)
    # if len(s1["citations"]) > 0:
    num_removed_sent += 1
    #     print(s1["citations"]," ", s2["citations"])
    return doc


def remove_extra_sentence_stuff(sent):
    if "parse" in sent:
        del sent["parse"]
    return sent


def fix_sentence_splitting_in_docs(guids):
    global num_papers_removed_sent

    for guid in tqdm(guids):
        doc = cp.Corpus.loadSciDoc(guid)
        # print(guid, "-", doc["metadata"]["title"])

        to_merge = []
        join_previous = False
        previous_id = None
        for sentence in doc.allsentences:
            sentence = remove_extra_sentence_stuff(sentence)

            if join_previous:
                to_merge.append((previous_id, sentence["id"]))

            join_previous = False

            if sentence["text"].endswith("et al."):
                join_previous = True
                previous_id = sentence["id"]

        to_merge.reverse()

        for pair in to_merge:
            # print(doc.element_by_id[pair[0]]["text"], " + ", doc.element_by_id[pair[1]]["text"])
            # print(doc.element_by_id[pair[0]])
            # print(doc.element_by_id[pair[1]])
            doc = mergeSentences(pair[0], pair[1], doc)

        cp.Corpus.saveSciDoc(doc)

        if len(to_merge) > 0:
            num_papers_removed_sent += 1

        # if len(to_merge) > 2:
        #     print(guid)
        #     print(json.dumps(doc.data, indent=3))
        #     assert False

def fix_stranded_citations_in_docs(guids):
    for guid in tqdm(guids):
        doc = cp.Corpus.loadSciDoc(guid)
        for cit in doc.citations:
            if not cit["parent_s"] in doc.element_by_id:
                num=getFirstNumberFromString(cit["parent_s"])
                for cnt in range(num,0,-1):
                    new_sent_id= "s"+str(cnt)
                    if new_sent_id in doc.element_by_id:
                        cit["parent_s"]=new_sent_id
                        break
        cp.Corpus.saveSciDoc(doc)


def main():
    cp.useElasticCorpus()
    cp.Corpus.connectCorpus(getRootDir("aac"), endpoint=MINERVA_ELASTICSEARCH_ENDPOINT)
    cp.Corpus.setCorpusFilter("AAC")

    # fix_sentence_splitting_in_docs(cp.Corpus.listPapers())
    fix_stranded_citations_in_docs(cp.Corpus.listPapers())

    global num_removed_sent, num_papers_removed_sent
    print("Removed {} sentences from {} papers".format(num_removed_sent, num_papers_removed_sent))


if __name__ == '__main__':
    main()
