# compute graph-based similarity between papers

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import json
import os
import scipy
import scipy.sparse
from tqdm import tqdm
from collections import defaultdict
import pickle

DTYPE = "float64"


# test_docs = ['p1 p2 p3',
#              'p3 p4',
#              'p5 p1',
#              'p3 p4',
#              'p8 p2',
#              'p10 p1',
#              'p8 p4',
#              'p8 p0',
#              'p5',
#              'p1',
#              'p1',
#              'p3',
#              'p4',
#              ]
#
# test_guids = []
# for test_doc in test_docs:
#     line_guids = test_doc.split()
#     test_guids.extend(line_guids)
# test_guids = list(set(test_guids))


class DocSimilarity(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.force_regenerate_resolvable_citations = False
        self.dice = defaultdict(lambda: 0.0)
        self.use_numpy = False

        self.guid_dict_filename = os.path.join(self.corpus.ROOT_DIR,
                                               "guid_map.json")
        self.cocitation_matrix_filename = os.path.join(self.corpus.ROOT_DIR,
                                                       "co_citation.matrix")
        self.dice_matrix_filename = os.path.join(self.corpus.ROOT_DIR,
                                                 "dice.matrix")
        self.missing_files_filename = os.path.join(self.corpus.ROOT_DIR,
                                                   "missing_files.txt")

    def saveGuidDict(self):
        self.all_guids = self.corpus.listPapers()

        count_model = CountVectorizer(vocabulary=self.all_guids,
                                      ngram_range=(1, 1))
        count_model.fit(self.all_guids)
        json.dump(count_model.vocabulary_, open(self.guid_dict_filename, "w"))
        self.guid_to_int = count_model.vocabulary_
        self.int_to_guid = {v: k for k, v in self.guid_to_int.items()}

    def loadGuidDict(self):
        self.guid_to_int = json.load(open(self.guid_dict_filename, "r"))
        self.int_to_guid = {v: k for k, v in self.guid_to_int.items()}
        self.all_guids = [self.int_to_guid[i] for i in range(len(self.guid_to_int))]

    def computeAndExportDiceMatrix(self):
        """
        Goes through the co-occurrence matrix, computing the distance scores

        :return: same matrix
        """
        diagonal = self.mx.diagonal()
        indices = self.mx.nonzero()

        dice = scipy.sparse.lil_matrix((len(self.guid_to_int), len(self.guid_to_int)), dtype=DTYPE)

        for index in tqdm(range(len(indices[0]))):
            y = indices[0][index]
            x = indices[1][index]

            separate = diagonal[y] + diagonal[x]
            if separate:
                together = self.mx[y, x]
                value = together / separate
                dice[y, x] = value
                # dice[x, y] = value
                # if value != 0.5:
                #     print(
                #         # self.all_guids[x], self.all_guids[y],
                #         int(together),
                #         int(separate), "=",
                #         int(diagonal[y]), "+", int(diagonal[x]),
                #         "%0.3f" % value)
            else:
                # mx[y, x] = 0
                pass

        self.mx = dice
        self.mx = self.mx.tolil()
        self.saveDiceMatrix()
        return self.mx

    def computeAndExportDiceDict(self):
        """
        Goes through the co-occurrence matrix, computing the distance scores

        :return: same matrix
        """
        self.dice = defaultdict(lambda: 0.0)

        for pair in tqdm(self.oc):
            y = pair[0]
            x = pair[1]

            separate = self.counts[y] + self.counts[x]
            if separate:
                together = self.oc[pair]
                value = together / separate
                self.dice[(x, y)] = value
                # dice[x, y] = value
                # if value != 0.5:
                #     print(
                #         # self.all_guids[x], self.all_guids[y],
                #         int(together),
                #         int(separate), "=",
                #         int(diagonal[y]), "+", int(diagonal[x]),
                #         "%0.3f" % value)
            else:
                # mx[y, x] = 0
                pass

        self.saveDiceMatrix()
        return self.dice

    def generateCocitationMatrix(self):
        """
        Fills and exports the co-occurrence matrix in resolvable citations

        :return:
        """
        counts = defaultdict(lambda: 0)

        if self.use_numpy:
            mx = scipy.sparse.lil_matrix((len(self.guid_to_int), len(self.guid_to_int)), dtype=DTYPE)
        else:
            oc = defaultdict(lambda: 0)

        missing_guids = []
        for guid in tqdm(self.all_guids):
            doc = self.corpus.loadSciDoc(guid)
            resolvable = self.corpus.loadOrGenerateResolvableCitations(
                doc, force_recompute=self.force_regenerate_resolvable_citations)
            resolvable = resolvable["resolvable"]

            same_para = {}

            for citation in resolvable:
                sent_id = citation["cit"]["parent_s"]

                para_id = None
                if sent_id in doc.element_by_id:
                    para_id = doc.element_by_id[sent_id]["parent"]
                else:
                    sent_num = int(sent_id[1:])
                    while sent_num > 0:
                        new_sent_id = "s" + str(sent_num)
                        if new_sent_id in doc.element_by_id:
                            sent_id = new_sent_id
                            para_id = doc.element_by_id[sent_id]["parent"]
                            break
                        sent_num -= 1

                if not para_id:
                    continue

                same_para[para_id] = {}

                if not "match_guids" in citation:
                    citation["match_guids"] = [citation["match_guid"]]

                for match_guid in citation["match_guids"]:
                    same_para[para_id][match_guid] = same_para[para_id].get(match_guid, 0) + 1

            for para_id in same_para:
                for match_guid in same_para[para_id]:
                    other_matches = set(same_para[para_id].keys()) - set(match_guid)
                    if match_guid not in self.guid_to_int:
                        # print("MISSING", match_guid)
                        missing_guids.append(match_guid)
                        continue

                    guid_id = self.guid_to_int[match_guid]

                    if self.use_numpy:
                        mx[guid_id, guid_id] += 1

                    counts[guid_id] += 1

                    for other_match in other_matches:
                        if other_match not in self.guid_to_int:
                            # print("MISSING", other_match)
                            missing_guids.append(other_match)
                            continue
                        other_id = self.guid_to_int[other_match]
                        if self.use_numpy:
                            mx[guid_id, other_id] += 1
                        else:
                            oc[(guid_id, other_id)] += 1

        if self.use_numpy:
            self.mx = mx
        else:
            self.oc = oc
            self.counts = counts
        self.saveCocitationMatrix(missing_guids)
        return oc, counts

    def buildSpeedIndex(self):
        self.speed_index = defaultdict(lambda: list())
        for pair in self.dice:
            self.speed_index[pair[0]].append(pair[1])

    def saveCocitationMatrix(self, missing_guids):
        if self.use_numpy:
            scipy.sparse.save_npz(self.cocitation_matrix_filename, self.mx.tocsc())
        else:
            pickle.dump({"oc": dict(self.oc), "counts": dict(self.counts)},
                        open(self.cocitation_matrix_filename, "wb"))

        with open(self.missing_files_filename, "w") as f:
            for missing in missing_guids:
                f.write(missing)
                f.write("\n")

    def loadCocitationMatrix(self):
        if self.use_numpy:
            self.mx = scipy.sparse.load_npz(self.cocitation_matrix_filename)
            self.mx = self.mx.tolil()
        else:
            obj = pickle.load(open(self.cocitation_matrix_filename, "rb"))
            self.oc = defaultdict(lambda: 0, obj["oc"])
            self.counts = defaultdict(lambda: 0, obj["counts"])

    def loadDiceMatrix(self):
        if self.use_numpy:
            self.mx = scipy.sparse.load_npz(self.dice_matrix_filename)
            self.mx = self.mx.tolil()
        else:
            dice = pickle.load(open(self.dice_matrix_filename, "rb"))
            self.dice = defaultdict(lambda: 0.0, dice)
            self.buildSpeedIndex()

    def saveDiceMatrix(self):
        if self.use_numpy:
            scipy.sparse.save_npz(self.dice_matrix_filename, self.mx.tocsc())
        else:
            dice = dict(self.dice)
            pickle.dump(dice, open(self.dice_matrix_filename, "wb"))

    def getDiceDistance(self, guid1, guid2):
        assert guid1 in self.guid_to_int
        assert guid2 in self.guid_to_int

        int1 = self.guid_to_int[guid1]
        int2 = self.guid_to_int[guid2]

        if self.use_numpy:
            return self.mx[int1, int2]
        else:
            return self.dice[(int1, int2)]

    def generateOrLoadCorpusMatrix(self, force_recompute=False):
        """
        This method calls the generator functions or loads the existing matrices

        :param force_recompute: if True, rebuild everything even if it exists
        :return:
        """
        if force_recompute or not os.path.exists(self.guid_dict_filename):
            self.saveGuidDict()
        else:
            self.loadGuidDict()

        if force_recompute or not os.path.exists(self.dice_matrix_filename):
            if not os.path.exists(self.cocitation_matrix_filename):
                self.generateCocitationMatrix()
            else:
                self.loadCocitationMatrix()

            if self.use_numpy:
                self.computeAndExportDiceMatrix()
            else:
                self.computeAndExportDiceDict()
        else:
            # self.mx = scipy.sparse.load_npz(self.dice_matrix_filename)
            self.loadDiceMatrix()

    def getNonZero(self):
        if self.use_numpy:
            nzx, nzy = self.mx.nonzero()
            return zip(nzx, nzy)
        else:
            return self.dice.items()

    def getMatrixEntry(self, key):
        if self.use_numpy:
            return self.mx[key]
        else:
            return self.dice[key]

    def getRelatedPapersWithSimilarity(self, guid, threshold):
        int1 = self.guid_to_int[guid]
        res = []
        slice = self.mx[int1, :]
        for index, guid in enumerate(self.all_guids):
            if index == int1:
                continue
            if slice[index] >= threshold:
                res.append((guid, slice[index]))

        return res

    def getRelatedPapers(self, guid, threshold):
        return [p[0] for p in self.getRelatedPapersWithSimilarity(guid, threshold)]

    def expandRelevantPapers(self, guids, threshold, max_papers):
        """
        Takes a list of match_guids and finds a set of papers that are relevant to
        one or more of the papers, u

        :param guids: list of match_guids already annotated
        :param max_papers: maximum of papers to return
        :return:
        """
        if max_papers == 0:
            return []

        potentials = {}

        for guid in guids:
            new_related = self.getRelatedPapersWithSimilarity(guid, threshold)
            for paper in new_related:
                potentials[paper[0]] = potentials.get(paper[0], 0) + paper[1]

        for guid in guids:
            if guid in potentials:
                print(guid, "is already in list of relevant papers")
                del potentials[guid]

        sorted_pot = sorted(potentials.items(),
                            key=lambda x: x[1],
                            reverse=True)

        return sorted_pot[:max_papers]
