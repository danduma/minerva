from db.ez_connect import ez_connect
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def test_dict():
    corpus = ez_connect("AAC", "koko")
    # corpus = ez_connect("PMC_CSC", "koko")
    corpus.doc_sim.loadGuidDict()
    mx, oc, counts = corpus.doc_sim.generateCocitationMatrix()
    dice = []

    for pair in tqdm(oc):
        y = pair[0]
        x = pair[1]

        separate = counts[y] + counts[x]
        if separate:
            together = oc[pair]
            value = together / separate
            dice.append(value)
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

    #     if dist != 0.5:
    #         print(g1, g2, dist)

    pd.DataFrame(dice).hist()
    plt.show()


def test():
    # corpus = ez_connect("AAC", "koko")
    corpus = ez_connect("PMC_CSC", "koko")
    corpus.doc_sim.use_numpy = False
    corpus.doc_sim.generateOrLoadCorpusMatrix()

    all_dices = []

    for pair in corpus.doc_sim.getNonZero():
        # g1 = corpus.doc_sim.all_guids[pair[0]]
        # g2 = corpus.doc_sim.all_guids[pair[1]]
        # dist = corpus.doc_sim.getDiceDistance(g1, g2)
        dist = corpus.doc_sim.getMatrixEntry(pair)
        all_dices.append(dist)
    #     if dist != 0.5:
    #         print(g1, g2, dist)

    pd.DataFrame(all_dices).hist()
    plt.show()

if __name__ == '__main__':
    test()
