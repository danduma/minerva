import numpy as np

np.warnings.filterwarnings('ignore')

from evaluation.keyword_annotation_measurement import runAndMeasureOneQuery
from retrieval.elastic_retrieval import ElasticRetrievalBoost

from db.ez_connect import ez_connect
from collections import Counter
from copy import deepcopy

import os
import math

import random
from models.keyword_features import FeaturesReader, tokenWeight, getRootDir
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def expcurve(x, max):
    x = x / max
    mid = max / 2
    return pow(x - mid, 2)


def makeQueryFromContext(context):
    counts = Counter([t["text"].lower() for t in context["tokens"]])

    query = []
    for kw in context["best_kws"]:
        # (text, count, boost, bool, field, distance)
        new_token = (kw[0], counts[kw[0]], tokenWeight(kw), None, None, None)
        # new_token = (text, counts[text], float(weight)/counts[text], None, None, None)
        query.append(new_token)

    return query


def measureScore(corpus, queries):
    for query in queries:
        runAndMeasureOneQuery(query, {"_all_text": 1})


def baselineW(kws, mod=None):
    pass


def multiplyW(kws, mul):
    for kw in kws:
        kws[kw] *= mul


def addW(kws, add):
    for kw in kws:
        kws[kw] += add


def randomW(kws, param=None):
    for kw in kws:
        kws[kw] = random.random()


def addRandomW(kws, param=None):
    for kw in kws:
        kws[kw] += random.random()


def allSetW(kws, set):
    for kw in kws:
        kws[kw] = set


def takeMaxW(kws, param=1):
    sorted_list = sorted(kws.items(), key=lambda x: tokenWeight(x), reverse=True)

    for kw in sorted_list[:param]:
        kws[kw[0]] *= 1.1


def takeMinW(kws, param=1):
    sorted_list = sorted(kws.items(), key=lambda x: tokenWeight(x), reverse=False)

    for kw in sorted_list[:param]:
        kws[kw[0]] *= 10


def takeMinMaxDivW(kws, param={}):
    sorted_list = sorted(kws.items(), key=lambda x: tokenWeight(x), reverse=param.get("reverse", False))

    for kw in sorted_list[:param["num"]]:
        kws[kw[0]] *= param["mul"]


def tweakW(kws, param={}):
    max_list = sorted(kws.items(), key=lambda x: tokenWeight(x), reverse=True)
    min_list = list(reversed(max_list))

    for kw in max_list[:param["max_num"]]:
        kws[kw[0]] *= param["max_mul"]

    for kw in min_list[:param["min_num"]]:
        kws[kw[0]] *= param["min_mul"]


def invertW(kws, param=1):
    for kw in kws:
        kws[kw] = param / kws[kw]


def adjustDistW(kws, param=1):
    sorted_list = sorted(kws.items(), key=lambda x: tokenWeight(x), reverse=False)
    max_val = max(kws.values())
    mean_val = max_val / len(kws)

    for kw in sorted_list:
        if kws[kw[0]] < mean_val * param:
            kws[kw[0]] *= 10
        else:
            break


def normW(kws, param=1):
    max_val = max(kws.values())

    for kw in kws:
        kws[kw] /= max_val


def sigmoidW(kws, param=1):
    max_val = max(kws.values())

    for kw in kws:
        kws[kw] /= max_val
        kws[kw] = sigmoid(kws[kw])


def powerW(kws, power=1):
    max_val = max(kws.values())

    for kw in kws:
        kws[kw] = math.pow(kws[kw], power)


def attenuateSigmoidW(kws, param={}):
    for kw in kws:
        kws[kw] /= math.pow(sigmoid(kws[kw]), param.get("pow", 2))

    max_val = max(kws.values())
    for kw in kws:
        kws[kw] /= max_val


def attenuateExpW(kws, param={}):
    max_val = max(kws.values())
    for kw in kws:
        # kws[kw] /= math.pow(expcurve(kws[kw], max_val), param.get("pow", 2))
        att = expcurve(kws[kw], max_val)
        # kws[kw] *= 1 - att
        kws[kw] *= 1 + pow(att, param.get("pow",2))
        # print("%s %0.8f *= 1 - %0.8f" % (kw, kws[kw], att))

    max_val = max(kws.values())
    for kw in kws:
        kws[kw] /= max_val

def attenuateTanhW(kws, param={}):
    max_val = max(kws.values())
    for kw in kws:
        # kws[kw] /= math.pow(expcurve(kws[kw], max_val), param.get("pow", 2))
        att = np.tanh([kws[kw]])
        # kws[kw] *= 1 - att
        kws[kw] *= att
        # print("%s %0.8f *= 1 - %0.8f" % (kw, kws[kw], att))

    max_val = max(kws.values())
    for kw in kws:
        kws[kw] /= max_val




def runMod(query, mod):
    if len(query["best_kws"]) == 0:
        return None
    kws = {kw[0]: tokenWeight(kw) for kw in query["best_kws"]}
    newq = deepcopy(query)

    func = globals()[mod["name"]]
    if mod.get("param"):
        func(kws, mod["param"])
    else:
        func(kws)

    newq["best_kws"] = [k for k in kws.items()]
    newq["structured_query"] = makeQueryFromContext(newq)
    return newq


def getModName(mod):
    if mod.get("param"):
        return mod["name"] + "_" + str(mod["param"])
    else:
        return mod["name"]


def testWeightDifference(corpus, queries):
    retrieval = ElasticRetrievalBoost("idx_az_ilc_az_annotated_aac_2010_1_paragraph",
                                      ""
                                      )

    params = {"_all_text": 1}

    mods = [
        {"name": "baselineW"},

        # {"name": "allSetW",
        #  "param": 1},
        # {"name": "allSetW",
        #  "param": 0.5},
        # {"name": "multiplyW",
        #  "param": 2},
        # {"name": "multiplyW",
        #  "param": 4},
        # {"name": "addRandomW"},
        # {"name": "randomW"},
        # {"name": "addW",
        #  "param": 1},
        # {"name": "addW",
        #  "param": 2},
        # {"name": "addW",
        #  "param": 0.1},
        #
        # {"name": "takeMaxW",
        #  "param": 1},
        # {"name": "takeMaxW",
        #  "param": 2},
        # {"name": "takeMaxW",
        #  "param": 3},
        #
        # {"name": "takeMinW",
        #  "param": 1},
        # {"name": "takeMinW",
        #  "param": 2},
        # {"name": "takeMinW",
        #  "param": 3},
        #
        # {"name": "takeMinW",
        #  "param": 4},
        # {"name": "takeMinW",
        #  "param": 5},
        # {"name": "takeMinW",
        #  "param": 6},
        #
        # {"name": "invertW",
        #  "param": 1},
        # {"name": "invertW",
        #  "param": 2},
        # {"name": "invertW",
        #  "param": 0.5},

        # {"name": "adjustDistW",
        # "param": 0.5},
        # {"name": "adjustDistW",
        # "param": 1},
        # {"name": "adjustDistW",
        # "param": 0.3},

        # {"name": "normW"},
        # {"name": "sigmoidW"},
        # {"name": "powerW",
        #  "param":2},
        # {"name": "powerW",
        #  "param":5},
        # {"name": "powerW",
        #  "param":-1},
        # {"name": "powerW",
        #  "param":0.2},
        # {"name": "powerW",
        #  "param":0.01},

        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 1,
        #            "mul": 0.9,
        #            "reverse": False}},
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 2,
        #            "mul": 0.9,
        #            "reverse": False}},
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 3,
        #            "mul": 0.9,
        #            "reverse": False}},
        #
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 1,
        #            "mul": 0.7,
        #            "reverse": False}},
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 2,
        #            "mul": 0.7,
        #            "reverse": False}},
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 3,
        #            "mul": 0.7,
        #            "reverse": False}},
        #
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 1,
        #            "mul": 0.9,
        #            "reverse": True}},
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 2,
        #            "mul": 0.9,
        #            "reverse": True}},
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 3,
        #            "mul": 0.9,
        #            "reverse": True}},
        #
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 1,
        #            "mul": 0.7,
        #            "reverse": True}},
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 2,
        #            "mul": 0.7,
        #            "reverse": True}},
        # {"name": "takeMinMaxDivW",
        #  "param": {"num": 3,
        #            "mul": 0.7,
        #            "reverse": True}},
        #
        # {"name": "tweakW",
        #  "param": {"min_num": 2,
        #            "min_mul": 0.2,
        #            "max_num": 2,
        #            "max_mul": 0.7,
        #            }},

        # {"name": "tweakW",
        #  "param": {"min_num": 2,
        #            "min_mul": 0.3,
        #            "max_num": 2,
        #            "max_mul": 0.8,
        #            }},
        #
        # {"name": "tweakW",
        #  "param": {"min_num": 2,
        #            "min_mul": 0.1,
        #            "max_num": 2,
        #            "max_mul": 1.2,
        #            }},
        {"name": "attenuateSigmoidW",
         "param": {"pow": 180
                   }},
        # {"name": "attenuateExpW",
        #  "param": {"pow": -1
        #            }
        #  },
        # {"name": "attenuateExpW",
        #  "param": {"pow": 2
        #            }
        #  },
        # {"name": "attenuateExpW",
        #  "param": {"pow": 4
        #            }
        #  },

    ]

    # for rev in [True, False]:
    #     for num in [1, 2, 3]:
    #         for mul in [0.1, 0.3, 0.5, 1.2, 1.4, 1.6]:
    #             mod = {"name": "takeMinMaxDivW",
    #                    "param": {"num": num,
    #                              "mul": mul,
    #                              "reverse": rev}}
    #             mods.append(mod)

    results = {getModName(m): [] for m in mods}

    for query in tqdm(queries):
        # print(query["best_kws"])
        for mod in mods:
            newq = runMod(query, mod)
            if not newq:
                print("Empty query!")
                continue
            scores = runAndMeasureOneQuery(newq, params, retrieval)
            results[getModName(mod)].append(scores["ndcg_score"])
            # results[getModName(mod)].append(scores["precision_score"])

    scores = []

    for mod in results:
        score = sum(results[mod]) / len(results[mod])
        scores.append((mod, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    for result in scores:
        print("%s %0.4f" % result)


def main():
    corpus = ez_connect("AAC", "koko")
    exp_dir = os.path.join(getRootDir("aac"), "experiments", "aac_generate_kw_trace")
    filename = os.path.join(exp_dir, "feature_data_test_w.json.gz")
    # filename = os.path.join(exp_dir, "feature_data_test_mms.json.gz")
    features = FeaturesReader(filename).getMiniBatch(1000)
    testWeightDifference(corpus, features)


if __name__ == '__main__':
    main()
