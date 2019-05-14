from models.keyword_features import FeaturesReader, getAnnotationListsForContext, tokenWeight, getRootDir

import os
import gzip
import json
from tqdm import tqdm


def getOneContextFeatures(context):
    """
        Prepares a single context's data for any nn. Takes ["token_features"] from list
        of sentences and returns a single list of token features.
    """
    all_keywords = {t[0]: tokenWeight(t) for t in context["best_kws"]}

    featureset = getAnnotationListsForContext(context["tokens"], all_keywords)
    tokens, to_extract, weights = zip(*featureset)
    max_weight = max(weights)
    if max_weight > 0:
        weights = [w / max_weight for w in weights]

    context["extract_mask"] = to_extract
    context["tokens"] = tokens
    context["weight_mask"] = weights

    return context


def processContexts(contexts):
    res = []
    for context in contexts:
        res.append(getOneContextFeatures(context))
    return res


def saveFixedContexts(path, precomputed_contexts):
    if ".gz" in path:
        f = gzip.open(path, "wt")
    else:
        f = open(path, "w")

    written = 0
    print("Exporting feature data...")
    for context in tqdm(precomputed_contexts):
        feature_data = getOneContextFeatures(context)

        f.write(json.dumps(feature_data) + "\n")
        written += 1

    f.close()
    print("Total written", written)


def main():
    exp_dir = os.path.join(getRootDir("aac"), "experiments", "aac_generate_kw_trace")
    to_process = [
        "feature_data_w.json.gz",
        # "feature_data_test_w.json.gz"
    ]
    for filename in to_process:
        full_filename = os.path.join(exp_dir, filename)
        features = FeaturesReader(full_filename)
        saveFixedContexts(full_filename + ".fixed", features)


if __name__ == '__main__':
    main()
