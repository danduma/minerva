import pandas as pd
import numpy as np
import os
import random

from tqdm import tqdm

from scripts.digest_results import addUniqueIDToCit, addUniqueIDWithLineNumberToCit

COLUMNS_TO_DROP = ["query_id", "file_guid", "citation_id", "doc_position", "first_result", "match_guid"]


def getRowsC6RepeatedAnova(df, methods, keys, metric="ndcg_score"):
    res = []
    for key in keys:
        row = {"unique_cit_id": key}
        for method in methods:
            row[method] = df.loc[(df["unique_cit_id"] == key) & (df["ml_method"] == method)][metric].iloc[0]
            # print(key, method, row[method])
        res.append(row)
    return res


def getRowsC6Anova(df, methods, keys, metric="ndcg_score"):
    res = []
    for key in keys:

        for method in methods:
            row = {"unique_cit_id": key}
            row[metric] = df.loc[(df["unique_cit_id"] == key) & (df["ml_method"] == method)][metric].iloc[0]
            row["ml_method"] = method
            res.append(row)
    return res


def joinCSVsForC6_10(data_dir: str, files: dict, output_filename: str, test="repeated_ANOVA", metric="ndcg_score"):
    """
    Joins the evaluation results of several ML methods and exports a CSV for running ANOVAs
    in JASP or similar.

    :param data_dir: working dir with sources and targe files
    :param files: dict of {"path": "method name"}
    :param output_filename: CSV to write it all to
    :return:
    """
    df = pd.DataFrame()
    for path in files:
        df2 = pd.read_csv(os.path.join(data_dir, path), sep="\t")
        if len(df2.columns) < 2:
            df2 = pd.read_csv(os.path.join(data_dir, path), sep=",")
        addUniqueIDToCit(df2)
        df2.drop(columns=COLUMNS_TO_DROP, inplace=True)
        df2["ml_method"] = files[path]
        df = pd.concat([df, df2])

    methods = files.values()
    keys = df["unique_cit_id"].unique()
    if test == "repeated_ANOVA":
        res = getRowsC6RepeatedAnova(df, methods, keys, metric=metric)
    elif test == "ANOVA":
        res = getRowsC6Anova(df, methods, keys, metric=metric)
    else:
        raise ValueError("Unknown test")

    res_df = pd.DataFrame(res)
    res_df.to_csv(os.path.join(data_dir, output_filename))


def joinCSVsForC6_8(path: str, output_filename: str, metric="ndcg_score"):
    """
    Exports the data from kw_selection_trace.csv to a CSV for running paired t-tests
    in JASP or similar.

    :param data_dir: working dir with sources and targe files
    :param files: dict of {"path": "method name"}
    :param output_filename: CSV to write it all to
    :return:
    """
    data_dir = os.path.dirname(path)
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) < 2:
        df = pd.read_csv(path, sep=",")

    results = []
    for index in range(0, df.shape[0], 2):
        res = {"unique_cit_id": df.iloc[index]["file_guid"] + "_" +
                                df.iloc[index]["cit_ids"][0] + "_" +
                                df.iloc[index]["match_guids"][0],
               "ndcg_score": df.iloc[index]["ndcg_score"],
               "c3_kw_score": df.iloc[index]["ndcg_score_kw"],
               "c3_kw_score_weight": df.iloc[index]["ndcg_score_kw_weight"],
               "c6_kw_score": df.iloc[index + 1]["ndcg_score_kw"],
               "c6_kw_score_weight": df.iloc[index + 1]["ndcg_score_kw_weight"],
               }
        # print(res["c3_kw_score_weight"], res["c6_kw_score_weight"], )
        results.append(res)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(data_dir, output_filename))


def getRowsRepeatedANOVA(df, selected_best, metric):
    all_cits = df["unique_cit_id"].unique()
    results = []
    for cit_id in tqdm(all_cits):
        res_dict = {"cit_id": cit_id}

        for method_type in ["int", "ext", "mix"]:
            doc_method = selected_best[method_type]["doc_method"]
            query_method = selected_best[method_type]["query_method"]
            key = method_type + "_" + doc_method + "_" + query_method
            value = df[(df["doc_method"] == doc_method) & (df["query_method"] == query_method) & (
                    df["unique_cit_id"] == cit_id)][metric].iloc[0]
            res_dict[key] = value
        results.append(res_dict)

    return results


def getRowsANOVA(df, selected_best, metric):
    all_cits = df["unique_cit_id"].unique()
    results = []
    for cit_id in tqdm(all_cits):

        for method_type in ["int", "ext", "mix"]:
            res_dict = {"cit_id": cit_id}
            doc_method = selected_best[method_type]["doc_method"]
            query_method = selected_best[method_type]["query_method"]

            key = method_type + "_" + doc_method + "_" + query_method
            value = df[(df["doc_method"] == doc_method) & (df["query_method"] == query_method) & (
                    df["unique_cit_id"] == cit_id)][metric].iloc[0]
            res_dict["method"] = key
            res_dict[metric] = value
            results.append(res_dict)

    return results


def prepareCSVForStatsC3(path, test="repeated_ANOVA", metric="ndcg_score"):
    """

    :param path:
    :return:
    """
    data_dir = os.path.dirname(path)
    df = pd.read_csv(path)
    addUniqueIDToCit(df)

    # df = df.drop(columns=COLUMNS_TO_DROP)

    table = pd.pivot_table(df,
                           values=metric,
                           index=['query_method'],
                           columns=['doc_method'],
                           aggfunc=np.mean)

    best_method = pd.DataFrame(table.idxmax(0))
    best_method.columns = ["query_method"]
    best_method["int_ext_mix"] = pd.Series(["-" for _ in range(best_method.shape[0])], index=best_method.index)
    best_method[metric] = pd.Series([0.0 for _ in range(best_method.shape[0])], index=best_method.index)

    for index, row in best_method.iterrows():
        method_type = ""
        if index.startswith("ilc_"):
            method_type = "mix"
        elif index.startswith("title_abstract") or index.startswith("full_text"):
            method_type = "int"
        elif index.startswith("inlink_context"):
            method_type = "ext"

        best_method.at[index, "int_ext_mix"] = method_type
        best_method.at[index, metric] = table[index].loc[row["query_method"]]
        pass

    selected_best = {}
    for method_type in ["int", "ext", "mix"]:
        selected_best[method_type] = {
            "doc_method": best_method[best_method["int_ext_mix"] == method_type][metric].idxmax(axis=1)}
        selected_best[method_type]["query_method"] = best_method.loc[selected_best[method_type]["doc_method"]][
            "query_method"]
        pass

    if test == "repeated_ANOVA":
        results = getRowsRepeatedANOVA(df, selected_best, metric)
    elif test == "ANOVA":
        results = getRowsANOVA(df, selected_best, metric)

    # print(selected_best)
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(data_dir, "all_results_data_analysis_" + test + ".csv"))


def generateRandomGroups(num_data_points, group_size, group_num):
    return [random.choices(range(num_data_points), k=group_size) for _ in range(group_num)]


def doC6_10():
    joinCSVsForC6_10("/Users/masterman/NLP/PhD/aac/experiments/aac_generate_kw_trace",
                     {
                         "16-9-2018_1-20_results_0373.csv": "Baseline2",
                         "16-9-2018_3-10_results_0383.csv": "LinearRegression",
                         "16-9-2018_1-37_results_0391.csv": "ExtraTrees",
                         "16-9-2018_2-09_results_0407.csv": "MLP",
                     },
                     "aac_ml_methods_anova.csv"
                     , test="ANOVA"
                     # "aac_ml_methods_repeated_anova.csv"
                     # , test = "repeated_ANOVA"
                     )

    joinCSVsForC6_10("/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_generate_kw_trace",
                     {
                         "6-5-2019_18-09_results_0349.csv": "Baseline2",
                         "16-9-2018_22-49_results_0340.csv": "LinearRegression",
                         "20-9-2018_19-49_results_0317.csv": "ExtraTrees",
                         "20-9-2018_17-02_results_0344.csv": "MLP",
                     },
                     # "pmc_ml_mesthods_anova.csv"
                     # , test="ANOVA"
                     "pmc_ml_methods_repeated_anova.csv"
                     , test="repeated_ANOVA"
                     )


def doC6_8():
    joinCSVsForC6_8("/Users/masterman/NLP/PhD/aac/experiments/aac_full_text_kw_selection/kw_selection_trace.csv",
                    "c6_aac_ttest.csv")
    joinCSVsForC6_8("/Users/masterman/NLP/PhD/pmc_coresc/experiments/pmc_full_text_kw_selection/kw_selection_trace.csv",
                    "c6_pmc_ttest.csv")


def doC3():
    prepareCSVForStatsC3("/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac/all_results_data.csv",
                         test="ANOVA")
    prepareCSVForStatsC3(
        "/Users/masterman/NLP/PhD/pmc_coresc/experiments/thesis_chapter3_experiment_pmc/all_results_data.csv",
        test="ANOVA")


def main():
    # doC3()
    # doC6_10()
    doC6_8()


if __name__ == '__main__':
    main()
