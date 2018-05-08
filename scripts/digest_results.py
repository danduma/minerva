import os, glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from numpy.random import rand


def listAllResultsFiles(path):
    """
    Returns a list of all

    :param path: path to search
    :return: all results files
    """
    return glob.glob(os.path.join(path, "*results*.csv"))


def loadAllResultsFiles(path):
    """

    :param path: path: path to search
    :return: DataFrame with all results
    """
    all_results = pd.DataFrame()
    for filename in listAllResultsFiles(path):
        df = pd.read_csv(filename, sep="\t")
        all_results = pd.concat([all_results, df])

    return all_results


def makeResultsGraph(path, table, metric="mrr_score", graphfile="results.png"):
    """
    Renders the pivot table as a bar graph

    :param path:
    :param table:
    :return:
    """

    values = table["Mean"]
    labels = []
    for item in table.iterrows():
        # if item[0] != "Mean":
        labels.append(item[0])

    positions = np.arange(start=0, stop=len(labels), step=0.5) + .5  # the bar centers on the y axis
    positions = positions[:len(values)]
    print(labels)
    print(positions)
    print(values)

    fig, ax = plt.subplots()
    width = 0.2  # the width of the bars
    ind = np.arange(len(values))  # the x locations for the groups

    ax.barh(positions, values, align='center', height=width)

    # ax.set_yticks(ind + width / 2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, minor=False)

    plt.gcf().subplots_adjust(left=0.35)

    # ax.set_yticks(pos, ('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))

    # ax.axvline(0,color='k',lw=3)   # poor man's zero level


    ax.set_xlabel('Performance metric')
    ax.set_title('All document representation methods')
    ax.grid(True)
    plt.savefig(os.path.join(path, graphfile), format="png")
    plt.show()


def digestResultsForPresentation(path, metric="mrr_score"):
    """
    Given an experiment directory, it lists all results.csv files, joins them all into a single file,
    creates the pivot table, outputs the table for including in paper, draws graph

    :return:
    """

    data = loadAllResultsFiles(path)
    print("metric", metric)
    table = pd.pivot_table(data,
                           values=metric,
                           index=['query_method'],
                           columns=['doc_method'],
                           aggfunc=np.mean)
    full_table = table.copy()
    full_table['Mean'] = full_table.mean(numeric_only=True, axis=1)
    full_table.loc['Mean'] = full_table.mean()
    full_table.to_csv(os.path.join(path, "pivot_output.csv"))

    makeResultsGraph(path, full_table)


def main(path, metric="mrr_score"):
    # print(listAllResultsFiles("/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac"))
    # print(len(loadAllResultsFiles("/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac")))
    digestResultsForPresentation(path, metric=metric)


if __name__ == '__main__':
    import plac

    plac.call(main)
