import os, glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

import seaborn as sns

sns.set()

rcParams.update({'figure.autolayout': True})


def mk_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups


# def add_line(ax, xpos, ypos):
#     line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
#                       transform=ax.transAxes, color='black')
#     line.set_clip_on(False)
#     ax.add_line(line)

def add_line(ax, ypos, xpos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)


def label_group_bar(ax, data):
    groups = mk_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)

    ax.bar(xticks, y, align='center')
    ax.set_xticks(xticks)
    ax.set_xticklabels(x)
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True)

    scale = 1. / ly
    for pos in range(ly + 1):
        add_line(ax, pos * scale, -.1)
    ypos = -.2
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos * scale, ypos)
            pos += rpos
        add_line(ax, pos * scale, ypos)
        ypos -= .1


def listAllResultsFiles(path):
    """
    Returns a list of all

    :param path: path to search
    :return: all results files
    """
    return glob.glob(os.path.join(path, "*_results.csv"))


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


def addNumberToBars(rects, ax):
    rect_labels = []
    # Lastly, write in the ranking inside each bar to aid in interpretation
    for rect in rects:
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = rect.get_width()

        text = str(width)
        # The bars aren't wide enough to print the ranking inside
        if width < 0.05:
            # Shift the text to the right side of the right edge
            xloc = width + 0.01
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = 0.98 * width
            # White on magenta
            clr = 'white'
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height() / 2.0
        label = ax.text(xloc, yloc, text,
                        horizontalalignment=align,
                        verticalalignment='center',
                        color=clr,
                        weight='bold',
                        fontsize="10",
                        clip_on=False)
        rect_labels.append(label)
    return rect_labels


def makeDocumentMethodsGraph(path, data, metric="mrr_score",
                             graphfile="results.png",
                             title="Document representation performance",
                             column="Mean"):
    """
    Renders the pivot table for document methods as a bar graph

    :param path:
    :param table:
    :param data: the pivot table to be rendered
    :param metric: the label to use as
    :return:
    """

    internal_methods = {}
    external_methods = {}
    mixed_methods = {}

    labels = []

    for label, value in data.items():
        if label.startswith("ilc_"):
            labels.append(label)
            mixed_methods[label] = value
        elif "inlink_context" in label:
            labels.append(label)
            external_methods[label] = value
        elif label == "Mean":
            mean = (label, value)
        elif label == "Max":
            max = (label, value)
        else:
            labels.append(label)
            internal_methods[label] = value

    i_methods = pd.DataFrame(pd.Series(internal_methods), columns=[column])
    e_methods = pd.DataFrame(pd.Series(external_methods), columns=[column])
    m_methods = pd.DataFrame(pd.Series(mixed_methods), columns=[column])

    i_methods = i_methods.sort_values(by="Mean", ascending=False)
    e_methods = e_methods.sort_values(by="Mean", ascending=False)
    m_methods = m_methods.sort_values(by="Mean", ascending=False)

    data = pd.concat([m_methods, e_methods, i_methods])

    values = [x[0] for x in data.values]

    fig, ax = plt.subplots(figsize=(7, 7))

    width = 0.7  # the width of the bars

    # colours = ["r", "g", "b"]
    colours=sns.color_palette()
    positions = []
    line_cols = []
    last_pos = 0
    for index, method_type in enumerate([mixed_methods, external_methods, internal_methods]):
        for item in method_type:
            positions.append(last_pos)
            last_pos += width + 0.2
            line_cols.append(colours[index])
        last_pos += 0.5

    # positions = np.arange(start=0, stop=len(labels), step=0.5) + .5  # the bar centers on the y axis
    # positions = positions[:len(values)]

    rects = ax.barh(positions, values, align='center', height=width, color=line_cols)

    addNumberToBars(rects, ax)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, minor=False)

    # plt.gcf().subplots_adjust(left=0.35)

    # ax.axvline(0,color='k',lw=3)   # poor man's zero level

    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="x")
    # axes = data.plot.barh()

    # for i, v in enumerate(values):
    #     fig.text(v + 3, i + .25, str(v), color='blue', fontweight='bold', fontsize="30")

    # THE STRUCTURED WAY
    # data = {'Internal methods': internal_methods,
    #         'External methods': external_methods,
    #         'Mixed methods': mixed_methods,
    #         }
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel(metric)
    # ax.set_title(title)
    # ax.grid(True)
    # label_group_bar(ax, data)
    # fig.subplots_adjust(bottom=0.3)

    plt.savefig(os.path.join(path, graphfile), format="png", dpi=300)


def makeQueryMethodsGraph(path, data, metric="mrr_score", graphfile="results.png",
                          title="Query method performance"):
    """
    Renders the pivot table as a bar graph

    :param path:
    :param table:
    :return:
    """
    values = []
    labels = []
    line_cols = []
    colours=sns.color_palette()

    for label, value in data.items():
        if label not in ["Max", "Mean"]:
            labels.append(label)
            values.append(value)
        if label.startswith("sentence"):
            line_cols.append(colours[1])
        elif label.startswith("window"):
            line_cols.append(colours[0])

    positions = np.arange(start=0, stop=len(labels), step=0.5) + .5  # the bar centers on the y axis
    positions = positions[:len(values)]

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 0.4  # the width of the bars

    rects = ax.barh(positions, values, align='center', height=width, color=line_cols)
    addNumberToBars(rects, ax)

    # ax.set_yticks(ind + width / 2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, minor=False)

    plt.gcf().subplots_adjust(left=0.35)

    # ax.axvline(0,color='k',lw=3)   # poor man's zero level

    # txt = ax.text(0.5, 0.2, 'text', color='blue', fontweight='bold', fontsize="15")

    # for i, v in enumerate(values):
    #     text = str(v)
    #     ax.text(v - (0.01 * (len(text))),
    #             positions[i] - (width / 4),
    #             text,
    #             fontsize="10",
    #             color='white',
    #             # fontweight='bold'
    #             )

    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(True)



    plt.savefig(os.path.join(path, graphfile), format="png", dpi=300)
    # plt.show()


def makeResultsGraph(path, data, metric="mrr_score", graphfile="results.png"):
    """
    Renders the pivot table as a bar graph

    :param path:
    :param table:
    :return:
    """
    values = data.values
    labels = data.keys()
    positions = np.arange(start=0, stop=len(labels), step=0.5) + .5  # the bar centers on the y axis
    positions = positions[:len(values)]

    fig, ax = plt.subplots()
    width = 0.4  # the width of the bars
    ind = np.arange(len(values))  # the x locations for the groups

    # ax.barh(positions, values, align='center', height=width)

    # ax.set_yticks(ind + width / 2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, minor=False)

    plt.gcf().subplots_adjust(left=0.35)

    # ax.set_yticks(pos, ('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))

    # ax.axvline(0,color='k',lw=3)   # poor man's zero level

    # for i, v in enumerate(values):
    #     fig.text(v + 3, i + .25, str(v), color='blue', fontweight='bold', fontsize="30")
    #
    # ax.set_xlabel(metric)
    ax.set_title('All document representation methods')
    ax.grid(True)
    plt.savefig(os.path.join(path, graphfile), format="png", dpi=300)
    # plt.show()


def savePrintPivotTable(table, path):
    table = table.sort_values(by="Max", ascending=True)
    new_table = table.transpose()
    new_table = new_table.sort_values(by="Max", ascending=True)
    new_table.to_csv(path)


def makeHeatmap(data, path, corpus="AAC", graphfile="pivot_heatmap.png"):
    fig, ax = plt.subplots(figsize=(10, 10))

    data = data.sort_values(by="Mean", ascending=True)
    data = data.drop(["Mean", "Max"], axis=1)
    data = data.transpose()
    data = data.sort_values(by="Mean", ascending=True)
    data = data.drop(["Mean", "Max"], axis=1)
    data.index.names = ["Document representation methods"]
    data.columns.names = ["Query extraction methods"]

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .15}
    # f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(30, 12))
    # ax = sns.heatmap(data, ax=ax,
    #                  cbar_ax=cbar_ax,
    #                  cbar_kws={"orientation": "vertical", "label": "Number of accidents"},
    #                  cmap="YlGnBu")
    # set font size of colorbar labels
    # cbar_ax.tick_params(labelsize=25)

    # ax = sns.heatmap(data, cmap="YlGnBu")
    sns.set(font_scale=1.2)
    ax = sns.heatmap(data,
                     cmap=sns.diverging_palette(0, 130, sep=1, n=100, as_cmap=True),
                     linewidths=.5,
                     annot=True,
                     annot_kws={"size": 12},
                     fmt=".2f",
                     ax=ax)
    plt.savefig(os.path.join(path, graphfile), format="png", dpi=300)


def digestResultsForPresentation(path, metric="mrr_score", corpus="aac"):
    """
    Given an experiment directory, it lists all results.csv files, joins them all into a single file,
    creates the pivot table, outputs the table for including in paper, draws graph

    :return:
    """
    data = loadAllResultsFiles(path)
    data.to_csv(os.path.join(path, "all_results_data.csv"))
    print("metric", metric)
    table = pd.pivot_table(data,
                           values=metric,
                           index=['query_method'],
                           columns=['doc_method'],
                           aggfunc=np.mean)
    full_table = table.copy()
    full_table['Mean'] = full_table.mean(numeric_only=True, axis=1)
    full_table.loc['Mean'] = full_table.mean()

    full_table['Max'] = full_table.max(numeric_only=True, axis=1)
    full_table.loc['Max'] = full_table.max()

    full_table = full_table.round(3)
    full_table = full_table.sort_values(by="Mean", ascending=True)

    savePrintPivotTable(full_table, os.path.join(path, "pivot_output.csv"))

    if corpus == "aac":
        corpus_label = "AAC"
    elif corpus == "pmc":
        corpus_label = "PMC-OAS"

    sns.set()

    makeHeatmap(full_table, path, graphfile="pivot_heatmap_%s.png" % corpus)

    makeQueryMethodsGraph(path, full_table["Max"].sort_values(ascending=False),
                          metric="Maximum average MRR score",
                          graphfile="query_method_results_%s.png" % corpus,
                          title="Query method performance (%s)" % corpus_label
                          )

    makeDocumentMethodsGraph(path, full_table.loc["Max"].sort_values(ascending=False),
                             metric="Maximum average MRR score",
                             graphfile="doc_method_results_%s.png" % corpus,
                             title="Document method performance (%s)" % corpus_label)


def main(path, metric="mrr_score"):
    # print(listAllResultsFiles("/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac"))
    # print(len(loadAllResultsFiles("/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac")))

    if "/aac/" in path:
        corpus = "aac"
    elif "/pmc" in path:
        corpus = "pmc"

    digestResultsForPresentation(path, metric=metric, corpus=corpus)

    # full_table = pd.read_csv(os.path.join(path, "pivot_output.csv"), index_col="query_method")
    # makeDocumentMethodsGraph(path, full_table.loc["Mean"].sort_values(ascending=False), metric="MRR score", graphfile="doc_method_results.png")
    # makeHeatmap(full_table, path, graphfile="pivot_heatmap_%s.png" % corpus)


if __name__ == '__main__':
    import plac

    plac.call(main)
