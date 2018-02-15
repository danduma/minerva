#-------------------------------------------------------------------------------
# Name:        results_analysis
# Purpose:     Contains graphing and analysis functions to look at results
#
# Author:      dd
#
# Created:     12/02/2015
# Copyright:   (c) dd 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import print_function
import glob

import pandas
from pandas import Series, DataFrame
from scipy.stats.mstats import mode
import matplotlib.pyplot as plt
import seaborn as sns

import db.corpora as cp
from proc.general_utils import ensureTrailingBackslash, getFileDir, getFileName
from six.moves import range
##from testingPipeline5 import measureCitationResolution, AZ_ZONES_LIST, CORESC_LIST

def generateEqualWeights():
    weights={x:INITIAL_VALUE for x in all_doc_methods[method]["runtime_parameters"]}

def statsOnResults(data, metric="avg_mrr"):
    """
        Returns the mean of the top results that have the same number
    """
    res={}
    if len(data)==0:
        return
    val_to_match=data[metric].iloc[0]
    index=0
    while data[metric].iloc[index]==val_to_match:
        index+=1
    lines=data.iloc[:index]
    means=lines.mean()
    print("Averages:")
    for zone in AZ_ZONES_LIST:
        res[zone]=means[zone]
        print(zone,":",means[zone])
    return res

def drawSimilaritiesGraph(filename,metric, smooth_graph=False):
    """
        Draws graph from CSV file
    """
    dir=getFileDir(filename)
    if dir=="":
        filename=cp.Corpus.dir_output+filename

    print("Drawing graph for",filename)
    data=pandas.read_csv(filename)
##    columns=AZ_ZONES_LIST+[metric]

    columns=[u"ilc_CSC_"+zone for zone in CORESC_LIST]+[metric]
    if columns[0] not in data.columns:
        columns=[zone for zone in CORESC_LIST]+[metric]
        if columns[0] not in data.columns:
            columns=[zone for zone in AZ_ZONES_LIST]+[metric]

##    columns=["OWN","OTH","CTR","BKG","AIM"]+[metric]
##    columns=["OWN","OTH","CTR"]+[metric]

##    f = lambda x: mode(x, axis=None)[0]
##    print data[columns].head(20).apply(f)

##    print data.describe()
    numrows=data.shape[0] # (y,x)
##    print data[columns].head(10).mean()

    data=data.sort(metric, ascending=False)

    # smoothing function
    rows_to_group=100
##    rows_to_group=numrows/700
    f=lambda x:100-(x/rows_to_group)

    results=[]

##    index=0
##    while index < numrows:
##        means=data[columns].iloc[index:index+rows_to_group].mean()
####        print means[metric]
##        index+=rows_to_group
##        results.append(means)
##    results.reverse()

    if smooth_graph:
        df=data[columns].groupby([f], sort=True)
        results=df[columns].mean()
    else:
        data["g_"+metric]=data[metric]
        results=data.groupby(["g_"+metric])[columns].mean()

##    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "baby pink"]
##    sns.palplot(sns.xkcd_palette(colors))
##    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#002354", ""]
##    sns.palplot(sns.color_palette(flatui))
##    print sns.color_palette("Set2")

##    sns.set_style("white")
    sns.set_style("whitegrid")
##    sns.set_style("whitegrid", {"grid.linewidth": .5})
##    sns.set_context("talk")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
##    sns.set_palette(sns.color_palette("hls", 7))
##    sns.set_palette(sns.color_palette("husl", 7))
##    sns.set_palette("bright",n_colors=14,desat=0.9)
    sns.set_palette(sns.color_palette("gist_rainbow", 11),11,0.9)
##    sns.palplot(sns.color_palette())
##    sns.palplot(sns.color_palette(n_colors=14) )

    results_data=DataFrame(results)
##    results_data.plot(x=metric, kind="line")

    results_data[columns].plot(title=filename,fontsize=20, x=metric)
    sns.despine()

def compareResultsBetweenSimilarities(filenames):
    """
        Loads a CSV of parameter-based results, compares selected measures by
        re-running them with a different simliarity, outputs results
    """
    metric="precision_total"
##    metric="avg_mrr"
    for filename in filenames:
    ##    data['precision_2'] = Series(0, index=data.index)
        drawSimilaritiesGraph(filename,metric, True)

    plt.show()
##    plt.savefig()


def saveGraphForResults(filename,metric):
    """
    """
    dir=ensureTrailingBackslash(getFileDir(filename))
    drawSimilaritiesGraph(filename,metric,True)
    name=getFileName(filename)
    plt.savefig(dir+name+'.png', bbox_inches='tight')
    plt.close()


def computeOverlap(filename, overlap_in="rank", overlap_between=["az_annotated_1_ALL","az_annotated_1_ALL_EXPLAIN"]):
    """
    """
    data=pandas.read_csv(cp.Corpus.dir_output+filename)
##    data['precision_2'] = Series(0, index=data.index)
##    data=DataFrame(self.overall_results)
    print(data.describe())
    group=data.groupby(["file_guid","citation_id"])

    all_overlaps=[]
    for a,b in group:
        numitems=b.shape[0]
        results=[]
        for i in range(numitems):
            doc_method=b.iloc[i]["doc_method"]
            rank=b.iloc[i][overlap_in]
            if doc_method in overlap_between:
                results.append(rank)

        this_one=1
        for index in range(len(results)-1):
            if results[index] != results[index+1]:
                this_one=0
                break

        all_overlaps.append(this_one)

    print("Overlap between", overlap_between," in ",overlap_in,": %02.4f" % (sum(all_overlaps) / float(len(all_overlaps))))


def drawGraphOfScorePerMethod(data):
    """
    """
    # !TODO IMPLEMENT
    columns=[]

    print(data.describe())
    numrows=data.shape[0] # (y,x)
    print(data[columns].head(10).mean())

    data=data.sort(metric, ascending=False)

    rows_to_group=100
    f=lambda x:100-(x/rows_to_group)

    results=[]

##    index=0
##    while index < numrows:
##        means=data[columns].iloc[index:index+rows_to_group].mean()
####        print means[metric]
##        index+=rows_to_group
##        results.append(means)
##    results.reverse()

##    data["g_"+metric]=data[metric]
##    results=data.groupby(["g_"+metric])[columns].mean()
    results=data[columns].groupby([f], sort=True)[columns].mean()

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_palette("bright",8,0.9)


    results_data=DataFrame(results)
##    results_data.plot(x=metric, kind="line")

    results_data[columns].plot(title=filename,fontsize=20, x=metric)
    sns.despine()


def compareResultsBetweenMethods(filename, metric="precision_total"):
    """
    """
    # !TODO IMPLEMENT
    data=pandas.read_csv(cp.Corpus.dir_output+filename)
##    metric="avg_mrr"
    for filename in filenames:
    ##    data['precision_2'] = Series(0, index=data.index)
        drawSimilaritiesGraph(filename,metric)

    plt.show()


def drawNewSimilaritiesGraph(filename, metric, single_out="OWN", smooth_graph=False):
    """
        Draws graph from CSV file
    """
    data=pandas.read_csv(cp.Corpus.dir_output+filename)
    columns=AZ_ZONES_LIST+[metric]

##    f = lambda x: mode(x, axis=None)[0]
##    print data[columns].head(20).apply(f)

    for zone in AZ_ZONES_LIST:
        data["pct_"+zone]=data[zone]/data[AZ_ZONES_LIST].sum(axis=1)
        columns.append("pct_"+zone)

    print(data.describe())
    numrows=data.shape[0] # (y,x)
    print(data[columns].head(10).mean())

    data=data.sort(metric, ascending=False)

    # smoothing function
    rows_to_group=100
    f=lambda x:100-(x/rows_to_group)

    results=[]

    if smooth_graph:
        results=data[columns].groupby([f], sort=True)[columns].mean()
    else:
        data["g_"+metric]=data[metric]
        results=data.groupby(["g_"+metric])[columns].mean()

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_palette("bright",8,0.9)

    results_data=DataFrame(results)
##    results_data.plot(x=metric, kind="line")

    ax=results_data[["pct_"+single_out,metric]].plot(title=filename,fontsize=20, y=metric, x="pct_"+single_out)
    ax.set_ylabel(metric)
    sns.despine()


def compareNewSimilaritiesGraph(filenames):
    """
        Loads a CSV of parameter-based results, compares selected measures by
        re-running them with a different simliarity, outputs results
    """
    metric="precision_total"
##    metric="avg_mrr"
    for filename in filenames:
    ##    data['precision_2'] = Series(0, index=data.index)
        drawNewSimilaritiesGraph(filename,metric, "BKG",True)

    plt.show()

def makeAllGraphsForExperiment(exp_dir):
    """
        Iterates through all weight*.csv files in the experiment's directory and
        saves a graph for each
    """
##    metric="avg_mrr"
##    metric="precision_total"
    metric="avg_precision"
    exp_dir=ensureTrailingBackslash(exp_dir)
##    working_dir=Corpus.dir_experiments+exp_name+os.sep
    for path in glob.glob(exp_dir+"weight*.csv"):
        saveGraphForResults(path,metric)

def drawWeights(exp,weights,name):
    """
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_palette(sns.color_palette("gist_rainbow", 11),11,0.9)

    results_data=DataFrame(weights,[0])
    results_data.plot(kind="bar",title="weights",fontsize=20, ylim=(-15,15))
    sns.despine()
##    plt.show()

##    name=getFileName(name)
    plt.savefig(exp["exp_dir"]+name+'.png', bbox_inches='tight')
    plt.close()


def drawScoreProgression(exp,scores,name):
    """
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_palette(sns.color_palette("gist_rainbow", 11),11,0.9)

    print(scores)

    results_data=DataFrame(scores)
    results_data.plot(kind="line",title="scores",fontsize=20, xlim=(0,8))
    sns.despine()
##    plt.show()

    plt.savefig(exp["exp_dir"]+str(name)+'.png', bbox_inches='tight')
    plt.close()


def main():
##    filenames=["weights_OWN_max2_inc2.00_ini1_FA_bulkScorer.csv","weights_OWN_max2_inc1.00_ini1_BooleanQuery.csv"]
##    filenames=["weights_OWN_['1', '3', '5']_FA_bulkScorer_first1000.csv","weights_OWN_['1', '3', '5']_FA_bulkScorer_second1000.csv"]
##    filenames=["weights_CTR_max2_inc1.00_ini1.csv"]
##    filenames=["weights_BKG_max2_inc1.00_ini1.csv"]
##    filenames=["weights_OTH_['1', '3', '5']_FA_BulkScorer_first650.csv","weights_OTH_['1', '3', '5']_FA_BulkScorer_second650.csv"]

##    filenames=["weights_OWN_['1', '3', '5']test_optimized.csv","weights_OWN_['1', '3', '5']_FA_bulkScorer_second1000.csv"]
##    filenames=["weights_OWN_['1', '3', '5']test_optimized_defaultsim_first1000.csv","weights_OWN_['1', '3', '5']test_optimized_defaultsim_second1000.csv"]
##    filenames=["weights_OWN_['1', '3', '5']_FA_bulkScorer_first1000.csv","weights_OWN_['1', '3', '5', '7', '9']test_optimized_fa_first1000.csv"]
    filenames=["weights_OWN_['1', '3', '5', '7', '9']test_optimized_fa_first1000.csv","weights_OWN_['1', '3', '5', '7', '9']test_optimized_fa_second1000.csv"]
    filenames=["weights_OWN_['1', '3', '5', '7', '9']test_optimized_fa_second1000.csv","weights_OWN_[1, 3, 5, 7]test_optimized_fa_third1000.csv"]

    filenames=["weights_Hyp_[1, 5]_s1.csv","weights_Mot_[1, 5]_s1.csv"]
    filenames=["weights_Bac_[1, 5]_s1.csv","weights_Goa_[1, 5]_s1.csv"]

    filenames=[r"C:\NLP\PhD\bob\experiments\w20_ilcpar_csc_fa_w0135\weights_Bac_[1, 5]_s2.csv"]

##    compareResultsBetweenSimilarities(filenames)
##    compareNewSimilaritiesGraph(filenames[:1])

##    compareResultsBetweenSimilarities("weights_OWN_max2_inc2.00_ini1_FA_bulkScorer.csv")
##    compareResultsBetweenSimilarities("weights_OWN_max2_inc1.00_ini1_BooleanQuery.csv")
##    compareResultsBetweenSimilarities("weights_AIM_max3_inc1.00_ini1.csv")
##    compareResultsBetweenSimilarities("weights_BKG_max2_inc1.00_ini1.csv")
##    compareResultsBetweenSimilarities("weights_CTR_max2_inc1.00_ini1.csv")
##    compareResultsBetweenSimilarities("weights_BAS_max2_inc1.00_ini1.csv")
##    computeOverlap("overlap_bulkScorer_explain.csv", overlap_in="precision_score")

##    makeAllGraphsForExperiment(r"C:\NLP\PhD\bob\experiments\w20_csc_csc_fa_w0135")
##    makeAllGraphsForExperiment(r"C:\NLP\PhD\bob\experiments\w20_ilcpar_csc_fa_w0135")

##    drawWeights("",{"AIM":9,"BAS":5,"BKG":1,"CTR":9,"OTH":0,"OWN":1,"TXT":1})
    drawScoreProgression({"exp_dir":r"C:\NLP\PhD\bob\experiments\w20_az_az_fa\\"},[1,2,3,4,4,4,5,6],0)
    pass

if __name__ == '__main__':
    main()
