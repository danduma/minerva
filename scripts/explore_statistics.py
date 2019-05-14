import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
plt.rcParams["figure.figsize"] = (8, 5)

AZ_ZONES = ['AIM', 'BAS', 'BKG', 'CTR', 'OTH', 'OWN', 'TXT']
CORESC_LIST = ["Hyp", "Mot", "Bac", "Goa", "Obj", "Met", "Exp", "Mod", "Obs", "Res", "Con"]
CORESC_LIST = [x.lower() for x in CORESC_LIST]

CORESC_COLS = ["#FFFF00", "#FF0000", "#0000FF", "#FFC0CB", "#90EE90", "#A52A2A", "#008080", "#FF00FF", "#40E0D0",
               "#90EE90", "#008000"]

AZ_COLS = ["#FFFF59", "#FFBE49", "#49F7FF", "#FF79F8", "#F87571", "#BEFF59", "#BEC7FF"]

# df_pmc = pd.read_csv("/Users/masterman/NLP/PhD/pmc_coresc/output/corpus_metadata.csv")
df_aac = pd.read_csv("/Users/masterman/NLP/PhD/aac/output/corpus_metadata.csv")


def get_annotation_stats(data, annotation="az"):
    if annotation == "az":
        zones = AZ_ZONES
    elif annotation == "coresc":
        zones = CORESC_LIST

    annot = data[zones]
    annot2 = pd.DataFrame()
    for zone in zones:
        annot2["has_" + zone] = np.where(annot[zone] > 0, 1, 0)

    sums = pd.DataFrame(annot.sum())
    sums.columns = ["Total"]
    total_sentences = float(sums.sum())
    sums["Ratio"] = sums["Total"] / total_sentences
    sums["Total"] = sums["Total"].round(0)
    sums.loc["Total"] = [total_sentences, 1.0]
    return sums, annot2


def main():
    annot, annot2 = get_annotation_stats(df_aac, "az")


if __name__ == '__main__':
    main()
