# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function
import json

##from proc.nlp_functions import CORESC_LIST
import pandas as pd
from six.moves import range

CORESC_LIST=["Bac","Con","Exp","Goa","Hyp", "Met","Mod","Mot","Obj","Obs","Res"]
AZ_ZONES_LIST=["AIM","BAS","BKG","CTR","OTH","OWN","TXT"]

def make_stacks(stacks):
    """
    """
    lines=[]
    for index, stack in enumerate(stacks):
        lines.append("sankey.stack(%d,%s);" % (index, json.dumps(stack,indent=2)))

    return "\n".join(lines)

def make_raw_data(raw_data):
    """
    """
    return "var raw_data=%s;" % json.dumps(raw_data)

def generate_sankey(input_file, zones=CORESC_LIST):
    """
    """
    raw_data=[]
    stacks=[[]]
    df=pd.DataFrame.from_csv(input_file)

    for index in range(df.shape[0]):
        cit_class=df.iloc[index].name+"-"
        stacks[0].append(cit_class)
        for zone in zones:
            val=df.iloc[index][zone]*50
            if val > 0:
                raw_data.append([cit_class,val,zone])

    stacks.append(zones)

    print(make_raw_data(raw_data))
    print(make_stacks(stacks))

def main():
##    generate_sankey(r"C:\Users\dd\Documents\Dropbox\PhD\WOSP16\sankey_diagram\sankey_ilc.csv")
##    generate_sankey(r"C:\Users\dd\Documents\Dropbox\PhD\WOSP16\sankey_diagram\sankey_full_text.csv")
    generate_sankey(r"C:\Users\Masterman\Dropbox\PhD\3rd year review\sankey_diagram\sankey_az_full_text.csv", zones=AZ_ZONES_LIST)
    pass

if __name__ == '__main__':
    main()
