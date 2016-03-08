# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from copy import deepcopy
from collections import OrderedDict

def getDictOfTestingMethods(methods):
    """
        Make a simple dictionary of {method_10:{method details}}

        new: prepare for using a single index with fields for the parameters
    """
    res=OrderedDict()
    for method in methods:
        for parameter in methods[method]["parameters"]:
            if methods[method]["type"] in ["standard_multi", "inlink_context"]:
                addon="_"+str(parameter)
                indexName=method+addon
                res[indexName]=deepcopy(methods[method])
                res[indexName]["method"]=method
                res[indexName]["parameter"]=parameter
                res[indexName]["index_filename"]=methods[method].get("index",indexName)+addon
                res[indexName]["runtime_parameters"]=methods[method]["runtime_parameters"]
            elif methods[method]["type"] in ["ilc_mashup"]:
                for ilc_parameter in methods[method]["ilc_parameters"]:
                    addon="_"+str(parameter)+"_"+str(ilc_parameter)
                    indexName=method+addon
                    res[indexName]=deepcopy(methods[method])
                    res[indexName]["method"]=method
                    res[indexName]["parameter"]=parameter
                    res[indexName]["ilc_parameter"]=ilc_parameter
                    res[indexName]["index_filename"]=methods[method].get("index",indexName)+addon
                    res[indexName]["runtime_parameters"]=methods[method]["runtime_parameters"]
            elif methods[method]["type"] in ["annotated_boost"]:
                for runtime_parameter_name in methods[method]["runtime_parameters"]:
                    indexName=method+"_"+str(parameter)+"_"+runtime_parameter_name
                    res[indexName]=deepcopy(methods[method])
                    res[indexName]["method"]=method
                    res[indexName]["parameter"]=parameter
                    res[indexName]["runtime_parameter_name"]=runtime_parameter_name
                    res[indexName]["runtime_parameters"]= methods[method]["runtime_parameters"][runtime_parameter_name]
                    res[indexName]["index_filename"]=methods[method].get("index",indexName)+"_"+str(parameter)
            elif methods[method]["type"] in ["ilc_annotated_boost"]:
                for ilc_parameter in methods[method]["ilc_parameters"]:
                    for runtime_parameter_name in methods[method]["runtime_parameters"]:
                        indexName=method+"_"+str(ilc_parameter)+"_"+runtime_parameter_name
                        res[indexName]=deepcopy(methods[method])
                        res[indexName]["method"]=method
                        res[indexName]["parameter"]=parameter
                        res[indexName]["runtime_parameter_name"]=runtime_parameter_name
                        res[indexName]["runtime_parameters"]=methods[method]["runtime_parameters"][runtime_parameter_name]
                        res[indexName]["ilc_parameter"]=ilc_parameter
                        res[indexName]["index_filename"]=methods[method].get("index",indexName)+"_"+str(parameter)+"_"+str(ilc_parameter)
    return res


def main():
    pass

if __name__ == '__main__':
    main()
