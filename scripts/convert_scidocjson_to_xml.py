#-------------------------------------------------------------------------------
# Name:        convert_scidocjson_to_xml
# Purpose:      batch convert all SciDocJSON in a folder to a given XML format
#
# Author:      dd
#
# Created:     28/03/2015
# Copyright:   (c) dd 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from minerva.scidoc.xmlformats.export_scixml import saveSciXML
from minerva.scidoc.xmlformats.export_sapientaxml import saveSapientaXML
from minerva.scidoc.xmlformats.export_jats import saveJATS_XML

from minerva.scidoc import SciDoc

from minerva.proc.general_utils import *
import glob, os

def batchConvertSciDocsDirToXML(path,output_dir,output_format="SciXML"):
    """
        Creates file list from directory, calls batchConvertSciDocsToXML()
    """
    files=glob.glob(path)[2:]
    batchConvertSciDocsToXML(files,output_dir,output_format)

def batchConvertSciDocsToXML(file_list,output_dir,output_format="SciXML"):
    """
        Converts files matching path from SciDocJSON to XML
    """
    save_function=None

    if output_format=="SciXML":
        save_function=saveSciXML
    elif output_format=="SapientaXML":
        save_function=saveSapientaXML
    else:
        raise ValueError("Unknown output format:"+output_format)
        return

    output_dir=ensureTrailingBackslash(output_dir)
    for filename in file_list:
        print "Converting",filename
        doc=SciDoc(filename)
        fn=os.path.basename(filename)
        save_function(doc,output_dir+os.path.splitext(fn)[0]+".xml")

def loadFileList(filename,input_dir):
    """
    """
    f=codecs.open(filename,"r",errors="ignore")
    input_dir=ensureTrailingBackslash(input_dir)
    file_list=[input_dir+line.strip("\n").replace(".xml",".json") for line in f]
    f.close()
    return file_list


def main():
##    batchConvertSciDocsDirToXML(r"C:\NLP\PhD\bob\fileDB\jsonDocs\*.*",r"C:\NLP\PhD\bob\converted_scixml","SapientaXML")

    file_list=loadFileList(r"C:\NLP\PhD\bob\daniel_converted_out\daniel_failed.txt",r"C:\NLP\PhD\bob\fileDB\jsonDocs")
##    batchConvertSciDocsDirToXML(r"C:\NLP\PhD\bob\fileDB\jsonDocs\*.*",r"C:\NLP\PhD\bob\converted_scixml","SapientaXML")
    batchConvertSciDocsToXML(file_list,r"C:\NLP\PhD\bob\converted_scixml","SapientaXML")

    pass

if __name__ == '__main__':
    main()
