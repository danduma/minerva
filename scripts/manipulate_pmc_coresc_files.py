# <purpose>
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

import glob, os

def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        os.rename(pathAndFilename,
                  pathAndFilename.replace(".xml.gz.gz",""))

def tar_all(dir, pattern):
    dir=dir.strip("\\")
    from subprocess import call
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if os.path.isdir(pathAndFilename) and title != "output":
            call([r"C:\cygwin\bin\7za.exe", "a", pathAndFilename+".tar", pathAndFilename])
            call([r"C:\cygwin\bin\7za.exe", "a", pathAndFilename+".gz", pathAndFilename+".tar"])
            call(["cmd", "/c", "del", pathAndFilename+".tar"])

    call(["cmd", "/c", "ren",dir+"\*.gz",dir+"\*.tar.gz"])

def main():
##    rename(r"G:\NLP\PhD\pmc_coresc\inputXML","*.xml.gz.gz","")
    tar_all(r"G:\NLP\PhD\pmc_coresc\inputXML", "*")
    pass

if __name__ == '__main__':
    main()

