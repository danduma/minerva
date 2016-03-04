# <purpose>
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function
import sys

from prebuild_functions import prebuildMulti
from minerva.squad.tasks import prebuildBOWTask

class BasePrebuilder(object):
    """
        Wrapper around the prebuilding functions.
    """
    def __init__(self, use_celery=False):
        """
        """
        self.use_celery=use_celery

    def prebuildBOWsForTests(self, prebuild_list, exp, options):
        """
            Generates BOWs for each document from its inlinks, stores them in a
            corpus cached file

            :param parameters: list of parameters
            :param maxfiles: max. number of files to process. Simple parameter for debug
            :param force_prebuild: should BOWs be rebuilt even if existing?

        """

        maxfiles=options.get("max_files_to_process",sys.maxint)

        if len(self.exp.get("rhetorical_annotations",[]) > 0):
            print("Loading AZ/CFC classifiers")
            cp.Corpus.loadAnnotators()

        print("Prebuilding BOWs for", min(len(cp.Corpus.ALL_FILES),maxfiles), "files...")
        numfiles=min(len(cp.Corpus.ALL_FILES),maxfiles)

        if self.use_celery:
            print("Queueing tasks...")
            tasks=[]
            for guid in cp.Corpus.ALL_FILES[:maxfiles]:
                for entry in self.exp["prebuild_bows"]:
                    if self.use_celery:
                        tasks.append(prebuildBOWTask.apply_async(args=[
                            entry,
                            self.exp["prebuild_bows"][entry]["parameters"],
                            self.exp["prebuild_bows"][entry]["function"],
                            guid,
                            doc,
                            doctext,
                            self.options["force_prebuild"],
                            self.exp.get("rhetorical_annotations",[])],
                            queue="build_bows"))

        else:
            progress=ProgressIndicator(True, numfiles, False)
            for guid in cp.Corpus.ALL_FILES[:maxfiles]:
                for entry in self.exp["prebuild_bows"]:
                    prebuildMulti(
                                  entry,
                                  self.exp["prebuild_bows"][entry]["parameters"],
                                  self.exp["prebuild_bows"][entry]["function"],
                                  guid,
                                  doc,
                                  doctext,
                                  self.options["force_prebuild"],
                                  self.exp.get("rhetorical_annotations",[])
                                  )
                progress.showProgressReport("Building BOWs")


def main():
    pass

if __name__ == '__main__':
    main()
