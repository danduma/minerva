# BasePrebuilder: calls prebuildMulti() or schedules celery task
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import print_function
from __future__ import absolute_import
import sys

from .prebuild_functions import prebuildMulti
from multi.tasks import prebuildBOWTask
from celery import group

import db.corpora as cp
from proc.results_logging import ProgressIndicator


class BasePrebuilder(object):
    """
        Wrapper around the prebuilding functions.
    """

    def __init__(self, use_celery=False):
        """
        """
        self.use_celery = use_celery
        self.exp = {}
        self.options = {}

    def prebuildBOWsForTests(self, exp, options):
        """
            Generates BOWs for each document from its inlinks, stores them in a
            corpus cached file

            :param parameters: list of parameters
            :param maxfiles: max. number of files to process. Simple parameter for debug
            :param overwrite_existing_bows: should BOWs be rebuilt even if existing?

        """
        self.exp = exp
        self.options = options

        maxfiles = options.get("max_files_to_process", sys.maxsize)

        if len(self.exp.get("rhetorical_annotations", [])) > 0:
            print("Loading AZ/CFC classifiers")
            cp.Corpus.loadAnnotators()

        print("Prebuilding BOWs for", min(len(cp.Corpus.ALL_FILES), maxfiles), "files...")
        numfiles = min(len(cp.Corpus.ALL_FILES), maxfiles)

        if self.use_celery:
            print("Queueing tasks...")

            all_tasks = []
            for guid in cp.Corpus.ALL_FILES[:maxfiles]:
                for method_name in self.exp["prebuild_bows"]:
                    # run_annotators=self.exp.get("rhetorical_annotations",[]) if self.exp.get("run_rhetorical_annotators",False) else []

                    all_tasks.append(prebuildBOWTask.s(
                        method_name,
                        self.exp["prebuild_bows"][method_name]["parameters"],
                        self.exp["prebuild_bows"][method_name]["function_name"],
                        guid,
                        self.options["overwrite_existing_bows"],
                        self.exp.get("filter_options_ilc",{}),
                        self.options.get("overwrite_existing_bows")
                        ))

            jobs = group(all_tasks)

            result = jobs.apply_async(queue="prebuild_bows", exchange="prebuild_bows", route_name="prebuild_bows")
            print("Waiting for tasks to complete...")
            result.join()
        else:
            progress = ProgressIndicator(True, numfiles, False)
            for guid in cp.Corpus.ALL_FILES[:maxfiles]:
                for method_name in self.exp["prebuild_bows"]:

                    prebuildMulti(
                        method_name,
                        self.exp["prebuild_bows"][method_name]["parameters"],
                        self.exp["prebuild_bows"][method_name]["function"],
                        None,
                        None,
                        guid,
                        self.options["overwrite_existing_bows"],
                        self.exp.get("filter_options_ilc", {}),
                        force_rebuild=self.options.get("overwrite_existing_bows")
                    )
                progress.showProgressReport("Building BOWs")


def main():
    pass


if __name__ == '__main__':
    main()
