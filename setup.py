from __future__ import absolute_import
from setuptools import setup, find_packages

setup(
    name = "minerva",
    version = "0.2",
    packages = find_packages(),

    #install requirements
    install_requires = [
            "beautifulsoup",
            "celery",
            "citeproc-py",
            "elasticsearch",
            "elasticsearch-dsl",
            "flask>=0.9",
            "flask-compress",
            "nltk>=2.0.4",
            "pandas",
            "pybtex>=0.19",
            "requests",
            ],
    #author details
    author = "Daniel Duma",
    author_email = "danielduma@gmail.com",
    description = "A framework for context-based citation recommendation experiments",
    url=""
)

