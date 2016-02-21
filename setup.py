from setuptools import setup, find_packages

setup(
    name = "minerva",
    version = "0.1",
    packages = find_packages(),

    #install requirements
    install_requires = ["Flask>=0.9",
            "nltk>=2.0.4",
            "pybtex>=0.19",
            "citeproc-py",
            "elasticsearch",
            "elasticsearch-dsl",
            ],
    #author details
    author = "Daniel Duma",
    author_email = "danielduma@gmail.com",
    description = "A framework for context-based citation recommendation experiments",
    url=""
)

