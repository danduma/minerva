# XML format readers/writers
#
# Copyright (C) 2014 Daniel Duma
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT


"""
Scholarly XML readers.  The modules in this package provide functions
that can be used to read and write XML files in a variety of formats to and
from an in-memory SciDoc instance.


Supported Formats
=================

- JATS/NLM XML: as defined at http://http://jats.nlm.nih.gov/
    The purpose is to read PubMed Central .nxml files.
    - Reader: JATSXMLReader
    - Writer: not implemented yet
- SciXML: reads Sapienta XML files
    - Reader: SciXMLReader
    - Writer: not implemented yet

"""

##from base_classes import *
##from read_jatsxml import JATSXMLReader
##from read_paperxml import *