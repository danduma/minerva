# Cermine interface library. Simple server, client and reader to JSON metadata
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from .server import startStandaloneServer
from .client import CermineClient
from .read_cermine import CermineReader