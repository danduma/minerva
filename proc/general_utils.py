# A bunch of file and text functions to make life easier
#
# Copyright:   (c) Daniel Duma 2013
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

from __future__ import absolute_import
from __future__ import print_function
import os, re, codecs, datetime, random, sys, unicodedata, math
import six
from six.moves import range

try:
    import six.moves.cPickle
except:
    import pickle


class AttributeDict(dict):
    """
        Access all keys in a dictionary as attributes of the dictionary
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def levenshtein(seq1, seq2):
    """
        Simple string edit distance function from Wikibooks
        http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]


# ------------------------------------------------------------------------------
#   File and directory functions
# ------------------------------------------------------------------------------

def ensureTrailingBackslash(path):
    return os.path.normpath(path) + os.sep


def exists(filename):
    """
        Checks that a file exists on disk
    """
    return os.path.isfile(filename)


def ensureDirExists(dir):
    # type: (basestring) -> None
    """
        Makes sure directory exists. If not, it creates it.
    """
    dir = os.path.normpath(dir)
    if not os.path.isdir(dir):
        try:
            os.makedirs(dir)
        except:
            print(("Failed to create directory %s" % dir))


def deleteFiles(path, file_list):
    """
        WARNING! For cleanup only. Deletes the files on the list in the directory specified
    """
    for fn in file_list:
        if os.path.exists(path + fn):
            os.remove(path + fn)


def getFileName(filename):
    """
        Returns only the file name of a the tail of a path
    """
    fileName, fileExtension = os.path.splitext(os.path.basename(filename))
    return fileName


def getFileDir(filename):
    """
        Returns only the path component of a filename, without os.sep
    """
    return os.path.dirname(filename)


def getSafeFilename(full_filename):
    """
        Checks if file exists. If it does, it adds an int _01 to the end,
        incrementing until it finds an int that hasn't been taken yet
        process has a lock on it, so a new name is created and returned
    """
    new_number = 1
    format_digits = 2

    while exists(full_filename):
        path, filename = os.path.split(full_filename)
        file_name, ext = os.path.splitext(os.path.basename(filename))
        match = re.search(r"(.+)_(\d+)", file_name)

        if match:
            new_number = int(match.group(2)) + 1
            if int(math.log(new_number))  > format_digits:
                format_digits += 1
            file_name = "{}{}".format(match.group(1), "_%02d" % new_number)
        else:
            file_name = "{}{}".format(file_name, "_%02d" % new_number)

        full_filename = os.path.join(path, file_name + ext)
    return full_filename


def getTimestampedFilename(full_path, timestamp):
    """
    Makes a file with date and time as prefix

    :param full_path: full path of the file
    :param timestamp: timestamp to add to the front of the file name
    :return: timestamped full path
    """
    prefix = "{}-{}-{}_{}-{:02d}_".format(timestamp.day,
                                          timestamp.month,
                                          timestamp.year,
                                          timestamp.hour,
                                          timestamp.minute)
    path, full_path = os.path.split(full_path)
    file_name, ext = os.path.splitext(os.path.basename(full_path))
    file_name = prefix + file_name
    return os.path.join(path, file_name + ext)


# ------------------------------------------------------------------------------
#   Loading and saving functions
# ------------------------------------------------------------------------------

def readFileText(filename, encoding="utf-8"):
    """
        Same as loadFileText below
    """
    return loadFileText(filename, encoding)


def loadFileText(filename, encoding="utf-8"):
    """
        Opens a file using codecs, reads full contents into a string, returns string.
    """
    with codecs.open(filename, "rb", encoding, errors="replace") as f:
        lines = f.readlines()
        text = u"".join(lines)
        return text
    return None


def writeFileText(text, filename):
    """
        Write a buffer of text to a file
    """
    f2 = codecs.open(filename, "w", "utf-8", errors="replace")
    f2.write(text)
    f2.close()


def normalizeUnicode(text):
    """
        Tries to normalize crazy unicode into more "normal" text
    """
    text = unicodedata.normalize('NFKD', text)
    return text


def saveFileList(filelist, filename):
    """
        Write a list of strings to a file
    """
    f2 = codecs.open(filename, "w", "utf-8", errors="replace")
    for line in filelist:
        f2.write(line + "\n")
    f2.close()


def loadFileList(filename):
    """
        Returns a list of lines from a file or None if the file does not exist
    """
    if os.path.exists(filename):
        f2 = codecs.open(filename, "r", "utf-8", errors="replace")
        res = [line.strip() for line in f2.readlines()]
        f2.close()

        return res
    return None

def loadListFromTxtFile(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        guids = []
        for line in lines:
            line = line.strip()
            if line:
                guids.append(line)
    return guids


def saveListToTxtFile(guids, filename):
    with open(filename, "w") as f:
        for guid in guids:
            guid = guid.strip()
            f.write(guid + "\n")
    return guids

def writeTuplesToCSV(columns, tuples, filename):
    """
        Rakes a list of columns and a lsit of tuples, assumes each tuple has the required number of elements
    """
    try:
        f = codecs.open(filename, "wb", "utf-8", errors="replace")
    except:
        f = codecs.open(filename + str(random.randint(10, 100)), "wb", "utf-8", errors="replace")

    line = u"".join([c + u"\t" for c in columns])
    line = line.strip(u"\t")
    line += u"\n"
    f.write(line)

    pattern = u"".join([u"%s\t" for c in columns])
    pattern = pattern.strip()
    pattern += u"\n"

    for l in tuples:
        try:
            line = pattern % l
            f.write(line)
        except:
            print(("error writing: " + l.__repr__()))

    f.close()


def writeDictToCSV(columns, data, filename):
    """
        Makes a CSV from the values of a dict, given a list of keys
        columns: list of keys
        data: dictionary
    """
    f = codecs.open(filename, "wb", "utf-8", errors="replace")

    line = u"".join([c + u"\t" for c in columns])
    line = line.strip(u"\t")
    line += u"\n"
    f.write(line)

    pattern = u"".join([u"%s\t" for c in columns])
    pattern = pattern.strip()
    pattern += u"\n"

    for data_point in data:
        line = ""
        try:
            for column_name in columns:
                line += str(data_point.get(column_name, "")) + "\t"
            line = line.strip("\t")
            line += "\n"
            f.write(line)
        except:
            print(("error writing: " + sys.exc_info()[:2]))

    f.close()


def getListAnyway(item):
    """
        If the item is not a list, it returns a list with the item as only element
    """
    if isinstance(item, list):
        res = item
    else:
        res = [item]
    return res


def most_common(L):
    """
        returns the most common element in a list

        from http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
    """
    groups = itertools.groupby(sorted(L))

    def _auxfun(packed):
        (item, iterable) = packed
        return len(list(iterable)), -L.index(item)

    return max(groups, key=_auxfun)[0]


def cs(s):
    """
        Clears tabs and such to print value as CSV
    """
    s = s.replace("\t", " ").replace("\n", " ")
    return s


def savePickle(filename, what):
    """
        Pickle an object
    """
    #    f=codecs.open(filename,"wb","utf-8", errors="replace")
    f = open(filename, "wb")
    six.moves.cPickle.dump(index, f)
    f.close()


def loadPickle(filename):
    """
        Load the saved (pickled) object from file
    """

    #    f=codecs.open(filename,"rb","utf-8", errors="replace")
    f = open(filename, "rb")
    index = six.moves.cPickle.load(f)
    return index


def copyDictExceptKeys(dict_to_copy, except_keys):
    """
        Returns a new dict with pointers to the keys that aren't skipped
    """
    newdict = {}
    for key in dict_to_copy:
        if key not in except_keys:
            newdict[key] = dict_to_copy[key]
    return newdict


def removeSymbols(text):
    """
        Remove all symbols from a string, including all kinds of brackets and quotation marks.
    """
    return re.sub(r"[\"\#\$\%\&\\\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\¿\!\¡\@\[\]\^\_\`\{\|\}\~]", " ", text)


def normalizeTitle(title):
    """
        Returns a "hashed" title for easy matching
    """
    title = title.lower()
    title = title.replace("-  ", "").replace("- ", "")
    title = removeSymbols(title)
    title = re.sub(r"\s+", " ", title)
    title = title.strip()
    title = title[:200]
    return title


def pathSelect(root, path, recursive=True):
    """
        An XPath-inspired easy way to specify an XML element

        Works on top of BeautifulSoup

        :param path: path to node we seek
        :type path: string or list
    """
    if isinstance(path, six.string_types):
        path = path.lower().split("/")

    for element in path:
        optional = False
        if element.endswith("?"):
            element = element[:-1]
            optional = True

        next = root.find(element, recursive=recursive)
        if next:
            root = next
        else:
            if optional:
                continue
            else:
                return None
    return root


def reportTimeLeft(current_file, num_files, time0, msg="", start_at=0):
    """
        Prints a progress report: file#/total_files, time left
    """
    current_file -= start_at
    now = datetime.datetime.now()
    diff = (now - time0)
    each = diff.total_seconds() / float(current_file)
    total = each * (num_files - current_file)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if msg != "": msg += ", "
    print(("%d/%d: %s %02d:%02d:%02d left" %
           (current_file, num_files, msg, int(hours), int(minutes), int(seconds))))


def safe_unicode(obj, *args):
    """ return the unicode representation of obj """
    try:
        return six.text_type(obj, *args)
    except UnicodeDecodeError:
        # obj is byte string
        ascii_text = str(obj).encode('string_escape')
        return six.text_type(ascii_text)


def safe_str(obj):
    """ return the byte string representation of obj """
    try:
        return str(obj)
    except UnicodeEncodeError:
        # obj is unicode
        return six.text_type(obj).encode('unicode_escape')


def getRootDir(subdir=None):
    """
    Returns the root directory for data storage (index, json files, etc.) based on where
    the script is being run

    :return:
    """
    dir_options = ["g:\\nlp\\phd", "c:\\nlp\\phd", "/Users/masterman/NLP/PhD", "/home/iomasterman_gmail_com", "/home/ubuntu/PhD"]
    for option in dir_options:
        if os.path.isdir(option):
            if not subdir:
                return option
            else:
                return os.path.join(option, subdir)


def main():
    ##    files=loadFileText(r"C:\NLP\PhD\bob\fileDB\db\files_to_ignore.txt").lower().split()
    ##    deleteFiles(r"C:\NLP\PhD\bob\inputXML"+os.sep,files)
    print(getSafeFilename("/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac/results.csv"))
    print(getTimestampedFilename("/Users/masterman/NLP/PhD/aac/experiments/thesis_chapter3_experiment_aac/results.csv",
                                 datetime.datetime.now()))
    pass


if __name__ == '__main__':
    main()


