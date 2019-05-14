import json


def convert_arxiv_categories(filename):
    f=open(filename, "r")
    categories={}
    for line in f:
        if len(line.strip())==0:
            continue
        if not line.startswith("<li"):
            identifier, title = line.split(":")
            categories[identifier] = title.strip()


    print(json.dumps(categories, indent=3))



def main():
    convert_arxiv_categories("/Users/masterman/Dropbox/PhD/minerva3/minerva/scyence/arxiv_categories.txt")

if __name__ == '__main__':
    main()