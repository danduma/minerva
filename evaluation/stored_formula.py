# <purpose>
#
# Copyright:   (c) Daniel Duma 2015
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

class StoredFormula:
    """
        Stores a Lucene explanation and makes it easy to set weights on the
        formula post-hoc and recompute
    """
    def __init__(self, formula=None):
        if formula:
            self.formula=formula
        else:
            self.formula={}
        self.round_to_decimal_places=4

    def __getitem__(self, key): return self.formula[key]

    def __setitem__(self, key, item): self.formula[key] = item

    def truncate(self, f, n):
        '''Truncates/pads a float f to n decimal places without rounding'''
        s = '{}'.format(f)
        if 'e' in s or 'E' in s:
            return '{0:.{1}f}'.format(f, n)
        i, p, d = s.partition('.')
        return '.'.join([i, (d+'0'*n)[:n]])

    def fromLuceneExplanation(self,explanation):
        """
            Loads the formula from a Lucene explanation
        """
        original_value=explanation.getValue()

        if not explanation.isMatch():
            self.formula={"coord":0,"matches":[]}
            return

        details=explanation.getDetails()
        self.formula["matches"]=[]

        if "weight(" in details[0].getDescription(): # if coord == 1 it is not reported
            matches=details
            self.formula["coord"]=1
        else:
            matches=details[0].getDetails()
            self.formula["coord"]=details[1].getValue()

        for match in matches:
            desc=match.getDescription()
            field=re.match(r"weight\((.*?)\:",desc,re.IGNORECASE)
            # using dicts
##            newMatch={"field":field.group(1)}
##            elem=match.getDetails()[0]
##            if "fieldWeight" in elem.getDescription():
##                # if the queryWeight is 1, .explain() will not report it
##                newMatch["qw"]=1.0
##                newMatch["fw"]=elem.getValue()
##            else:
##                elements=elem.getDetails()
##                newMatch["qw"]=elements[0].getValue()
##                newMatch["fw"]=elements[1].getValue()
            # using namedtuple
##            newMatch=namedtuple("retrieval_result",["field","qw","fw"])
##            newMatch.field=str(field.group(1))
##            elem=match.getDetails()[0]
##            if "fieldWeight" in elem.getDescription():
##                # if the queryWeight is 1, .explain() will not report it
##                newMatch.qw=1.0
##                newMatch.fw=elem.getValue()
##            else:
##                elements=elem.getDetails()
##                newMatch.qw=elements[0].getValue()
##                newMatch.fw=elements[1].getValue()

            # using tuple
            field_name=str(field.group(1))
            elem=match.getDetails()[0]
            if "fieldWeight" in elem.getDescription():
                # if the queryWeight is 1, .explain() will not report it
                newMatch=(field_name,1.0,elem.getValue())
            else:
                elements=elem.getDetails()
                newMatch=(field_name,elements[0].getValue(),elements[1].getValue())
            self.formula["matches"].append(newMatch)

        # just checking
##        original_value=self.truncate(original_value,self.round_to_decimal_places)
##        computed_value=self.truncate(self.computeScore(defaultdict(lambda:1),self.round_to_decimal_places))
##        assert(computed_value == original_value)

    def fromElasticExplanation(self,explanation):
        """
            Loads the formula from a Lucene explanation
        """
        original_value=explanation.getValue()

        if not explanation.isMatch():
            self.formula={"coord":0,"matches":[]}
            return

        details=explanation.getDetails()
        self.formula["matches"]=[]

        if "weight(" in details[0].getDescription(): # if coord == 1 it is not reported
            matches=details
            self.formula["coord"]=1
        else:
            matches=details[0].getDetails()
            self.formula["coord"]=details[1].getValue()

        for match in matches:
            desc=match.getDescription()
            field=re.match(r"weight\((.*?)\:",desc,re.IGNORECASE)
            # using dicts
##            newMatch={"field":field.group(1)}
##            elem=match.getDetails()[0]
##            if "fieldWeight" in elem.getDescription():
##                # if the queryWeight is 1, .explain() will not report it
##                newMatch["qw"]=1.0
##                newMatch["fw"]=elem.getValue()
##            else:
##                elements=elem.getDetails()
##                newMatch["qw"]=elements[0].getValue()
##                newMatch["fw"]=elements[1].getValue()
            # using namedtuple
##            newMatch=namedtuple("retrieval_result",["field","qw","fw"])
##            newMatch.field=str(field.group(1))
##            elem=match.getDetails()[0]
##            if "fieldWeight" in elem.getDescription():
##                # if the queryWeight is 1, .explain() will not report it
##                newMatch.qw=1.0
##                newMatch.fw=elem.getValue()
##            else:
##                elements=elem.getDetails()
##                newMatch.qw=elements[0].getValue()
##                newMatch.fw=elements[1].getValue()

            # using tuple
            field_name=str(field.group(1))
            elem=match.getDetails()[0]
            if "fieldWeight" in elem.getDescription():
                # if the queryWeight is 1, .explain() will not report it
                newMatch=(field_name,1.0,elem.getValue())
            else:
                elements=elem.getDetails()
                newMatch=(field_name,elements[0].getValue(),elements[1].getValue())
            self.formula["matches"].append(newMatch)

        # just checking
##        original_value=self.truncate(original_value,self.round_to_decimal_places)
##        computed_value=self.truncate(self.computeScore(defaultdict(lambda:1),self.round_to_decimal_places))
##        assert(computed_value == original_value)



    def computeScore(self,parameters):
        """
            Simple recomputation of a Lucene explain formula using the values in
            [parameters] as per-field query weights
        """
        match_sum=0.0
        for match in self.formula["matches"]:
##            match_sum+=(match["qw"]*parameters[match["field"]])*match["fw"]
            match_sum+=(match[1]*parameters[match[0]])*match[2]
        total=match_sum*self.formula["coord"]
##        total=self.truncate(total, self.round_to_decimal_places) # x digits of precision
        return total


def main():
    pass

if __name__ == '__main__':
    main()
