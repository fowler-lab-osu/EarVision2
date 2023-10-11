import re

'''

Example name:

    B2x121E-1_m2

1. First character is year

    B = 2022
    C = 2023
    X = 2018
    Y = 2019

2. x separates male/pollen from female/ear parents. F first, M second

    2 = female parent
    121E-1 = male parent

3. Usually one of the parents will be wildtype or outcross; naming specifics are year dependent; mostly interested in 
    F ears being wildtype and M pollen containing the insertion allele

    2 is the outcross for this example

4. The parent with the insertion allele will have a -, separating familiy and plant number. Each family will have a 
    specific insertion allele associated with it. This data will be year specific

    121E is the family
    1 is the plant number

5. m# indicates male pollination number, if the same plant was used multiple times as a pollen parent. Not super 
    important to allele tracking

    m2 means the pollen from plant 1 from family 121E was used at least twice, this being the second time

6. Some image names will NOT follow these rules. Also @ indicates a self cross


Output columns:
    Year (letter)
    Ear Parent family (number)
    Ear Parent Subfamily (letter, not always present)
    Ear Parent plant (number, not always present)
    Pollen parent family (number)
    Pollen parent subfamily (letter, not always present)
    Pollent parent plant (number, not always present)
    Pollen parent male number/incidence (number, not always present)
'''


class EarName:
    def __init__(self, earName):

        self.yearDict = {
            "A": 2021,
            "B": 2022,
            "C": 2023,
            "X": 2018,
            "Y": 2019,
            "Z": 2020
        }

        #self.bYearFamilyAllele = {}
        #self.__getBYearFamilyData__()

        self.earName = earName

        self.yearLetter = re.findall(r'^[A-Z]', self.earName)[0]
        self.year = self.yearDict[self.yearLetter]
        self.earData, self.pollenData = self.earName[1:].split('x')

        self.earFamily = self.__setFamily__(self.earData)
        self.earSubFamily = self.__setSubFamily__(self.earData)
        self.earPlant = self.__setPlant__(self.earData)

        self.pollenFamily = self.__setFamily__(self.pollenData)
        self.pollenSubFamily = self.__setSubFamily__(self.pollenData)
        self.pollenPlant = self.__setPlant__(self.pollenData)
        self.pollenMaleNum = self.__setMale__(self.pollenData)

        self.earAllele = None
        self.pollenAllele = None
        self.crossType = self.__setCrossType__()

    def __setCrossType__(self):
        if self.pollenFamily == 2:
            return "Ear"
        elif self.earFamily == 2:
            return "Pollen"
        else:
            return "Other"

    def __setFamily__(self, data):
        if re.findall(r'^[0-9]+', data):
            return int(re.findall(r'^[0-9]+', data)[0])
        return None

    def __setSubFamily__(self, data):
        if re.findall(r'[A-Z]', data):
            return re.findall(r'[A-Z]', data)[0]
        return None
    
    def __setPlant__(self, data):
        if re.findall(r'-[0-9]+', data):
            return int(re.findall(r'-[0-9]+', data)[0][1:])
        return None
    
    def __setMale__(self, data):
        if re.findall(r'm[0-9]+', data):
            return re.findall(r'm[0-9]+', data)[0]
        return None
    
    def __csvEarData__(self):
        return f"{self.earName},{self.year},{self.crossType},{self.earFamily},{self.earSubFamily},{self.earPlant},{self.earAllele},{self.pollenFamily},{self.pollenSubFamily},{self.pollenPlant},{self.pollenMaleNum},{self.pollenAllele},"


