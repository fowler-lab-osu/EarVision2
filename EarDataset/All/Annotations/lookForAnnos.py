import os

fileList = os.listdir(".")
ambig = []
for xmlFile in fileList:
    if xmlFile.find(".xml") != -1:
        with open(xmlFile, "r") as readingXmlFile:
            for line in readingXmlFile:
                if line.find("ambiguous") != -1:
                    #print(xmlFile)
                    if xmlFile not in ambig:
                        ambig.append(xmlFile)
                    
print(len(ambig))
print(len(fileList))
print(ambig)