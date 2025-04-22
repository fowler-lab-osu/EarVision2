import os
import shutil
import pandas as pd

print("Moving Inferences Files Into Folders Organized by Allele")

'''
Moves files from Inference output to another folder organized into alleles (targetDir)
Note: may have to change modelFolder names so they all have the same name, they end up with different timestamps when inference is run
Requires Info that associates image file names with allele (here AllEarKernelCountData.csv)
When this .csv lists 'E' for 'Fiji_or_Earvision', script moves Earvision inference file.
When it lists 'F', moves Fiji handcount file.

'''

inferenceResultDir= "FullEar2018-2022_AmbigAdjust"
modelFolderID = "InferenceOutput-Jose_07.18.23_11.24AM-027"

targetDir = "D:\\target dir goes here"
fileDataFile = "./AllEarKernelCountData.csv"
fileData  = pd.read_csv(fileDataFile)

print(fileData)


earDir = targetDir+"\\Ear_Crosses"
pollenDir = targetDir+"\\Pollen_Crosses"

os.makedirs(earDir, exist_ok=True)
os.makedirs(pollenDir, exist_ok=True)

earCrosses = fileData.loc[fileData['cross_type']=='Ear']
pollenCrosses = fileData.loc[fileData['cross_type']=='Pollen']

print(earCrosses)
print(pollenCrosses)

notFoundList = open(targetDir+"\\"+"notFound.txt", "a")

def moveFiles(alleleDF, targetDir):
    alleles = set(alleleDF['allele'].tolist())
    print(alleles)
    for a in alleles:
        alleleDir = targetDir+"\\"+a
        os.makedirs(alleleDir, exist_ok=True)
        earIDs = alleleDF.loc[alleleDF['allele'] == a]
        fijiEarsDF = earIDs.loc[earIDs['Fiji_or_Earvision'] == 'F']
        earvisionEarsDF = earIDs.loc[earIDs['Fiji_or_Earvision'] == 'E']

        fijiEars = fijiEarsDF['name'].tolist()
        earvisionEars = earvisionEarsDF['name'].tolist()

        for f in fijiEars:
            fileName = f + ".xml"
            year = f[0]+'year'
            try:
                shutil.copy2(inferenceResultDir+"\\"+year+"\\"+fileName, alleleDir)
            except:
                print("Could not find: ", inferenceResultDir+"\\"+year+"\\"+fileName)
                notFoundList.write(fileName + "\n")

        for e in earvisionEars:
            fileName = e + "_inference.xml"
            year = e[0]+'year'
            try:
                shutil.copy2(inferenceResultDir+"\\"+year + "\\"+ modelFolderID+ "\\" +fileName, alleleDir)
            except:
                print("Could not find: ", inferenceResultDir+"\\"+year + "\\"+ modelFolderID+ "\\" +fileName)
                notFoundList.write(fileName + "\n")
       
moveFiles(earCrosses, earDir)
moveFiles(pollenCrosses, pollenDir)

notFoundList.close()
