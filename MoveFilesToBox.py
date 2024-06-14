import os
import shutil
import pandas as pd

print("Moving Inferences Files to Box")

inferenceResultDir= "FullEar2018-2022_AmbigAdjust"
modelFolderID = "InferenceOutput-Jose_07.18.23_11.24AM-027"

boxDir = "D:\\Box\\Michelle\\SpatialAnalysis_Alleles_80"
fileDataFile = boxDir + "\\AllEarKernelCountData_MichellePaperv2.csv"
fileData  = pd.read_csv(fileDataFile)

print(fileData)

earDir = boxDir+"\\Ear_Crosses"
pollenDir = boxDir+"\\Pollen_Crosses"

os.makedirs(earDir, exist_ok=True)
os.makedirs(pollenDir, exist_ok=True)

'''
for i,row in fileData.iterrows():
    if row['cross_type'] == 'Ear':
        print(fileData.iloc[i:i+1]
        '''

earCrosses = fileData.loc[fileData['cross_type']=='Ear']
pollenCrosses = fileData.loc[fileData['cross_type']=='Pollen']

print(earCrosses)
print(pollenCrosses)

notFoundList = open(boxDir+"\\"+"notFound.txt", "a")

def moveFiles(alleleDF, boxDir):
    alleles = set(alleleDF['allele'].tolist())
    print(alleles)
    #print(len(alleles))  #should be 80
    for a in alleles:
        alleleDir = boxDir+"\\"+a
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
