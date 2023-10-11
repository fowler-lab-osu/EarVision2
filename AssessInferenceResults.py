'''
EarVision 2.0:
Assess Inference Results

This script creates graphs from Inference Output data.
'''

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from Infer import Infer
from tkinter import Tk
from tkinter import *
import tkinter.filedialog as filedialog
import os

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress




actTrans, actFluor, actNonFluor = {}, {}, {}


def makeInteractive():
    homeDirec = os.getcwd()

    root = Tk()
    root.withdraw()
    inferenceDirectoryPath = filedialog.askdirectory(initialdir=homeDirec+"/Inference")

    print("Opening " + inferenceDirectoryPath)

    root.destroy()
    
def graphDriver(filename, title, filter):
    xt, yt = makeListsForGraph(filename, "ActualTransmission", "PredictedTransmission", filter)
    print(f"# pred trans: {len(xt)}")
    print(f"# act trans: {len(yt)}")
    makeGraph(xt, yt, "pred", "act", title + " Transmission", 0, 100, 0, 100)
    #xf, yf = makeListsForGraph(filename, "ActualFluor", "PredictedFluor", filter)
    #makeGraph(xf, yf, "ActualFluor", "PredictedFluor", title + " Fluorescent Kernels", 0, 500, 0, 500)
    #xnf, ynf = makeListsForGraph(filename, "ActualNonFluor", "PredictedNonFluor", filter)
    #makeGraph(xf, yf, "ActualNonFluor", "PredictedNonFluor", title + " NonFluorescent Kernels", 0, 500, 0, 500)



def makeGraph(x, y, xLabel, yLabel, title, xmin, xmax, ymin, ymax):
    #x, y = makeListsForGraph(filename, xLabel, yLabel, filter)
    #print(y)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.scatter(x, y)
    plt.axis('square')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    #plt.axis('equal')
    plt.show()


def findProblemImages(filenames):
    ambigEars = {}
    for filename in filenames:
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            if row["TrainingSet"] != True:
                if row["EarName"] not in ambigEars:
                    ambigEars[row["EarName"]] = []
                ambigEars[row["EarName"]].append(row["AmbiguousKernels"])

    problemEars = {}
    for ear in ambigEars:
        for ambigs in ambigEars[ear]:
            if ambigs >= 8:
                if ear not in problemEars:
                    problemEars[ear] = []
                problemEars[ear] = ambigEars[ear]

    print(problemEars.keys())


def makeListsForGraph(filename, colNameX, colNameY, filter):
    xList, yList = [], []
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        if row["TrainingSet"] != True:
            if row["ActualTransmission"] > -1:
                if filter==0:
                    #if row["PredictedTotal"] <= 100 or row["AmbiguousKernels"] >= 8:
                    #if row["PredictedTotal"] > 60 and row["AmbiguousKernels"] <= 7:
                        #print(f"{row['EarName']}\t{row['PredictedTotal']}\t{row['AmbiguousKernels']}")
                    xList.append(row[colNameX])
                    yList.append(row[colNameY])
                elif filter==1:
                    if row["PredictedTotal"] > 100 and row["AmbiguousKernels"] < 15:
                    #print(f"{row['EarName']}\t{row['PredictedTotal']}\t{row['AmbiguousKernels']}")
                        xList.append(row[colNameX])
                        yList.append(row[colNameY])
                elif filter==2:
                    if row["AmbiguousKernelPercentage"] < 0.2 and row["PredictedTotal"] > 100:
                    #print(f"{row['EarName']}\t{row['PredictedTotal']}\t{row['AmbiguousKernels']}")
                        xList.append(row[colNameX])
                        yList.append(row[colNameY])
                        
    return xList, yList

def numImgs(df):
    numImgsUnder200 = 0
    numImgsUnder150 = 0
    numImgsTotal = 0
    for index, row in df.iterrows():
        if row["TrainingSet"] != True:
            numImgsTotal += 1
            if row["PredictedTotal"] < 200:
                numImgsUnder200 += 1
                #numImgsOver150 += 1
            if row["PredictedTotal"] < 150:
                numImgsUnder150 += 1
    print(numImgsUnder200)
    print(numImgsUnder150)
    print(numImgsTotal)


def newInfData(df):
    predFluor, predNonFluor, predTrans = {}, {}, {}
    for index, row in df.iterrows():
        if row["TrainingSet"] != True:
            if row["ActualTransmission"] > -1:
                earName = row["EarName"]
                if earName not in predTrans:
                    predTrans[earName] = 0
                predTrans[earName] = row["PredictedTransmission"]
                if earName not in predFluor:
                    predFluor[earName] = 0
                predFluor[earName] = row["PredictedFluor"]
                if earName not in predNonFluor:
                    predNonFluor[earName] = 0
                predNonFluor[earName] = row["PredictedNonFluor"]

                if earName not in actTrans:
                    actTrans[earName] = row["ActualTransmission"]
                if actTrans[earName] != row["ActualTransmission"]:
                    print(f"issue with {earName} actual transmission")
                if earName not in actFluor:
                    actFluor[earName] = row["ActualFluor"]
                if actFluor[earName] != row["ActualFluor"]:
                    print(f"issue with {earName} actual fluorescent kernels")
                if earName not in actNonFluor:
                    actNonFluor[earName] = row["ActualNonFluor"]
                if actNonFluor[earName] != row["ActualNonFluor"]:
                    print(f"issue with {earName} actual nonfluor kernels")
    return predFluor, predNonFluor, predTrans
  

def makeWarmanCompGraphs(df, modelName, warmanList):
    # Create dicts where key is earname and val is number (numFluor, numNonFluor, trans)
    # This fuction also fills global structs for actual values
    predFluor, predNonFluor, predTrans = newInfData(df)
    yNewModPredFluor, yNewModPredNonFluor, yNewModPredTrans = [], [], []
    xActFluor, xActNonFluor, xActTrans = [], [], []
    # We only want to look at ears that were used in Warman's tests
    for ear in warmanList:
        # Make sure ear has actual vals associated with it (should always be true) and that it has model predicted values
        # in it (should also always be true)
        if ear in actFluor and ear in predFluor:
            # Make lists to use in graph, assoc. values added in the same order
            yNewModPredFluor.append(predFluor[ear])
            yNewModPredNonFluor.append(predNonFluor[ear])
            yNewModPredTrans.append(predTrans[ear])
            
            xActFluor.append(actFluor[ear])
            xActNonFluor.append(actNonFluor[ear])
            xActTrans.append(actTrans[ear])

    # Graph that stuff
    '''
    print(modelName)
    print(f"maxActFluor:{max(xActFluor)}\tmaxNewFluor:{max(yNewModPredFluor)}")
    print(f"maxActNonFluor:{max(xActNonFluor)}\tmaxNewNonFluor:{max(yNewModPredNonFluor)}")
    print(f"maxActTrans:{max(xActTrans)}\tmaxNewTrans:{max(yNewModPredTrans)}")
    print()
    '''
    makeGraph(xActFluor, yNewModPredFluor, "Actual Number of Fluorescent Kernels", "Predicted Number of Fluorescent Kernels", modelName + " Fluor", 0, 500, 0, 500)
    makeGraph(xActNonFluor, yNewModPredNonFluor, "Actual Number of NonFluorescent Kernels", "Predicted Number of NonFluorescent Kernels", modelName + " NonFluor", 0, 500, 0, 500)
    makeGraph(xActTrans, yNewModPredTrans, "Actual Transmission", "Predicted Transmission", modelName + " Transmission", 0, 100, 0, 100)


def checkFiles(warmanList):
    pngFiles = []
    xmlFiles = []
    for file in os.listdir("Inference\curatedImagesX"):
        if file.endswith(".png"):
            pngFiles.append(file.split(".")[0])
        if file.endswith(".xml"):
            xmlFiles.append(file.split(".")[0])
    for earname in warmanList:
        if earname not in pngFiles:
            print(f"{earname} does not have a .png image")
        if earname not in xmlFiles:
            print(f"{earname}")
    

def getImgList(filename):
    newList = []
    with open(filename, "r") as listfile:
        for line in listfile:
            earname = line.split(".")[0]
            newList.append(earname.strip())
    return newList

def  getCSVfiles(dirname,keyword):
    infFilePaths = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.startswith(keyword) and file.endswith(".csv"):
                infFilePaths.append(os.path.join(root, file))
    return infFilePaths


def recreateWarmanGraphs(dfw, warmanList, dataset):
    warmanFluor, warmanNonFluor, warmanPredTrans = [], [], []
    warmanActFluor, warmanActNonFluor, warmanActTrans = [], [], []
    for index, row in dfw.iterrows():
        # if earname is in the testing set and has actual values associated with it (both should always be true)
        if row["image_name"] in warmanList and row["image_name"] in actFluor:
            warmanFluor.append(row["GFP_tf"])        # Grab fluor vals
            warmanNonFluor.append(row["wt_tf"])      # Grab nonfluor vals
            # Calculate Trans vals
            warmanPredTrans.append(row["GFP_tf"] / (row["wt_tf"] + row["GFP_tf"]) * 100)#

            # Make lists of associated actual values
            warmanActFluor.append(actFluor[row["image_name"]])
            warmanActNonFluor.append(actNonFluor[row["image_name"]])
            warmanActTrans.append(actTrans[row["image_name"]])   
    '''
    print(f"Warman {dataset}")
    print(f"maxWarmanFluor:{max(warmanFluor)}")
    print(f"maxWarmanNonFluor:{max(warmanNonFluor)}")
    print(f"maxWarmanTrans:{max(warmanPredTrans)}")
    print()
    '''

    makeGraph(warmanActFluor, warmanFluor, "Actual Number of Fluorescent Kernels", "Predicted Number of Fluorescent Kernels", f"Warman {dataset} Fluor", 0, 400, 0, 400)
    makeGraph(warmanActNonFluor, warmanNonFluor, "Actual Number of NonFluorescent Kernels", "Predicted Number of NonFluorescent Kernels", f"Warman {dataset} NonFluor", 0, 400, 0, 400)
    makeGraph(warmanActTrans, warmanPredTrans, "Actual Transmission", "Predicted Transmission", f"Warman {dataset} Transmission", 0, 100, 0, 100)

    # TODO: actually make and save graphs!!!


def stuff(infList, data, dataset):
    for fileName in infList:
        df = pd.read_csv(fileName)
        modelName = fileName.split("\\")[-2].split("_")[-1]
        makeWarmanCompGraphs(df, modelName + " " + dataset, data)


def warmanGraphDriver(warmanX, warmanY):
    # This data was used in Figure 5 of the Warman paper, found at https://github.com/fowler-lab-osu/maize_ear_scanner_and_computer_vision_statistics/blob/master/data/test_set_two_models_predictions_2018_summary.tsv
    # and https://github.com/fowler-lab-osu/maize_ear_scanner_and_computer_vision_statistics/blob/master/data/test_set_two_models_predictions_2019_summary.tsv
    dfwx = pd.read_csv("Inference/testingSetWarmanPaperX/test_set_two_models_predictions_2018_summary.tsv", sep="\t")
    dfwy = pd.read_csv("Inference/testingSetWarmanPaperY/test_set_two_models_predictions_2019_summary.tsv", sep="\t")

    # Recreate Warman's Fig. 5 results for as close to a one-to-one comparison as possible
    recreateWarmanGraphs(dfwx, warmanX, "2018 X")
    recreateWarmanGraphs(dfwy, warmanY, "2019 Y")


def kernelCounts(infList, dataset):
    for fileName in infList:
        df = pd.read_csv(fileName)
        modelName = fileName.split("\\")[-2].split("_")[-1]

        predFluor, predNonFluor, predTotal = [], [], []
        actFluor, actNonFluor, actTotal = [], [], []

        '''
        Percent difference = abs(difference) / avgOfNums
        '''
        fluorPerDiffList, nonFluorPerDiffList, totalPerDiffList = [], [], []

        with open("KernelCountComparisons/KernelCounts_" + modelName + "_" + dataset + "_PercentDifference.csv", "w") as infile:
            infile.write("EarName,FluorPerDiff,NonFluorPerDiff,TotalPerDiff\n")
            for index, row in df.iterrows():
                if row["TrainingSet"] != True:
                    if row["ActualTransmission"] > -1:
                        predFluor.append(row["PredictedFluor"])
                        predNonFluor.append(row["PredictedNonFluor"])
                        predTotal.append(row["PredictedTotal"])
                        actFluor.append(row["ActualFluor"])
                        actNonFluor.append(row["ActualNonFluor"])
                        actTotal.append(row["ActualTotal"])

                        fluorPerDiff = ((row["PredictedFluor"] - row["ActualFluor"])) / ((row["PredictedFluor"] + row["ActualFluor"]) / 2)
                        nonFluorPerDiff = ((row["PredictedNonFluor"] - row["ActualNonFluor"])) / ((row["PredictedNonFluor"] + row["ActualNonFluor"]) / 2)
                        totalPerDiff = ((row["PredictedTotal"] - row["ActualTotal"])) / ((row["PredictedTotal"] + row["ActualTotal"]) / 2)

                        #if (fluorPerDiff > 0.05) or (nonFluorPerDiff > 0.05) or (totalPerDiff > 0.05):
                        infile.write(f"{row['EarName']},{fluorPerDiff:.4f},{nonFluorPerDiff:.4f},{totalPerDiff:.4f}\n")
                        fluorPerDiffList.append(fluorPerDiff)
                        nonFluorPerDiffList.append(nonFluorPerDiff)
                        totalPerDiffList.append(totalPerDiff)  
            avgFluorPerDiff = sum(fluorPerDiffList) / len(fluorPerDiffList)
            avgNonFluorPerDiff = sum(nonFluorPerDiffList) / len(nonFluorPerDiffList)
            avgTotalPerDiff = sum(totalPerDiffList) / len(totalPerDiffList)
            infile.write(f"Average,{avgFluorPerDiff:.4f},{avgNonFluorPerDiff:.4f},{avgTotalPerDiff:.4f}\n")          

            fluorPerDiffList = [abs(i) for i in fluorPerDiffList if i == i]
            nonFluorPerDiffList = [abs(i) for i in nonFluorPerDiffList if i == i]
            totalPerDiffList = [abs(i) for i in totalPerDiffList if i == i]
            avgFluorPerDiff = sum(fluorPerDiffList) / len(fluorPerDiffList)
            avgNonFluorPerDiff = sum(nonFluorPerDiffList) / len(nonFluorPerDiffList)
            avgTotalPerDiff = sum(totalPerDiffList) / len(totalPerDiffList)
            infile.write(f"AbsoluteAverage,{avgFluorPerDiff:.4f},{avgNonFluorPerDiff:.4f},{avgTotalPerDiff:.4f}")          

        #makeGraph(predFluor,actFluor,"PredictedFluor","ActualFluor","title",0,400)
        #makeGraph(predNonFluor,actNonFluor,"PredictedNonFluor","ActualNonFluor","title",0,400)
        #makeGraph(predTotal,actTotal,"PredictedTotal","ActualTotal","title",0,600)


def getHyperparams(model):
    #03.06.23_12.55PM_022  
    model = model[:-4]
    hyperparams = {}
    for root, dirs, files in os.walk("SavedModels"):
        for dir in dirs: 
            if dir.endswith(model):
                hyperparamPath = os.path.join("SavedModels", dir)
                try:
                    with open(hyperparamPath + "/Hyperparameters.txt", "r") as hfile:
                        lines = hfile.readlines()
                        for line in lines:
                            pair = line.split("=")
                            hyperparams[pair[0].strip()] = pair[1].strip()
                except:
                    hyperparams['validationPercentage'] = -1
                    hyperparams['batchSize'] = -1
                    hyperparams['learningRate'] = -1
                    hyperparams['epochs'] = -1
                    hyperparams['rpn_pre_nms_top_n_train'] = -1
                    hyperparams['rpn_post_nms_top_n_train'] = -1
                    hyperparams['rpn_pre_nms_top_n_test'] = -1
                    hyperparams['rpn_post_nms_top_n_test'] = -1
                    hyperparams['rpn_fg_iou_thresh'] = -1
                    hyperparams['rpn_batch_size_per_image'] = -1
                    hyperparams['min_size'] = -1
                    hyperparams['trainable_backbone_layers'] = -1
                    hyperparams['box_nms_thresh'] = -1
                    hyperparams['box_score_thresh'] = -1

    return hyperparams


def compareInferences(csvList, dataset):
    scoreDict = {}
    statsDict = {}
    for csvFile in csvList:
        df = pd.read_csv(csvFile)
        if "Stats" in csvFile:
            outStr = ""
            for index, row in df.iterrows():
                try:
                    try:
                        outStr = f"{row['Inference']},{row['Model']},{row['NotInTrainingAvgTransABSDiff']}," + \
                        f"{row['NotInTrainingAvgPredActTransRatio']},{row['NumberImagesForHandAnnotation']}," + \
                        f"{row['NotInTrainingImagesAvgPredAmbigs']},{row['NotInTrainingAvgTransDiff']}," + \
                        f"{row['NotInTrainingAvgFluorPerDiff']},{row['NotInTrainingFluorPerDiffABS']}," + \
                        f"{row['NotInTrainingNonFluorPerDiff']},{row['NotInTrainingNonFluorPerDiffABS']}," + \
                        f"{row['NotInTrainingTotalPerDiff']},{row['NotInTrainingTotalPerDiffABS']}," + \
                        f"{row['NotInTrainingAvgEarScore']}"
                    except:
                        outStr = f"{row['Inference']},{row['Model']},{row['NotInTrainingSetAvgTransABSDiff']}," + \
                        f"{row['NotInTrainingSetAvgPredActTransRatio']},{row['NumberImagesForHandAnnotation']}," + \
                        f"{row['AllImagesAvgPredAmbigs']},{row['NotInTrainingSetAvgTransDiff']},na,na,na,na,na,na,"
                except:
                    print(f"This is ridiculous {csvFile}")

                statsDict[row['Inference']] = outStr
        else:
            scores, perDiffAll = [], []
            for index, row in df.iterrows():
                if row["TrainingSet"] != True:
                    scores.append(row['AverageEarScoreAll'])
                    
            avgScore = sum(scores) / len(scores)
            infName = csvFile.split("\\")[-1]
            scoreDict[infName[:-4]] = avgScore
    
    with open("InferenceOutputAggregatedData_" + dataset + "_0725.csv", "w") as outfile:
        outfile.write(
            "Inference,Model,NotInTrainingAvgTransABSDiff,NotInTrainingAvgPredActTransRatio,NumberImagesForHandAnnotation," +
            "NotInTrainingImagesAvgPredAmbigs,NotInTrainingAvgTransDiff,NotInTrainingAvgFluorPerDiff," +
            "NotInTrainingFluorPerDiffABS,NotInTrainingNonFluorPerDiff,NotInTrainingNonFluorPerDiffABS," +
            "NotInTrainingTotalPerDiff,NotInTrainingTotalPerDiffABS,NotInTrainingAvgEarScore\n")

        for key in scoreDict:
            if key in statsDict:
                outfile.write(f"{statsDict[key]},\n")
            else:
                outfile.write(f"{key},{scoreDict[key]}\n")



def main():
    # Make graphs for Jose Data
    #bInf = "Inference/Feb2023_Scans_BeforeEarVision/InferenceOutput-Jose_07.18.23_11.24AM-027-08.11_14.42/InferenceOutput-Jose_07.18.23_11.24AM-027-08.11_14.42.csv"
    bInf = "Inference/B2023_Ears_FixedOrientation/InferenceOutput-Jose_07.18.23_11.24AM-027-09.20_11.33/InferenceOutput-Jose_07.18.23_11.24AM-027-09.20_11.33.csv"
    #xInf = "Inference/testingSetWarmanPaperX/InferenceOutput-Jose_07.18.23_11.24AM-027-07.21_04.02/InferenceOutput-Jose_07.18.23_11.24AM-027-07.21_04.02.csv"
    #yInf = "Inference/testingSetWarmanPaperY/InferenceOutput-Jose_07.18.23_11.24AM-027-07.21_02.49/InferenceOutput-Jose_07.18.23_11.24AM-027-07.21_02.49.csv"

    #graphDriver(bInf, "2022 B totalPreds > 100, numAmbigs < 15", True)
    #graphDriver(xInf, "2018 X totalPreds > 100, numAmbigs < 15", True)
    graphDriver(bInf, "stuff", 0)
    graphDriver(bInf, "otherStuff", 1)
    graphDriver(bInf, "moreStuff", 2)


    # Get lists of all images that were in each testing set, from https://datacommons.cyverse.org/browse/iplant/home/shared/EarVision_maize_kernel_image_data/testing_images
    #warmanX = getImgList("Inference/2018X_earLIst.txt")
    #warmanY = getImgList("Inference/2019Y_earList.txt")

    # Make lists of PATHS to InferenceOutput csv files
    #bInfs = getCSVfiles("Inference/Feb2023_Scans_BeforeEarVision", "InferenceOutput")
    #yInfs = getCSVfiles("Inference/testingSetWarmanPaperY", "InferenceOutput")
    #xInfs = getCSVfiles("Inference/testingSetWarmanPaperX", "InferenceOutput")

    # Extract data from inference csv files and graph results
    # These graphs will have the new inference data for each inference, but will only contain image data that was used 
    # in Fig 5 of Warman's paper (see above for datacommons link) 

    #bCSVs = getCSVfiles("Inference/Feb2023_Scans_BeforeEarVision", "Inference")
    #yCSVs = getCSVfiles("Inference/testingSetWarmanPaperY", "Inference")
    #xCSVs = getCSVfiles("Inference/testingSetWarmanPaperX", "Inference")
    
    #compareInferences(xCSVs, "X")
    #compareInferences(yCSVs, "Y")
    #compareInferences(bCSVs, "B")


    #stuff(yInfs, warmanY, "Y")
    #stuff(xInfs, warmanX, "X")

    #warmanGraphDriver(warmanX, warmanY)

    #kernelCounts(yInfs, "Y")
    #kernelCounts(xInfs, "X")
    #kernelCounts(bInfs, "B")



main()