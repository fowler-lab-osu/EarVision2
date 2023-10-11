'''
EarVision 2.0:
Infer

This script loads a pre-trained model, sets a device to run the inference on (preferrably GPU), iterates over the given 
image dataset, and performs an inference using that model. 

Model may be changed in main() where function Infer is called. It is currently set as the Jose model developed 2023-07

Predictions and metrics may be found in C:/Users/CornEnthusiast/Projects/EarVision/Inference/{dataset}
'''

import torch
import torchvision.transforms.functional as TF
import torchvision.models.detection as objDet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import datetime
from Train import outputAnnotatedImgCV, loadHyperparamFile
from Utils import *
from EarName import *
from Metrics import *


def Infer(modelDir, epochStr, dirPath = os.getcwd(), filters = [100, 15, 0, 0]):
    time = datetime.datetime.now().strftime('%m.%d_%H.%M')
    bDict = getBYearFamilyData()
    numImagesHandAnno = 0

    print("Running EarVision 2.0 Inference")
    print("----------------------")
    print("FINDING GPU")
    print("----------------------")
    print("Currently running CUDA Version: ", torch.version.cuda)

    device = findDevice()

    print(f"Loading Saved Model: {modelDir}\tEpoch: {epochStr}")

    hyperparameters = loadHyperparamFile(f"SavedModels/{modelDir}/Hyperparameters.txt")

    try:
        model = objDet.fasterrcnn_resnet50_fpn_v2(
            weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            box_detections_per_img=700, 
            rpn_pre_nms_top_n_train = hyperparameters["rpn_pre_nms_top_n_train"], 
            rpn_post_nms_top_n_train = hyperparameters["rpn_post_nms_top_n_train"], 
            rpn_pre_nms_top_n_test = hyperparameters["rpn_pre_nms_top_n_test"],   
            rpn_post_nms_top_n_test = hyperparameters["rpn_post_nms_top_n_test"], 
            rpn_fg_iou_thresh = hyperparameters["rpn_fg_iou_thresh"], 
            trainable_backbone_layers = hyperparameters["trainable_backbone_layers"],  
            rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"], 
            box_nms_tresh = hyperparameters["box_nms_thresh"], 
            box_score_thresh = hyperparameters["box_score_thresh"]
        )
    except:
        model = objDet.fasterrcnn_resnet50_fpn_v2(
            weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            box_detections_per_img=700, 
            rpn_pre_nms_top_n_train = hyperparameters["rpn_pre_nms_top_n_train"], 
            rpn_post_nms_top_n_train = hyperparameters["rpn_post_nms_top_n_train"], 
            rpn_pre_nms_top_n_test = hyperparameters["rpn_pre_nms_top_n_test"],   
            rpn_post_nms_top_n_test = hyperparameters["rpn_post_nms_top_n_test"], 
            rpn_fg_iou_thresh = hyperparameters["rpn_fg_iou_thresh"], 
            trainable_backbone_layers = hyperparameters["trainable_backbone_layers"],  
            rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"]
        )
    
    # Potentially add to hyperparameters:
    # rpn_score_thresh = 0.15
    
    # Weird but unless you do this it defaults to 91 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Give the number of classes, including background class
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3) 
    model.to(device)

    # Load saved model and set to eval (because some layers are set to train upon creation)
    model.load_state_dict(torch.load(f"SavedModels/{modelDir}/EarVisionModel_{epochStr}.pt"))
    model.eval() 

    trainingSet = getTrainingSet(modelDir)
    imagePaths = buildImagePathList(dirPath)

    modelID = modelDir.split("/")[-1]    
    inferenceIdentifier = f"InferenceOutput-{modelID}-{epochStr}-{time}"
    print(f"{inferenceIdentifier} hyperparameters:\n{hyperparameters}\n")
    outputDirectory = dirPath + "/" + inferenceIdentifier
    os.makedirs(outputDirectory, exist_ok = True)

    # needAnnotations directory is for images that fail the filters. Copies of those images will also appear in the main
    # output directory for the inference and will need to be manually replaced if a hand annotation is done.
    newAnnoDir = outputDirectory + "/needAnnotations"
    os.makedirs(newAnnoDir, exist_ok=True)

    outFile = open(f"{outputDirectory}/{inferenceIdentifier}.csv", "w")
    outFile.write(f"EarName,Year,CrossType,EarFamily,EarSubFamily,EarPlantNumber,EarAllele,PollenFamily,PollenSubFamily,PollenPlantNumber" +
                  f",PollenPollinationNumber,PollenAllele,TrainingSet,PredictedFluor,PredictedNonFluor,PredictedTotal,PredictedTra" +
                  f"nsmission,AmbiguousKernels,AmbiguousKernelPercentage,AverageEarScoreFluor,AverageEarScoreNonFluor" +
                  f",AverageEarScoreAll,ActualFluor,ActualNonFluor,ActualAmbiguous,ActualTotal,ActualTransmission,Flu" +
                  f"orKernelDiff,FluorKernelABSDiff,NonFluorKernelDiff,NonFluorKernelABSDiff,TotalKernelDiff,TotalKer" +
                  f"nelABSDiff,TransmissionDiff,TranmissionABSDiff,PredtoActTransmissionRatio,FluorPerDiff,NonFluorPe" +
                  f"rDiff,TotalPerDiff\n")

    # Inference metrics to be used in comparing inferences
    statsDict = {
        "listTransABSDiff": [],
        "listPredActTransRatios": [],
        "listPredAmbigs": [],
        "listTransDiff": [],
        "listFluorPerDiff": [],
        "listNonFluorPerDiff": [],
        "listTotalPerDiff": [],
        "listAmbigsNotInTraining": [],
        "listScores": [],
        "listTransABSDiffFilter": [],
        "listPredActTransRatiosFilter": [],
        "listPredAmbigsFilter": [],
        "listTransDiffFilter": [],
        "listFluorPerDiffFilter": [],
        "listNonFluorPerDiffFilter": [],
        "listTotalPerDiffFilter": [],
        "listAmbigsNotInTrainingFilter": [],
        "listScoresFilter": []
    }

    for path in tqdm(imagePaths):
        # Bring sample in as RGB
        image = Image.open(path).convert('RGB') 
        imageTensor = TF.to_tensor(image).to(device).unsqueeze(0)
        actualFluor, actualNonFluor, actualAmb, actualTransmission, actualTotal = 0, 0, 0, 0, 0
        xmlAvail = True

        try:
            xmlTree = ET.parse(path.split(".")[0] + ".xml")
        except:
            xmlAvail = False

        if xmlAvail:
            xmlAvail, actualFluor, actualNonFluor, actualAmb, actualTransmission, actualTotal = \
                parseXMLData(xmlAvail, xmlTree)


        with torch.no_grad(): 
            prediction = model(imageTensor)[0]

        #keptBoxes = torchvision.ops.nms(prediction['boxes'], prediction['scores'], 0.2 )
        finalPrediction = prediction

        fileName = path.replace("\\", "/").split("/")[-1]

        # Next three lines create files
        outputAnnotatedImgCV(
            imageTensor[0], finalPrediction, outputDirectory+"/"+ fileName.split(".")[0] + "_inference.png")
        outputPredictionAsXML(finalPrediction, outputDirectory+"/" + fileName.split(".")[0]+"_inference.xml")
        convertPVOC(outputDirectory+"/" + fileName.split(".")[0]+"_inference.xml", image.size)

        predNonFluor = finalPrediction['labels'].tolist().count(1)
        predFluor = finalPrediction['labels'].tolist().count(2)  
    
        # Next line creates file AND returns number of ambiguous kernels
        ambiguousKernelCount = findAmbiguousCalls(
            imageTensor[0], finalPrediction, outputDirectory+"/"+ fileName.split(".")[0] + "_inference.png")
        statsDict["listPredAmbigs"].append(ambiguousKernelCount)

        earName = fileName.split(".")[0]
        earNameObj = EarName(earName) 
        setAllele(earNameObj, bDict)
        outFile.write(f"{earNameObj.__csvEarData__()}")

        inTrainingSet = False
        if earName in trainingSet:
            inTrainingSet = True

        predNonFluor -= ambiguousKernelCount
        predFluor -= ambiguousKernelCount 
        predTotal = predFluor + predNonFluor 

        imgsForHandAnnotation = False
        #filters = [total, ambigs, score, per]
        # Filter out images that have fewer than the set number of kernels total
        if not inTrainingSet and predTotal <= filters[0]:
            imgsForHandAnnotation = True
        # Filter out images that have more than the set number of ambiguous kernels
        elif not inTrainingSet and ambiguousKernelCount >= filters[1]:
            imgsForHandAnnotation = True

        # Image, xml, and json files are created for the handAnnotation directory. These are copies of the ones 
        # available from the main inference folder.
        if imgsForHandAnnotation:
            numImagesHandAnno += 1
            outputAnnotatedImgCV(
                imageTensor[0], finalPrediction, newAnnoDir+"/"+ fileName.split(".")[0] + "_inference.png")
            outputPredictionAsXML(finalPrediction, newAnnoDir+"/" + fileName.split(".")[0]+"_inference.xml")
            convertPVOC(newAnnoDir+"/" + fileName.split(".")[0]+"_inference.xml", image.size)
            x = findAmbiguousCalls(
                imageTensor[0], finalPrediction, newAnnoDir+"/"+ fileName.split(".")[0] + "_inference.png")

        try:
            ambiguousKernelPercentage = round(
                ambiguousKernelCount/(predFluor + predNonFluor - ambiguousKernelCount)*100, 3)     
        except:
            ambiguousKernelPercentage  = "N/A"

        try:
            predTransmission =   predFluor /  (predTotal) * 100
        except:
            predTransmission = "N/A"

        scores = finalPrediction['scores']
        labels = finalPrediction['labels']

        fluorScores = [score.item() for ind,score in enumerate(scores)  if labels[ind].item()== 2 ]
        nonFluorScores =[score.item() for ind,score in enumerate(scores) if labels[ind].item()== 1 ] 

        # Confidence in the predictions
        avgEarScoreFluor = round(np.mean(fluorScores), 3)
        avgEarScoreNonFluor = round(np.mean(nonFluorScores), 3)
        avgEarScoreAll = round(torch.mean(scores).item(), 3)

        if(inTrainingSet):
            outFile.write("True,")
        else:
            outFile.write(",")

        outFile.write(f"{predFluor},{predNonFluor},{predTotal},{predTransmission},{ambiguousKernelCount}," + 
                      f"{ambiguousKernelPercentage},{avgEarScoreFluor},{avgEarScoreNonFluor},{avgEarScoreAll},")      

        if(xmlAvail):
            #xmlStats = calculateCountMetrics(
            #    [predFluor, predNonFluor], [actualFluor, actualNonFluor], actualTotalInclAmbig = actualTotal)
            xmlStats = CountMetrics(predFluor, predNonFluor, actualFluor, actualNonFluor)

            outFile.write(f"{actualFluor},{actualNonFluor},{actualAmb},{actualTotal},{actualTransmission}," +
                          f"{xmlStats.fluorKernelDiff},{xmlStats.fluorKernelABSDiff}," + 
                          f"{xmlStats.nonFluorKernelDiff},{xmlStats.nonFluorKernelABSDiff}," +
                          f"{xmlStats.totalKernelDiff},{xmlStats.totalKernelABSDiff}," +
                          f"{xmlStats.transmissionDiff},{xmlStats.transmissionABSDiff}," +
                          f"{predTransmission/actualTransmission},{xmlStats.fluorPerDiff}," +
                          f"{xmlStats.nonFluorPerDiff},{xmlStats.totalPerDiff}")

            if not inTrainingSet:
                statsDict["listTransDiff"].append(xmlStats.transmissionDiff)
                statsDict["listTransABSDiff"].append(xmlStats.transmissionABSDiff)
                statsDict["listPredActTransRatios"].append(predTransmission/actualTransmission)
                statsDict["listFluorPerDiff"].append(xmlStats.fluorPerDiff)
                statsDict["listNonFluorPerDiff"].append(xmlStats.nonFluorPerDiff)
                statsDict["listTotalPerDiff"].append(xmlStats.totalPerDiff)
                statsDict["listAmbigsNotInTraining"].append(ambiguousKernelCount)
                statsDict["listScores"].append(avgEarScoreAll)
            if not imgsForHandAnnotation and not inTrainingSet:
                statsDict["listTransDiffFilter"].append(xmlStats.transmissionDiff)
                statsDict["listTransABSDiffFilter"].append(xmlStats.transmissionABSDiff)
                statsDict["listPredActTransRatiosFilter"].append(predTransmission/actualTransmission)
                statsDict["listFluorPerDiffFilter"].append(xmlStats.fluorPerDiff)
                statsDict["listNonFluorPerDiffFilter"].append(xmlStats.nonFluorPerDiff)
                statsDict["listTotalPerDiffFilter"].append(xmlStats.totalPerDiff)
                statsDict["listAmbigsNotInTrainingFilter"].append(ambiguousKernelCount)
                statsDict["listScoresFilter"].append(avgEarScoreAll)
        outFile.write("\n")
        
    outFile.close()

    createInfStatsFile(outputDirectory, modelID, epochStr, inferenceIdentifier, numImagesHandAnno, statsDict)



def buildImagePathList(imageDirectory):
    imagePaths = []
    for imgIndex, file in enumerate(sorted(os.listdir(imageDirectory))): 
        if(file.endswith((".png", ".jpg", ".tif"))):
            try:
                imagePath = os.path.join(imageDirectory, file)  
                imagePaths.append(imagePath)

            except Exception as e:
                print(str(e))
                pass
    return imagePaths


def parseXMLData(xmlAvail, xmlTree):
    '''
    Parse XML file for images with annotation data available. Return ground truth kernel counts and transmission data. 
    '''
    markerTypeCounts = [0,0,0]
    actualFluor, actualNonFluor, actualAmb, actualTransmission, actualTotal = 0, 0, 0, 0, 0
    xmlRoot = xmlTree.getroot()
    markerData =  xmlRoot.find('Marker_Data')

    for markerType in markerData.findall("Marker_Type"):
        typeID = int(markerType.find('Type').text)
        if(typeID in [1,2,3]):
            markerCount = len(markerType.findall("Marker"))
            markerTypeCounts[typeID-1] = markerCount

    if sum(markerTypeCounts[0:1]) == 0:
        xmlAvail = False
    if xmlAvail:
        actualFluor = markerTypeCounts[0]
        actualNonFluor = markerTypeCounts[1]
        actualAmb = markerTypeCounts[2]

        actualTransmission = actualFluor / (actualNonFluor+actualFluor) * 100

        # should only include fluro and nonfluor, subtract ambiguous
        # actualTotal = sum(markerTypeCounts)   #this would include the ambiguous kernels in the actual total
        actualTotal = actualFluor + actualNonFluor

    return xmlAvail, actualFluor, actualNonFluor, actualAmb, actualTransmission, actualTotal


def createInfStatsFile(outputDirectory, modelID, epochStr, inferenceIdentifier, numImagesHandAnno, statsDict):
    with open(f"{outputDirectory}/InferenceStats-{modelID}-{epochStr}.csv", "w") as statsFile:
        statsFile.write(f"Inference,Model,Date,NumberImagesForHandAnnotation,NotInTrainingAvgTransDiff,FilterNotInTra" +
                        f"iningAvgTransDiff,NotInTrainingAvgTransABSDiff,FilterNotInTrainingAvgTransABSDiff,NotInTrai" +
                        f"ningAvgPredActTransRatio,FilterNotInTrainingAvgPredActTransRatio,NotInTrainingAvgPredAmbigs" + 
                        f",FilterNotInTrainingAvgPredAmbigs,NotInTrainingAvgFluorPerDiff,FilterNotInTrainingAvgFluorP" +
                        f"erDiff,NotInTrainingFluorPerDiffABS,FilterNotInTrainingFluorPerDiffABS,NotInTrainingNonFluo" +
                        f"rPerDiff,FilterNotInTrainingNonFluorPerDiff,NotInTrainingNonFluorPerDiffABS,FilterNotInTrai" +
                        f"ningNonFluorPerDiffABS,NotInTrainingTotalPerDiff,FilterNotInTrainingTotalPerDiff,NotInTrain" +
                        f"ingTotalPerDiffABS,FilterNotInTrainingTotalPerDiffABS,NotInTrainingAvgEarScore,FilterNotInT" +
                        f"rainingAvgEarScore\n")
        
        time = datetime.datetime.now().strftime('%m.%d_%H.%M')

        avgFluorPerDiff = getAvg(statsDict["listFluorPerDiff"])
        avgNonFluorPerDiff = getAvg(statsDict["listNonFluorPerDiff"])
        avgTotalPerDiff = getAvg(statsDict["listTotalPerDiff"])
        
        fluorPerDiffList = [abs(i) for i in statsDict["listFluorPerDiff"] if i == i]
        nonFluorPerDiffList = [abs(i) for i in statsDict["listNonFluorPerDiff"] if i == i]
        totalPerDiffList = [abs(i) for i in statsDict["listTotalPerDiff"] if i == i]
        avgFluorPerDiffABS = getAvg(fluorPerDiffList)
        avgNonFluorPerDiffABS = getAvg(nonFluorPerDiffList)
        avgTotalPerDiffABS = getAvg(totalPerDiffList)
        
        avgFluorPerDiffFilter = getAvg(statsDict["listFluorPerDiffFilter"])
        avgNonFluorPerDiffFilter = getAvg(statsDict["listNonFluorPerDiffFilter"])
        avgTotalPerDiffFilter = getAvg(statsDict["listTotalPerDiffFilter"]) 
        
        fluorPerDiffListFilter = [abs(i) for i in statsDict["listFluorPerDiffFilter"] if i == i]
        nonFluorPerDiffListFilter = [abs(i) for i in statsDict["listNonFluorPerDiffFilter"] if i == i]
        totalPerDiffListFilter = [abs(i) for i in statsDict["listTotalPerDiffFilter"] if i == i]
        avgFluorPerDiffABSFilter = sum(fluorPerDiffListFilter) / len(fluorPerDiffListFilter)
        avgNonFluorPerDiffABSFilter = sum(nonFluorPerDiffListFilter) / len(nonFluorPerDiffListFilter)
        avgTotalPerDiffABSFilter = sum(totalPerDiffListFilter) / len(totalPerDiffListFilter)

        statsFile.write(f"{inferenceIdentifier},{modelID}_{epochStr},{time},{numImagesHandAnno}," +
                        f"{getAvg(statsDict['listTransDiff'])},{getAvg(statsDict['listTransDiffFilter'])}," +
                        f"{getAvg(statsDict['listTransABSDiff'])},{getAvg(statsDict['listTransABSDiffFilter'])}," +
                        f"{getAvg(statsDict['listPredActTransRatios'])}," +
                        f"{getAvg(statsDict['listPredActTransRatiosFilter'])},{getAvg(statsDict['listPredAmbigs'])}," +
                        f"{getAvg(statsDict['listPredAmbigsFilter'])},{avgFluorPerDiff},{avgFluorPerDiffFilter}," +
                        f"{avgFluorPerDiffABS},{avgFluorPerDiffABSFilter},{avgNonFluorPerDiff}," +
                        f"{avgNonFluorPerDiffFilter},{avgNonFluorPerDiffABS},{avgNonFluorPerDiffABSFilter}," +
                        f"{avgTotalPerDiff},{avgTotalPerDiffFilter},{avgTotalPerDiffABS},{avgTotalPerDiffABSFilter}," +
                        f"{getAvg(statsDict['listScores'])},{getAvg(statsDict['listScoresFilter'])},\n")


def getAvg(listName):
    avg = 0
    try:
        avg = sum(listName) / len(listName)
    except:
        avg = "NA"
    return avg

def findDevice():
    # Pointing to our GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU. Device: ", device)
    else:
        device = torch.device("cpu")
        print("Running on CPU. Device: ", device)
    return device


def getTrainingSet(modelDir):
    # Read training set data; metrics are taken on images that were not in the training set.
    trainingSetFile = open(f"SavedModels/{modelDir}/TrainingSet.txt").readlines()
    trainingSet = []
    for l in trainingSetFile:
        trainingSet.append(l.strip().replace('\\', '/').split('/')[-1].split('.')[0])
    return trainingSet


def getBYearFamilyData():
    bYearFamilyAllele = {}
    # dict format: {family1 : [allele1, allele2], family2 : [allele3], etc.}
    with open("Byear_Insertion_Family_forEarVision.csv", "r") as bfile:
        bfile.readline() # don'e need header
        for line in bfile:
            allele, family = line.split(",")
            family = family.strip()
            if int(family) not in bYearFamilyAllele:
                bYearFamilyAllele[int(family)] = ""
            if bYearFamilyAllele[int(family)] != allele:
                bYearFamilyAllele[int(family)] += f" {allele}"
    return bYearFamilyAllele


def setAllele(earnameObj, bYearFamilyAlleleDict):
    if earnameObj.earFamily == 2:
        earnameObj.earAllele = "WT"
    if earnameObj.pollenFamily == 2:
        earnameObj.pollenAllele = "WT"
    if earnameObj.earFamily in bYearFamilyAlleleDict:
        earnameObj.earAllele = bYearFamilyAlleleDict[earnameObj.earFamily]
    if earnameObj.pollenFamily in bYearFamilyAlleleDict:
        earnameObj.pollenAllele = bYearFamilyAlleleDict[earnameObj.pollenFamily]





if __name__ == "__main__":
    Infer("Jose_07.18.23_11.24AM", "027", "Inference/")