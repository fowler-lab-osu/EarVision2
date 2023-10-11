'''
EarVision 2.0:
RunInferencesOnMultipleModels

This script performs inferences on specified datasets using multiple models. It can be used when the user wants to test 
multiple models against each other with the same or multiple datasets. "NOTE" will be in the comments above places where 
the code can be easily customized to the current needs of the user.
'''

from Infer import Infer
import os
import logging
import datetime

homeDirec = os.getcwd()

# NOTE: Add or remove directories to run inference on
datasetsToTest = [
    os.path.join(homeDirec, "Inference/B2023_Ears_FixedOrientation")
    #os.path.join(homeDirec, "Inference/Feb2023_Scans_BeforeEarVision"),
    #os.path.join(homeDirec, "Inference/testingSetWarmanPaperY"),
    #os.path.join(homeDirec, "Inference/testingSetWarmanPaperX")
]

# NOTE: Add or remove models to test. Each sublist contains first the model identifier, then the epoch. 
modelsToTest = [
    ["Jose_07.18.23_11.24AM","027"]       # Jose
]

# NOTE: Customize outfolder and log names here. Output will appear in this folder, within Inference/{dataset}/.
outFolder = ""
logName = "newlogname.log"


for dataset in datasetsToTest:
    os.makedirs(f"{dataset}", exist_ok = True)
    logging.basicConfig(filename=f"{dataset}/{logName}", level=logging.INFO)
    for model in modelsToTest:  
        try:
            # NOTE: be sure to update the directories where the model may be found.
            modelID = f"{model[0]}"
            time = datetime.datetime.now().strftime('%H.%M')
            logging.info(f"{time} Testing model {modelID} at epoch {model[1]}")
            print(f"{time} Testing model {modelID} at epoch {model[1]} ")                 
            Infer(modelID, model[1], dataset)
        except:
            logging.exception("")