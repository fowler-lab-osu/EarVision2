'''
EarVision 2.0:
SelectInferenceDirectory

This script prompts the user to select their preferred dataset for inference during normal execution of the program. It 
will also prompt user for their preffered prediction filter settings, then calls Infer.
'''

from Infer import Infer
from tkinter import Tk
from tkinter import *
import tkinter.filedialog as filedialog
import os

homeDirec = os.getcwd()

root = Tk()
root.withdraw()
inferenceDirectoryPath = filedialog.askdirectory(initialdir=homeDirec+"/Inference")

print("Opening " + inferenceDirectoryPath)

print("Choose filters (press Enter for default)")
print("Total predicted kernels (100): ")
total = input()
print("Number ambiguous kernels (0): ")
ambigs = input()
#print("Average ear score, 0 to 1 (0): ")
#score = input()
print("Percent ambiguous [out of 100] (20): ")
per = input()
score = 0

if not total:
    total = 100
if not ambigs:
    ambigs = 0
if not score:
    score = 0
if not per:
    per = 20

filters = [int(total), int(ambigs), int(score), float(per)]

root.destroy()
Infer("Jose_07.18.23_11.24AM", "027", inferenceDirectoryPath, filters)