**Note: This is an extremely barebones readme specific for use in the Fowler Lab**

<hr>

**Adding Training Data**

To add training data, create a new folder in the _./EarDataset/_ directory and name it the year the data is from (e.g., 2014). Within that folder, create two new folders: _Annotations/_ and _Images/_. Place all image files in ._/EarDataset/Images/_ and all xml files in _./EarDataset/Annotations/_. Then, add copies of all image and xml files to _./EarDataset/All/Images/_ and _./EarDataset/All/Annotations/_, respectively. 

When retraining the model, the application uses image and annotation data from the _All/_ sundirectories. If the data is not present in there, it will not be included in the retraining.

After training is complete, the new model and the associated data will be found in _./EarVision/SavedModels/_ as a new directory with the timestamp of when you began the training.

**Adding Inference Data**

Create a new folder in _./EarVision/Inference/_ and name it whatever you want (recommend using a name indicating the year). Add all image and xml files to this new directory together.

When an inference is run, a new folder will be created inside your folder with the name of the model appended to it, like so: _./EarVision/Inference/MyNewData/InferenceMyNewData-ModelName/_ . Annotated images, xml files, json files, and images indicating ambiguous kernels will be generated in this new subdirectory, as well as accompanying csv data files.

