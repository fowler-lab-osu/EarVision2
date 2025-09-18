**EarVision.v2**

Phenotyping system described in paper "Spatial inheritance patterns across maize ears are associated with alleles that reduce pollen fitness".


Repo non-final.

Documentation in progress.

<hr>

**Adding Training Data**

To add training data, place image files in ._/TrainingData/All/Images/_ and all xml files containing annotations in _./TrainingData/All/Annotations/_.
When retraining the model, the application uses image and annotation data from the _All/_ subdirectories. If the data is not present in there, it will not be included in the retraining.

After training is complete, the new model and the associated data will be found in _./EarVision/SavedModels/_ as a new directory with the timestamp of when you began the training.
