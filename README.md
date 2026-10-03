# Roof-Type-Classification
Roof Type Classification with Innovative Machine Learning Approaches

Abstract

In this study, roof type classification was made with a small number of examples for training, in accordance with the one-shot learning approach and using the "Siamese Neural Network" method. 
The images used for training were artificially produced due to the difficulty of finding roof data. 
A dataset consisting of real roof pictures was used for the test. The test and training dataset consists of 3 different types: Flat, Gable and Hip. 
Finally, a CNN-based model and a Siamese Neural Network model were trained with the same datasets and the test results were compared with each other. 

Data
* Hip, Flat and Gable roof type of examples are given.
* Cati images are of three types of examples, Flat, Gable, and Hip, and were produced by the authors using Autodesk Maya software

Codes
* one_shot_model.py is the code for create Siamese model.
* one_shot_mydataset.py is the code for read and label the dataset.
* one_shot_train.py is the code for train your dataset and for get results.

Acknowledgements and licence

* The Siamese-network code in `one_shot_model.py`, `one_shot_mydataset.py` and `one_shot_train.py` is adapted from the open-source PyTorch implementation for Omniglot by fangpin (https://github.com/fangpin/siamese-pytorch), which declares no licence. The data loaders, the training script and all roof-specific changes are by the authors.
* The authors' modifications are released under the MIT licence (see `LICENSE`). The MIT licence does not cover the original code by fangpin.
* The synthetic roof images (`cati*.jpg`) were produced by the authors and are released under CC BY 4.0. Building data used elsewhere in the study are (c) OpenStreetMap contributors (ODbL). Google Maps imagery is not redistributed.
