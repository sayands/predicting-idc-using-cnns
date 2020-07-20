# Detection of Invasive Ductal Carcinoma using Convolutional Neural Networks

Sayan Deb Sarkar & J.R. Harish Kumar [Project Report](https://drive.google.com/file/d/1MR7A2ovi3dLsIqIiatKzMJyxKny1kw7u/view?usp=sharing)

### Abstract

We propose a simple and effective convolutional neural network for automated detection of invasive ductal carcinoma using whole-slide images of breast cancer tissues. The network comprises two feature extraction blocks of convolutional and pooling layers, followed by fully convolutional layers to distinguish between the features of each class, and subsequently ending with fully connected layers. Experimental evaluation demonstrated promising quantitative results for the detection
of invasive cancer tissues in terms of F1-score and balanced accuracy of 88.30% and 87.35%, respectively. In comparison to the best-performing handcrafted features and deep learning based state-of-the-art methods, the proposed architecture performs better by a margin of 12% and 2% in terms of F1-score and balanced accuracy, respectively. In addition, we achieve precision and recall scores of 0.8877 and 0.8809 respectively, on a highly unbalanced dataset of whole- slide histopathology
images from 162 women diagnosed with invasive ductal carcinoma at the Hospital of the University of Pennsylvania and The Cancer Institute of New Jersey.

### Installation

* Create a conda environment 
    ``` conda create -n idc-env python=3.7
        conda activate idc-env```
* Install the required packages
    ``` pip install -r requirements.txt```
