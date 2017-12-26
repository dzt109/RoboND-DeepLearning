# Quadrotor Follow Me Project #

## Deepak Trivedi, December 2017

### Contents

 - Project Introduction
 - Project Setup
 - Fully Convolutional Neutral Network Architecture
 - Model hyperparameters
 - Techniques and Concepts
 - Parameter tuning
 - Image manipulation
 - Limitations and Future Work 


### Project Introduction

The objective of this project is to train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics, such as advanced cruise control in autonomous vehicles or human-robot collaboration in industry. A necessary enabler to achieve this is "semantic segmentation" of images captured by the quadrotor drone. This project uses Fully Convolutional Networks (FCNs) in order to identify various parts of an image captured by the quadrotor as background, or a general person, or a specific target ("hero") that is to be tracked. Tensorflow is used to set up the FCN model, and due to the required computational resources, a virtual machine on Amazon Web Services (AWS) was deployed to train the model. 

A number of hyperparameters need to be tuned in order to get a good model. The metric used to assess the quality of the model is the IOU (intersection over union) metric. This metric takes the intersection of the prediction
pixels and ground truth pixels and divides it by their union.

The model was trained to allow the drone to follow a certain hero, which is the person marked red in the image below. The model will need to be retrained on a different hero, if the target changes. The model can successfully distinguish the hero from other human subjects. 

 

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

### Project Setup

Following is a summary of how the project was initially setup

1. Clone the repository
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

2. Download the data. Following is some initial training, validation and evaluation data : [Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip), [Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip), [Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

3. Download the QuadSim binary. The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

4. Install Dependencies. The project works with Python 3 and Jupyter Notebooks. Following are the frameworks and packages required to run the project: Python 3.x, Tensorflow 1.2.1, NumPy 1.11, SciPy 0.17.0, eventlet, Flask, h5py, PIL, python-socketio, scikit-image, transforms3d, PyQt4/Pyqt5

### Collecting Training Data

A basic training dataset has been provided in this project's repository. Additional training data could be collected by training running the quadrotor simulator in the `DL Training` mode and recording photographs while the quadrotor is flying. I was able to get the required model accuracy without having to collect additional training data.   

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```

## Fully Convolutional Neural Network Architecture

The FCN Network is implemented using Tensorflow in `model_training.ipynb` and trained on the AWS Udacity Robotics Laboratory Community AMI.  Keras, a high level deep learning API, is used to build the network. This allows us to focus on network architecture rather than on smaller details building up to the network.   

An FCN is composed of an input layer, one or more encoder layers, a middle layer (1x1 convolution), one or more 'decoder' layers, and an output layer. Following is a description of each of these layers:

Input layer: This is simply the image being processed. 

**Encoder layers**:  These are convolutional layers that reduce to a deeper 1x1 convolution layer. This in contrast to a flat fully connected layer used for basic classification of images. This difference has the effect of preserving spatial information from the image, and is therefore useful for semantic segmentation of images.
 
Due to its lower computational cost, we use **(depthwise) separable convolutions** for building the encoder layers. These comprise of a convolution performed over each channel of an input layer and followed by a 1x1 convolution that takes the output channels from the previous step and then combines them into an output layer. This leads to a reduction in the number of parameters to be tuned.

The **middle layer** is a 1x1 convolutional layer with a kernel size of 1 and a stride of 1. 

**Decoder layers** provide upsampling 

In all these layers, we use **batch normalization**. This means that instead of just normalizing the inputs to the network, we normalize the inputs to each layer within the network. During training, each layer's inputs are normalized using the mean and variance of the values in the current mini-batch. The benefits of doing this includes higher learning rates, faster learning, and an element of regularization for the network parameters.  


## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this. Or improve your network architecture and hyperparameters. 

**Obtaining a Leaderboard Score**

Share your scores in slack, and keep a tally in a pinned message. Scores should be computed on the sample_evaluation_data. This is for fun, your grade will be determined on unreleased data. If you use the sample_evaluation_data to train the network, it will result in inflated scores, and you will not be able to determine how your network will actually perform when evaluated to determine your grade.

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--pred_viz` parameter to `follower.py`
