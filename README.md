# Quadrotor Follow Me Project #

## Deepak Trivedi, 26 December 2017

[image_0]: ./docs/misc/sim_screenshot.png
[image_1]: ./docs/misc/network_arch.PNG
[image_2]: ./docs/misc/try1.png
[image_3]: ./docs/misc/try11.png
[image_4]: ./docs/misc/try111.png
[image_5]: ./docs/misc/try122.png
[image_6]: ./docs/misc/try165.png
[image_7]: ./docs/misc/try2.png
[image_8]: ./docs/misc/try21.png
[image_9]: ./docs/misc/try222.png
[image_10]: ./docs/misc/try3.png
[image_11]: ./docs/misc/try31.png
[image_12]: ./docs/misc/try34342.png
[image_13]: ./docs/misc/try36554.png
[image_14]: ./docs/misc/try4.png
[image_15]: ./docs/misc/try48678.png
[image_16]: ./docs/misc/try486784.png
[image_17]: ./docs/misc/try49870.png
[image_18]: ./docs/misc/try5-0908.png
[image_19]: ./docs/misc/try5.png
[image_20]: ./docs/misc/try509.png
[image_21]: ./docs/misc/try5980.png
[image_22]: ./docs/misc/Gradient_descent.png

### Contents

 - Project Introduction
 - Project Setup
 - Fully Convolutional Neutral Network Architecture
 - Model hyperparameters
 - Parameter tuning
 - Limitations and Future Work 


### Project Introduction

The objective of this project is to train a deep neural network to allow a drone to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics, such as advanced cruise control in autonomous vehicles or human-robot collaboration in industry. A necessary enabler to achieve this is "semantic segmentation" of images captured by the quadrotor drone. This project uses Fully Convolutional Networks (FCNs) in order to identify various parts of an image captured by the quadrotor as background, or a general person, or a specific target ("hero") that is to be tracked. Tensorflow is used to set up the FCN model, and due to the required computational resources, a virtual machine on Amazon Web Services (AWS) was deployed to train the model. 

A number of hyperparameters need to be tuned in order to get a good model. The metric used to assess the quality of the model is the IOU (intersection over union) metric. This metric takes the intersection of the prediction
pixels and ground truth pixels and divides it by their union.

The model was trained to allow the drone to follow a certain hero, which is the person marked red in the image below. The model will need to be retrained on a different hero, if the target changes. The model can successfully distinguish the hero from other human subjects. 

 


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

The **middle layer** is a 1x1 convolutional layer with a kernel size of 1 and a stride of 1. If we wish to feed the output of a convolutional layer into a fully connected layer, we flatten it into a 2D tensor. This results in the loss of spatial information, because pixel location information is lost. This problem is eliminated by introducing this 1x1 convolution.

We can avoid that by using 1x1 convolutions.

**Decoder layers** provide upsampling encoded layers back to higher dimension. The method used for achieving upsampling in this process is called *bilinear upsampling.* Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. 

In all these layers, we use **batch normalization**. This means that instead of just normalizing the inputs to the network, we normalize the inputs to each layer within the network. During training, each layer's inputs are normalized using the mean and variance of the values in the current mini-batch. The benefits of doing this includes higher learning rates, faster learning, and an element of regularization for the network parameters.  

Some **skip connections** are introduced between the encoder layer and the decoder layer to improve the resolution of the results. 

The figure below shows the architecture of the network implemented for this model, where E1,E2 and E3 denote the three encoder layers, while D1, D2 and D3 denote the three decoder layers. MID denotes the 1x1 convolutional layer. 
 
![alt text][image_1]

The following codeblock creates the architecture of the network

```python

def fcn_model(inputs, num_classes):
    f = 32
    s = 2
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    print("inputs: ")
    print(inputs.get_shape())
    encoder_layer1 = encoder_block(inputs, f, strides=s)
    print("encoder_layer1")
    print(encoder_layer1.get_shape())
    encoder_layer2 = encoder_block(encoder_layer1, filters=2*f, strides=s)
    print("encoder_layer2")
    print(encoder_layer2.get_shape())
    encoder_layer3 = encoder_block(encoder_layer2, filters=4*f, strides=s)
    print("encoder_layer3")
    print(encoder_layer3.get_shape())
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encoder_layer3,filters= 4*f, kernel_size=1, strides=1)
    print("conv_layer")
    print(conv_layer.get_shape())
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    print("decoder_layer1")
    decoder_layer1 = decoder_block(conv_layer,encoder_layer2,4*f)
    print(decoder_layer1.get_shape())
    
    decoder_layer2 = decoder_block(decoder_layer1,encoder_layer1,2*f)
    print("decoder_layer2")
    print(decoder_layer2.get_shape())
    decoder_layer3 = decoder_block(decoder_layer2,inputs,f)
    print("decoder_layer3")
    print(decoder_layer3.get_shape())
    x =decoder_layer3
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)

``` 

## Training, Predicting and Scoring ##

### Model Hyperparameters

Following are the hyperparameters of the model that need to be tuned in order to get good predictive performance from it. 

**Optimizer algorithm**

Several gradient descent-based optimizers are available in the keras library. Due to the large size of the data set, it makes sense to use a mini-batch gradient descent method. The figure below is referenced from     
[(Ref)](http://forums.fast.ai/t/how-do-we-decide-the-optimizer-used-for-training/1829/6) and is a good summary of the different algorithms available. 

![alt text][image_22]

[This reference](http://ruder.io/optimizing-gradient-descent/index.htm) provides more details on several of these  algorithms. In my experiments, I used `Adam` and `Nadam` algorithms. Both of these use the concept of `momentum` to speed up convergence. 

Both these algorithms are `Adaptive Gradient Algorithms` (AdaGrad), since they maintain a *per-parameter* learning rate. This improves performance on problems with sparse gradients (e.g. natural language and computer vision problems). 

Another characteristic is `Root Mean Square Propagation` (RMSProp), which also maintains per-parameter learning rates, adapted based on the average of recent magnitudes of the gradients for the weight. This makes the algorithm suitable for online and non-stationary problems (e.g. noisy). [(Ref)](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)


In addition to all characteristics of `Adam`, `Nadam` also uses Nesterov accelerated gradient (`NAG`) to slow down momentum by using expected future position of the parameters. 


**Learning rate** 

An optimal learning rate is important for fast and stable convergence. Two issues with a small learning rate are (1) slow convergence and (2) possibility of getting stuck on a local minimum. Too fast a learning rate will cause instabilities. In `keras`, the default learning rate for `Adam` is `0.001` and for `Nadam` is `0.002`. In my experiments, I also tried a learning rate of `0.004` for `Adam`. 

Other parameters of `Adam` and `Nadam` include `beta_1`, `beta_2`, `epsilon`, `decay` / `schedule_decay`. [Keras documentation](https://keras.io/optimizers) provide more details for these parameters. All these parameters were left to their default values in my experiments.   

**Batch size**

Training the neural network with all data at once may be infeasible, since larger the dataset, the more memory space one needs. 
If there are `m` examples in a batch, we need an `O(m)` computation and use `O(m)` memory, but we reduce the amount of uncertainty in the gradient by a factor of only `O(sqrt(m))`. Therefore there are diminishing marginal returns to putting more examples in the batch. In order to be able to use all samples, the number of steps needed is inversely proportional to the step size. Therefore, there is an optimal batch size to be used for efficient training of the network. In my experiments, I tried a number of batch sizes ranging from 50 to 25, and the best results were obtained for a batch size of 30.  

**Number of epochs**

This refers to the number of times the algorithm sees the entire data set. Large number of epochs has two disadvantages: (1) additional time is required and (2) may lead to overfitting. However, too few epochs will lead to underfitting. Therefore, there's an optimal value for number of epochs. I tried values between 20 and 45. The best result was with number of epochs = 40. 

**Steps per epoch**

In order to be able to use all training data available, steps per epoch is simply the size of the training dataset divided by batch size. In my experiments, the training dataset contains 4131 images. The number of steps per epoch is calculated accordingly. 

**Validation steps per epoch**

In order to be able to use all validation data available, validation steps per epoch is simply the size of the validation dataset divided by batch size. In my experiments, the training dataset contains 1184 images. The number of validation steps per epoch is calculated accordingly. 


**Number of workers**

This refers to the number of parallel threads that will be spawned to train the model. The experiment was run on a `p2.xlarge` instance on the Amazon Web Server. It's assumed that this has two processors. Therefore, any attempt to parallelize the training with a number of workers greater than 2 will not lead to any speedup. In fact, it will slow down the computations due to overheads. Therefore, the number of workers was fixed to 1. 
   

### Parameter tuning

In this project, parameters were tuned by brute force until a good enough model was obtained. The table below shows five experiments that were run with different values of hyperparameters. It was found that Experiment 3 has the best overall score, although Experiment 4 and 5 are close.

**Scoring**

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. Intersection over union is a useful metric for semantic segmentation tasks. It is the ratio between the area of overlap between the prediction and the ground truth, and the area of union. Perfect prediction will lead to an IoU of 1. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

The columns of the table corresponding to the hyperparameters were explained above. The columns related to the results of the experiment are explained below. 

Scores for while the quad is following behind the target.

* `iou1o`  -  average intersection over union for other people
* `iou1h`  -  average intersection over union for the hero

Scores for images while the quad is on patrol and the target is not visible
 
* `iou2o` -  average intersection over union for other people
* `iou2h` -  average intersection over union for the hero

This score measures how well the neural network can detect the target from far away 
* `iou3o` -  average intersection over union for other people
* `iou3h` -  average intersection over union for the hero
 
* `finalIOU` -  IoU for the subset of dataset that definitely includes the hero
* `finalScore` - This is `finalIOU` times `weight`, where `weight = all true positives / (all true positives + all false positives and negatives)`


| Experiment # | Learning rate |	Batch size |	Optimizer |	number of epochs	| Steps per epoch |	validation steps per epoch |	iou1o | iou1h |	iou2o |	iou2h |	iou3o |	iou3h	| finalIOU |	finalScore |
| --- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|1 |0.004 |	50 |	Adam |	20 |	83 |	24 |	0.29 |	0.81 |	0.71 |	0 |	0.38 |	0.11 |	0.46 |	0.33 |
|2 |0.002 |	30 |	Adam |	40 |	138 |	39 |	0.34 |	0.87 |	**0.76** |	0 |	0.43 |	0.17 |	0.52 |	0.39 |
|3 |0.002 |	30 |	Nadam |	40 |	138 |	39 |	**0.37** |	**0.91** |	**0.76** |	0 |	**0.45** |	0.25 |	0.58 |	**0.44** |
|4 |0.002 |	45 |	Nadam |	40 |	92 |	26 |	0.36 |	**0.91** |	0.73 |	0 |	0.44 |	0.24 |	0.58 |	0.43 |
|5 |0.002 |	25 |	Nadam |	45 |	165 |	47 |	0.32 |	**0.91** |	0.70 |	0 |	0.39 |	**0.27** |	**0.59** |	0.43 |

The model of Experiment 3 is chosen as the final model. This model has a final score of 0.44, which is greater than the requirement of 0.4. 

The images below are organized in the following way. For each experiment, the convergence history is shown first, with number of epochs on the x-axis and the loss function for both training and validation on the y-axis. This is followed by a sample of results from the experiments described above. On the left is the input sample image, taken by the drone, the middle image is the labeled ground truth, and the right image is the output of the model. The hero is labeled in blue and other people are labeled in green. Red represents the background.  
 
**Experiment 1**

![alt text][image_2]
![alt text][image_3]
![alt text][image_4]
![alt text][image_5]
![alt text][image_6]

**Experiment 2**

![alt text][image_7]
![alt text][image_8]
![alt text][image_9]

**Experiment 3**

![alt text][image_10]
![alt text][image_11]
![alt text][image_12]
![alt text][image_13]

**Experiment 4**

![alt text][image_14]
![alt text][image_15]
![alt text][image_16]
![alt text][image_17]
![alt text][image_18]

**Experiment 5**

![alt text][image_19]
![alt text][image_20]
![alt text][image_21]

### Limitations and Future Work ###

Following are some of the limitations of the model, and the corresponding future work that could help overcome these limitations.  

**Model accuracy**

The best overall score obtained in the above experiments was 0.44, leaving room for improvements. Following are some approaches that could be tried as future work: 

* More systematic hyperparameter tuning. Optimize hyperparameters using a systematic optimization approach, not unlike the approach used for learning actual model parameter
* Collect more data: Expand the training dataset by including more images. Make this more targeted by focusing on the metric in which the model is weak, such as `iou3h`. Then, retrain the model. 
*  With increased dataset size, try models with more hidden layers, to see if better accuracy could be obtained without overfitting. 

**Model capability**

The model is tasked with semantic segmentation with three classes: background, hero, other people. This model is unable to recognize objects such as dog, cat, car, etc. instead of a human. It would be interesting to extend this to more classes: trees, grass, building, street, vehicle, etc. This might require a lot more data collection and a model with more layers and/or training effort. This could be a step towards a fully autonomous drone.   
