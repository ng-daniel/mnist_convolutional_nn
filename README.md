# Simple Convolutional Neural Network in Pytorch
Training a CNN on the MNIST dataset in python using Pytorch

## Data

The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is a set of 70,000 28x28 pixel images of handwritten digits, split into 60,000 training and 10,000 testing images. The training and testing set are further split into batches of 1875 and ~312 batches respectively, with 32 images per batch (testing set does not divide evenly by 32). Though MNIST is an extensively researched dataset, it is still a great exercise for practicing convolutional neural network architecture and the machine learning pipeline.
<p align="center">
  <img src="https://github.com/user-attachments/assets/4cadecaf-9b3b-42f4-87bf-c73d48631ab7" alt="MNIST dataset sample" width=300 height=300/>
</p>

###### *25 random samples from the MNIST dataset along with their respective labels*

## Model Architecture

The model is based on the [VGG CNN architecture](https://arxiv.org/abs/1409.1556), which is composed of several convolution blocks and a fully connected linear layer at the very end. Each convolution block consists of 2-3 convolution layers, which learn features of the image, and a pooling layer, that reduces the size of the features in order to learn more general patterns. In the final step, all the feature information that the convolution blocks extracted is flattened into a one-dimensional tensor and passed through a standard fully connected linear layer to produce the output logits.

Because the initial size of our images is already so small, our modified VGG architecture begins at a humble 28x28px, which only lets us include two convolution blocks before we can't reduce the size any further.

The shape of the tensor as it passes through is commented throughout the VGG class in the `model.py` file, in case you are curious. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/d38825ba-bba8-4191-83e2-f8fc1c15331a" alt="CNN architecture" width=500 height=250/>
</p>

###### *Visualization of the VGG convolutional neural network, on a 224x224px image with 3 color channels.*

## Hyperparameters

- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 3
- \# of Filters per Convolution Layer: 16

## Training

## Results
![alt text](https://github.com/user-attachments/assets/f1f927e2-d379-46eb-afa4-1a76041f283c "Confusion matrix of the trained model")
*Confusion matrix of the trained model.*

## Conclusions

## How to run
