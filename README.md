# Simple Convolutional Neural Network in PyTorch
Training a CNN on the MNIST dataset in Python.\
Tools used: `Python, PyTorch, Sci-kit Learn, Torchmetrics, Jupyter Notebooks, VsCode, TQDM (for training progress bar)`

## Data

The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is a set of 70,000 28x28 pixel images of handwritten digits, split into 60,000 training and 10,000 testing images. Though MNIST is an extensively researched dataset, it is still a great exercise for practicing convolutional neural network architecture and the machine learning pipeline.
<p align="center">
  <img src="https://github.com/user-attachments/assets/4cadecaf-9b3b-42f4-87bf-c73d48631ab7" alt="MNIST dataset sample" width=300 height=300/>
</p>

###### *25 random samples from the MNIST dataset along with their respective labels.*

## Model Architecture

The model is based on the [VGG CNN architecture](https://arxiv.org/abs/1409.1556), which is composed of several convolution blocks and a fully connected linear layer at the very end. Each convolution block consists of 2-3 convolution layers, which learn features of the image, and a pooling layer, that reduces the size of the features in order to learn more general patterns. In the final step, all the feature information that the convolution blocks extracted is flattened into a one-dimensional tensor and passed through a standard fully connected linear layer to produce the output logits.

Because the initial size of our images is already so small, our modified VGG architecture begins at a humble 28x28px, which only lets us include two convolution blocks before we can't reduce the size any further.

In case you are curious, the shape of the tensor as it passes through is commented between the layers throughout the VGG class in the `model.py` file. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/d38825ba-bba8-4191-83e2-f8fc1c15331a" alt="CNN architecture" width=500 height=250/>
</p>

###### *Visualization of the VGG convolutional neural network, on a 224x224px image with 3 color channels.*

## Training

The training and testing datasets were split into 1875 and 313 (the testing set does not divide evenly by 32) batches, respectively, with 32 images per batch. The model is trained for 5 epochs, with each epoch iterating through the data and updating the model parameters after each training batch. 

As this is a multiclass classification problem, the ideal loss function is Cross-Entropy Loss, which penalizes confident misclassification and makes it well-suited for classification. The optimization method used is Stochastic Gradient Descent for its overall reliability and faster computation speed with larger datasets.

###### *Fun / Not Fun Fact: This project was originally just supposed to be in Google Colab, but because their free version got worse recently (they kicked me out of my session after just one hour!!! multiple times!!!) I decided to just set up a conda virtual environment for PyTorch on my machine, which stole an entire afternoon + evening from me :(.*

## Results

As expected of this dataset, the model performed quite well:


<p align="center">
  <img src="https://github.com/user-attachments/assets/2bdcb456-17c5-463a-9091-f678fc2767f1" alt="final loss" width=350 height=60/>
</p>

###### *Final loss and accuracy metrics for the MNIST dataset convolutional neural network. Very nice!*

<p align="center">
  <img src="https://github.com/user-attachments/assets/0b061c9e-4cba-44ec-a7ad-76c230d3d7e8" alt="confusion matrix" width=350 height=650/>
  <img src="https://github.com/user-attachments/assets/e5056c48-c2f8-4441-ae79-984498e002f2" alt="confusion matrix" width=350 height=650/>
</p>

###### *(Left) Jupyter notebook training cell output*
###### *(Right) Graphs displaying train/test loss/accuracy over the 5 epochs it trained for. It seems the model had an epiphany between epochs 1 and 2.*


<p align="center">
  <img src="https://github.com/user-attachments/assets/c3e6c226-7b22-473b-a0b1-0d77f246f5e2" alt="confusion matrix" width=350 height=300/>
</p>

###### *Confusion matrix of the trained model. Very nice! The model is most confident in predicting the number 1 since it's often just a straight line.*

<p align="center">
  <img src="https://github.com/user-attachments/assets/f5549cad-494e-4f7e-8b60-fcd94d1c2912" alt="confusion matrix" width=800 height=300/>
</p>

###### *Program output when running `train.py` and saving the model to the `models` directory. Results differ slightly from the notebook model due to inherent unpredictability in the training process.* 

## Conclusions
By learning how to build, train, and evaluate a neural network from the ground up, I've gained the foundational machine learning skills and knowledge necessary to pursue more complex and impact-driven endeavors. 

Though this was a fun project, learning everything by myself through online resources was extremely difficult and took up a lot of my time (way too much time). I look forward to continuing my learning by putting my abilities to practical use through independent data science projects and, more importantly, working with teams to solve real problems with machine learning. 

It's not often I feel this way, but I'm really excited about what the future has in store!
