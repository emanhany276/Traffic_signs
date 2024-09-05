# Traffic_signs
Traffic signs classification using convolution neural networks CNN 


# Introduction
Traffic sign recognition is a crucial aspect of Advanced Driver Assistance Systems (ADAS) and autonomous driving. By accurately detecting and classifying traffic signs, vehicles can make informed decisions, improving safety and efficiency on the road. This project utilizes deep learning techniques, particularly CNNs, to develop a model capable of recognizing various traffic signs from images.

# Dataset
Path: myData/
Labels File: labels.csv
Classes: The dataset is organized into folders, each representing a specific traffic sign class. The labels file contains the names of these classes.
Image Dimensions: 32x32 pixels with 3 color channels (RGB).
The dataset is split into training, validation, and test sets:

Training Set: 60% of the data
Validation Set: 20% of the remaining data
Test Set: 20% of the total data
# Model Architecture
The CNN model consists of the following layers:

Convolutional Layers: Extracts features from the input images using multiple convolutional layers with ReLU activation.
MaxPooling Layers: Reduces the spatial dimensions of the feature maps, preventing overfitting.
Dropout Layers: Randomly drops units during training to prevent overfitting.
Flatten Layer: Flattens the input into a single vector.
Fully Connected Layers: Dense layers with ReLU activation, culminating in a softmax output layer that classifies the input image into one of the traffic sign categories.
Training Process
# Optimizer: Adam with a learning rate of 0.001.
Loss Function: Categorical Crossentropy.
Batch Size: 50 images per batch.
Epochs: 10 epochs with 2000 steps per epoch.
Image Augmentation: Applied to the training set to artificially expand the dataset by creating modified versions of images. This includes random width and height shifts, zooming, shearing, and rotations.
The model is trained using the augmented dataset, and the performance is evaluated on the validation set.

# Real-Time Traffic Sign Detection
After training, the model is deployed in a real-time environment where it can recognize traffic signs from a live camera feed. The camera captures frames, which are then preprocessed and fed into the model. The model predicts the traffic sign class and displays the result on the video feed along with the probability of the prediction.
