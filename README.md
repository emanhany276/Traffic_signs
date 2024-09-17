# Traffic Signs Classification Using Convolutional Neural Networks (CNN)

## Introduction

Traffic sign recognition is a vital component of Advanced Driver Assistance Systems (ADAS) and autonomous driving technologies. Accurate detection and classification of traffic signs enable vehicles to make informed decisions, enhancing both safety and efficiency on the road. This project leverages deep learning techniques, specifically Convolutional Neural Networks (CNNs), to develop a model capable of recognizing and classifying various traffic signs from images.

## Dataset

- **Path:** `myData/`
- **Labels File:** `labels.csv`
- **Classes:** The dataset is organized into folders, each representing a specific traffic sign class. The `labels.csv` file contains the names of these classes.
- **Image Dimensions:** 32x32 pixels with 3 color channels (RGB).

### Dataset Split

- **Training Set:** 60% of the data
- **Validation Set:** 20% of the remaining data
- **Test Set:** 20% of the total data

## Model Architecture

The CNN model for this project consists of the following layers:

1. **Convolutional Layers:** Extract features from the input images using multiple convolutional layers. Each layer uses ReLU (Rectified Linear Unit) activation functions to introduce non-linearity and learn complex features.

2. **Pooling Layers:** Reduce the spatial dimensions of the feature maps, helping the model to focus on the most important features and reducing the computational load.

3. **Fully Connected Layers:** Combine the features extracted by the convolutional and pooling layers to classify the images into one of the traffic sign classes.

4. **Output Layer:** Produces the final classification results by applying a softmax activation function to provide probabilities for each class.

## Usage

1. **Prepare the Dataset:**

   Ensure that the dataset is properly organized into folders and the `labels.csv` file is in place.

2. **Train the Model:**

    Run the training script to start the training process. This will involve loading the dataset, building the CNN model, and training it using the training and validation sets.

3. **Evaluate the Model:**

    After training, use the test set to evaluate the model's performance and make predictions on unseen data.


4. **Make Predictions:**

    Use the trained model to classify new traffic sign images.




