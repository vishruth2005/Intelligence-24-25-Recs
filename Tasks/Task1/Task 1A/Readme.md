# Deepfake Detection Using Convolutional Neural Networks (CNN)

This project aims to classify images as either **real** or **fake** (deepfake) using a Convolutional Neural Network (CNN) model trained on augmented images. The solution includes image preprocessing, augmentation, and a CNN model for binary classification. The project is built using **TensorFlow** and **Keras** libraries.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Results](#results)

---

## Overview

The goal of this project is to detect deepfake images, which have become a significant concern due to their ability to manipulate facial features convincingly. By training a CNN on a dataset of real and fake images, this project identifies fake images by extracting relevant features using convolutional layers.

**Key Steps Involved:**

- Data loading and preprocessing.
- Data augmentation (to increase the model's generalizability).
- CNN model construction and training.
- Model evaluation and prediction.
- Storing predictions in a CSV file.

---

## Data Preprocessing and Augmentation

We use the **ImageDataGenerator** class to apply data augmentation techniques to enhance the model’s ability to generalize. Data augmentation is essential to simulate various conditions that the model might encounter in real-world scenarios.

**Augmentation Techniques Applied:**

- Rotation (up to 40 degrees)
- Width and height shifts (up to 20%)
- Shear and zoom transformations (up to 20%)
- Horizontal flipping
- Rescaling pixel values between 0 and 1

A validation split of 20% of the training data is used to assess the model’s performance during training.

---

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** designed to extract features from images and classify them as **real** or **fake**. The architecture consists of the following layers:

1. **Three convolutional layers**:

   - Initialized with **He normal** for better performance.
   - Each convolutional layer is followed by an activation function (ReLU) to introduce non-linearity.

2. **MaxPooling layers**:

   - Applied after each convolutional layer to down-sample the spatial dimensions and reduce the number of parameters.

3. **Flatten layer**:

   - Converts the 2D matrices into a 1D vector to prepare for the fully connected layers.

4. **Dropout layer**:

   - Dropout rate set to **50%** to prevent overfitting by randomly setting half of the inputs to 0 during training.

5. **Two dense layers**:
   - First dense layer with **128 neurons** and **ReLU activation**.
   - A final output dense layer with a **sigmoid activation** for **binary classification** (real or fake).

---

## Training

The model is trained using the `fit` function with the following specifications:

- **Training data generator**:

  - Real-time data augmentation is applied to improve the model's generalization ability. This includes random rotations, shifts, shearing, zooming, and horizontal flips.

- **Early stopping**:

  - Early stopping is implemented to prevent overfitting. The training process stops when the validation loss stops improving, and the best model weights (based on validation loss) are restored.

- **Validation data**:
  - Validation data is used to monitor the model's performance after each epoch, ensuring the model isn't overfitting to the training data.

---

## Evaluation

After training, the model is used to predict the class of unseen images (real or fake) from the test dataset. The predicted classes are stored in a CSV file (`output_tf2.csv`) with the following columns:

- **ID**: The unique identifier for each image.
- **TARGET**: The predicted class, where `0` represents a real image and `1` represents a fake image.

At the beginning the labeling had been taken wrong ,ie ,`0` for fake and `1` for real so to correct that the labels were inverted by using the logic result=1-result.

---

## Results

The model's accuracy and loss are tracked across both the training and validation datasets. These metrics provide insights into how well the model can differentiate between real and fake images.

- **Training Accuracy**: 96.88%
- **Validation Accuracy**: 98.75%
- **F1 Score in Kaggle Competition**: 0.94166

The final CSV file (`output_tf2.csv`) contains the test image IDs and their corresponding predicted classes:

- `0` for real images.
- `1` for fake images.

---
