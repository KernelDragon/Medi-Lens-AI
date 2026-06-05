# Smart Medicine Assistant

## Overview

The Smart Medicine Assistant is a machine learning-based application designed to identify medications from images. The project utilizes a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify pills, aiming to reduce medication errors. It also includes a web interface built with Streamlit for easy user interaction.

Currently, the model is configured to classify three specific types of medication:

* Paracetamol
* Vitamin C
* Multivitamin

## Repository Structure

The project is divided into three primary Python scripts:

* `train.py`: Contains the data loading pipeline, defines the CNN architecture, and handles the training process.
* `predict.py`: A standalone inference script to test the trained model on a single image. It processes the image, outputs the predicted class, confidence score, and associated medical information (e.g., usage and dosage).
* `app.py`: The frontend application built using Streamlit. It provides a graphical user interface where users can upload an image and view the model's prediction.

## Prerequisites

To run this project, you need Python installed on your system along with the following dependencies. You can install them using pip:

```bash
pip install tensorflow streamlit numpy pillow

```

## Usage Instructions

### 1. Training the Model

Before running the application or prediction scripts, you must train the model or have a pre-trained model saved as `pill_model.h5`.

To train the model, ensure your dataset is organized in directories corresponding to the class names within the root folder, then run:

```bash
python train.py

```

### 2. Testing via Command Line

To quickly test the model's accuracy on a single image without launching the web application, place an image named `test.jpg` in the project directory and execute the prediction script:

```bash
python predict.py

```

This will output the predicted medicine name, the confidence percentage, and the hardcoded medical information to your terminal.

### 3. Running the Web Application

To launch the interactive web interface, use the Streamlit CLI command:

```bash
streamlit run app.py

```

This will start a local server and open the graphical interface in your default web browser.

## Model Architecture

The underlying image classification model is a Sequential Convolutional Neural Network processing images at a 224x224 resolution. The architecture includes:

* An initial rescaling layer to normalize pixel values (1./255).
* Three Convolutional (Conv2D) layers using ReLU activation, progressively increasing in filter size (16, 32, 64).
* MaxPooling2D layers following each convolutional layer to reduce spatial dimensions.
* A Flatten layer to convert the 2D matrices into a 1D vector.
* A fully connected Dense layer with 128 units and ReLU activation.
* A final Dense output layer using Softmax activation to generate probability distributions across the defined classes.
