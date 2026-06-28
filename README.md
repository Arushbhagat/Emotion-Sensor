# Real-Time Emotion Detection using Deep Learning

## Overview

This project is a real-time facial emotion recognition system developed using **TensorFlow/Keras** and **OpenCV**. A Convolutional Neural Network (CNN) was trained on the **FER-2013** facial expression dataset using **Google Colab** and later integrated into a desktop application developed in **VS Code** for real-time emotion prediction through a webcam.

The application detects a human face, preprocesses the image, predicts the facial emotion, and displays the detected emotion live on the video feed.

---

## Features

* Real-time webcam emotion detection
* Face detection using OpenCV Haar Cascade
* CNN-based facial emotion classification
* Live prediction of facial expressions
* TensorFlow/Keras model integration
* Lightweight desktop application
* Fast real-time inference

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Google Colab
* Visual Studio Code

---

## Dataset

The model was trained using the **FER-2013 (Facial Expression Recognition 2013)** dataset.

Dataset Characteristics:

* 48 × 48 grayscale facial images
* 7 emotion classes
* Thousands of labeled facial expression images

---

## Detected Emotions

The model predicts the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## Model Architecture

The emotion classifier is a custom **Convolutional Neural Network (CNN)** built using TensorFlow/Keras.

The architecture includes:

* Convolutional (Conv2D) Layers
* Batch Normalization
* Max Pooling Layers
* Dropout Layers
* Flatten Layer
* Fully Connected (Dense) Layers
* Softmax Output Layer

---

## Training Pipeline

The model was trained using **Google Colab** with GPU acceleration.

Training pipeline included:

* Loading the FER-2013 dataset
* Image preprocessing
* Pixel value normalization (1/255)
* Data augmentation
* Training and validation split
* CNN model training
* Model evaluation
* Exporting the trained model as `emotion_model.h5`

---

## Training Configuration

* Framework: TensorFlow / Keras
* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Validation Split: 20%
* Input Image Size: 48 × 48
* Image Format: Grayscale

---

## Project Structure

```
Emotion-Sensor/
│
├── app.py
├── emotion_model.h5
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebook/
   └── emotionsensor.ipynb
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Arushbhagat/Emotion-Sensor.git
```

Move into the project folder:

```bash
cd Emotion-Sensor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## How It Works

1. The webcam captures live video frames.
2. OpenCV detects faces using a Haar Cascade classifier.
3. The detected face is converted to grayscale.
4. The face image is resized to **48 × 48** pixels.
5. The trained CNN predicts the emotion.
6. The predicted emotion is displayed in real time on the video stream.



## Future Improvements

* Multi-face emotion detection
* Emotion confidence percentage
* Emotion history logging
* Database integration
* PyQt6/CustomTkinter GUI
* FastAPI REST API
* TensorFlow Lite optimization
* Performance benchmarking

---

## Learning Outcomes

Through this project, I gained practical experience with:

* Computer Vision
* Deep Learning
* Convolutional Neural Networks
* Image Preprocessing
* TensorFlow/Keras
* OpenCV
* Model Training using Google Colab
* Real-Time AI Application Development

---

## Author

**Arush Bhagat**

GitHub: https://github.com/Arushbhagat
