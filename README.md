# Real-Time Emotion Detection using Deep Learning

## Overview

A real-time facial emotion recognition system built using TensorFlow/Keras and OpenCV. The model was trained on the FER-2013 dataset and integrated into a desktop application for live emotion prediction through a webcam.

## Features

- Real-time webcam emotion detection
- Face detection using OpenCV
- CNN-based emotion classification
- Live emotion prediction
- Pretrained model (.h5) integration
- Fast real-time inference

## Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Google Colab (Model Training)
- VS Code (Application Development)

## Dataset

**FER-2013 (Facial Expression Recognition 2013)**

The model was trained on the FER-2013 dataset containing grayscale facial images classified into seven emotions.

## Detected Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model Details

- Model Type: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input Size: 48 × 48 Grayscale Images
- Output Classes: 7

## Training

The model was trained using Google Colab.

Training included:

- Image preprocessing
- Normalization
- Data augmentation
- CNN model training
- Model evaluation
- Saving trained model as `emotion_model.h5`

## Project Structure

```
Emotion-Sensor/
│
├── app.py
├── emotion_model.h5
├── requirements.txt
├── README.md
├── .gitignore
└── Emotion_Training.ipynb
```

## Installation

Clone the repository

```bash
git clone https://github.com/Arushbhagat/Emotion-Sensor.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python app.py
```

## Screenshots

(Add screenshots here after uploading them.)

## Future Improvements

- Multi-face detection
- Emotion confidence percentage
- Modern GUI
- Emotion history
- Database integration
- REST API using FastAPI

## Author

**Arush Bhagat**

GitHub: https://github.com/Arushbhagat
