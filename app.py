import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the "Brain" and the Face Detector
model = load_model('emotion_model.h5')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Define the Emotion Labels (Must be in the same order as the dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 3. Open the Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Crop and Preprocess the face for the model
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0  # Normalize
            roi = np.expand_dims(roi, axis=0)       # Add batch dimension
            roi = np.expand_dims(roi, axis=-1)      # Add channel dimension

            # Make the prediction
            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            
            # Display the label on the screen
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()