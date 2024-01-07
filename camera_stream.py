import cv2
import numpy as np
from flask import Flask, Response, render_template
from keras.models import load_model
app = Flask(__name__)

# Load the trained model
model = load_model('emotion_detection_model.h5')  # Replace 'path_to_saved_model' with the actual path to your saved model

# Define the emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a VideoCapture object to access the camera
cap = cv2.VideoCapture(0)  # Change the parameter to the appropriate camera index if multiple cameras are connected

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no faces are detected, display "No face found" on the frame
        if len(faces) == 0:
            cv2.putText(frame, "No face found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Preprocess the face region
                face_roi = gray[y:y+h, x:x+w]
                resized = cv2.resize(face_roi, (48, 48))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))

                # Make predictions
                result = model.predict(reshaped)
                emotion_index = np.argmax(result)
                emotion_label = emotion_labels[emotion_index]

                # Display the emotion label on the frame
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convert the buffer to bytes and yield as a response
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
