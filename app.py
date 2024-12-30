import csv
from flask import Flask, render_template, request, jsonify
import cv2
import os

import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import threading

app = Flask(__name__)

# Global paths and variables
IMAGE_DIR = 'images'
MODEL_PATH = 'trained_model.pkl'
ATTENDANCE_FILE = 'attendance.csv'
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load or initialize attendance file
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv(ATTENDANCE_FILE, index=False)

# Step 1: Capture Images with Threading
def capture_images_thread(name):
    camera = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not access the camera")
        return jsonify({'status': 'Error: Could not access the camera'}), 400
    
    count = 0
    user_folder = os.path.join(IMAGE_DIR, name)
    os.makedirs(user_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while count < 10:  # Capture 10 images
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Detect faces in the captured frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("Warning: No face detected, please capture an image of your face.")
            continue  # Skip image capture if no face is detected

        img_path = os.path.join(user_folder, f'{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1
        # Optional: Show the captured frame
        cv2.imshow("Capturing Images", frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    camera.release()
    cv2.destroyAllWindows()

# Step 2: Train Model
def train_model():
    data = []
    labels = []
    label_map = {}
    current_label = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for name in os.listdir(IMAGE_DIR):
        user_folder = os.path.join(IMAGE_DIR, name)
        if not os.path.isdir(user_folder):
            continue

        for file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Detect face in the image
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:  # Proceed only if a face is detected
                    data.append(image)
                    labels.append(current_label)
                else:
                    print(f"Warning: No face detected in {img_path}, skipping training for this image.")
        
        label_map[current_label] = name
        current_label += 1

    if len(data) == 0:
        print("Error: No valid face images for training!")
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(data, np.array(labels))
    model.save(MODEL_PATH)

    with open('label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)

# Step 3: Mark Attendance
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    df = pd.read_csv(ATTENDANCE_FILE)
    
    # Create a new DataFrame with the new attendance record
    new_entry = pd.DataFrame({'Name': [name], 'Date': [date], 'Time': [time]})
    
    # Concatenate the new entry with the existing DataFrame
    df = pd.concat([df, new_entry], ignore_index=True)
    
    df.to_csv(ATTENDANCE_FILE, index=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    name = request.form['name']
    threading.Thread(target=capture_images_thread, args=(name,)).start()
    return jsonify({'status': 'Images capture started, check webcam for progress.'})

@app.route('/train', methods=['POST'])
def train():
    train_model()
    return jsonify({'status': 'Model trained successfully'})

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_route():
    camera = cv2.VideoCapture(0)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)
    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using OpenCV's Haar Cascade classifier (for better detection)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:  # No face detected
            return jsonify({'status': 'No face detected, cannot mark attendance'}), 404

        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            label, confidence = model.predict(face_region)

            if confidence < 90:  # Increased threshold for higher accuracy
                name = label_map[label]
                mark_attendance(name)
                return jsonify({'status': f'Attendance marked for {name}'}), 200
            else:
                return jsonify({'status': 'Face not recognized'}), 404
    
    camera.release()
    cv2.destroyAllWindows()

import pandas as pd

@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        
        if df.empty:
            return "No attendance records available."

        # Generate an HTML table
        table_html = '<table border="1" style="width:100%; text-align:left;">'
        table_html += '<tr><th>Name</th><th>Date</th><th>Time</th></tr>'
        for _, row in df.iterrows():
            table_html += f'<tr><td>{row["Name"]}</td><td>{row["Date"]}</td><td>{row["Time"]}</td></tr>'
        table_html += '</table>'

        return table_html
    except Exception as e:
        return f"Error: {str(e)}"






if __name__ == '__main__':
    app.run(debug=True) 