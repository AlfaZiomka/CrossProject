import sys
import os
import threading
import socket
import logging
import sqlite3
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, Response
from flask_socketio import SocketIO, emit
import time  # Import time module for adding delay

# Initialize the database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_count (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    count INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def insert_count(count):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO user_count (count) VALUES (?)', (count,))
    conn.commit()
    conn.close()

def get_counts():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT count FROM user_count')
    counts = [row[0] for row in c.fetchall()]
    conn.close()
    
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        avg_count = round(sum(counts) / len(counts))  # Round the average count
        current_count = counts[-1]
    else:
        min_count = max_count = avg_count = current_count = 0

    return {
        'min_count': min_count,
        'max_count': max_count,
        'avg_count': avg_count,
        'current_count': current_count
    }

# Initialize the database
init_db()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__, template_folder='Templates', static_folder='static')
app.secret_key = 'your_secret_key'
socketio = SocketIO(app)

# Create users table for CrossWise app
def create_users_table():
    db_path = os.path.join(os.path.dirname(__file__), 'users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_user(username):
    db_path = os.path.join(os.path.dirname(__file__), 'users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(username, password):
    db_path = os.path.join(os.path.dirname(__file__), 'users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()

# Create the users table
create_users_table()

# Routes for the app
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        logging.debug(f"Attempting login for user: {username}")
        if user:
            logging.debug(f"User found: {user}")
        if user and user[2] == password:  # Assuming the password is stored in the third column
            session['logged_in'] = True
            session['username'] = username
            logging.debug(f"Login successful for user: {username}")
            return redirect(url_for('dashboard'))
        else:
            logging.debug(f"Login failed for user: {username}")
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/kiosk')
def kiosk():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('kiosk.html')

@app.route('/melding')
def melding():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('melding.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if get_user(username):
            return render_template('register.html', error='User already exists')
        create_user(username, password)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/db')
def db():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('db.html')

@app.route('/count', methods=['GET', 'POST'])
def count():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('count.html')

@app.route('/get_counts', methods=['GET'])
def get_counts_route():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    counts = get_counts()
    return jsonify(counts)

@app.route('/update_count', methods=['POST'])
def update_count():
    try:
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        data = request.get_json()
        count = data.get('count')
        insert_count(count)
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error in update_count: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/current_count', methods=['GET'])
def current_count():
    try:
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        counts = get_counts()
        return jsonify({
            'current_count': counts['current_count'],
            'min_count': counts['min_count'],
            'max_count': counts['max_count'],
            'avg_count': counts['avg_count']
        })
    except Exception as e:
        logging.error(f"Error in current_count: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chart')
def db_chart():
    return render_template('chart.html')

@app.route('/send_alert', methods=['POST'])
def send_alert():
    data = request.get_json()
    message = data.get('message')
    socketio.emit('receive_alert', {'message': message})
    return jsonify({'status': 'success'})

# OpenCV and YOLO setup
logging.basicConfig(level=logging.DEBUG)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
video_path = "rtsp://Groepje6:bingchillin420@192.168.0.101:554/stream1"  # Use 0 for the default camera
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    logging.error(f"Error opening camera with URL: {video_path}")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)
model = YOLO('yolov8n.pt').to(device)
backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
ret, frame = cap.read()
if not ret:
    logging.error("Error reading frame from camera")
    exit()
original_size = frame.shape[1], frame.shape[0]
downscale_size = (640, 360)
prev_gray = cv2.cvtColor(cv2.resize(frame, downscale_size), cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
previous_people_count = 0

def send_message_to_website(count):
    url = "http://127.0.0.1:5004/send_alert"
    data = {"message": f"There are {count} people at the bar."}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        logging.info("Message sent successfully")
    else:
        logging.error("Failed to send message")

def send_human_count_to_db(count):
    url = 'http://127.0.0.1:5004/update_count'
    data = {'count': count}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            logging.info(f"Successfully sent human count: {count}")
        else:
            logging.error(f"Failed to send human count: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending human count: {e}")

def generate_video_feed():
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Could not read frame from video stream")
                continue  # Skip this iteration and try again

            # Apply the humanoid filter
            frame_resized = cv2.resize(frame, downscale_size)
            results = model(frame_resized)
            for result in results:
                for box in result.boxes:
                    if box.cls == 0:  # Class 0 is 'person' in YOLO
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                        # Blur the entire body
                        body = frame_resized[y1:y2, x1:x2]
                        body = cv2.GaussianBlur(body, (99, 99), 30)
                        frame_resized[y1:y2, x1:x2] = body

            # Encode the frame
            _, buffer = cv2.imencode('.jpg', frame_resized)
            frame = buffer.tobytes()

            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            time.sleep(5)  # Add a delay of 5 seconds

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            continue  # Skip this iteration and try again

def start_opencv():
    global prev_gray, previous_people_count  # Declare prev_gray and previous_people_count as global to modify them inside the function
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error reading frame from camera")
                continue  # Skip this iteration and try again

            # Resize frame
            frame_resized = cv2.resize(frame, downscale_size)

            # Apply YOLO model to detect people
            results = model(frame_resized)
            people_count = 0
            for result in results:
                for box in result.boxes:
                    if box.cls == 0:  # Class 0 is 'person' in YOLO
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                        people_count += 1
                        # Blur the entire body
                        body = frame_resized[y1:y2, x1:x2]
                        body = cv2.GaussianBlur(body, (99, 99), 30)
                        frame_resized[y1:y2, x1:x2] = body

            logging.debug(f"Detected people count: {people_count}")

            if people_count != previous_people_count:
                send_human_count_to_db(people_count)
                previous_people_count = people_count

            prev_gray = gray

            # Encode the frame
            _, buffer = cv2.imencode('.jpg', frame_resized)
            frame = buffer.tobytes()

            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            time.sleep(5)  # Add a delay of 5 seconds

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            continue  # Skip this iteration and try again

@socketio.on('connect')
def handle_connect():
    logging.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected')

@socketio.on('send_alert')
def handle_send_alert(data):
    logging.info(f"Received alert: {data}")
    socketio.emit('receive_alert', data)

if __name__ == '__main__':
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Server running at http://{local_ip}:5004/")
    opencv_thread = threading.Thread(target=start_opencv)
    opencv_thread.daemon = True
    opencv_thread.start()
    socketio.run(app, debug=True, host='0.0.0.0', port=5004)