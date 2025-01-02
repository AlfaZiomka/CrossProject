# filepath: /Users/dolfijntje/Visualme/CrossProject/Michal/Website CrossWise/main.py
from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import random
import threading
import time
import cv2
import sys
import os

# Add the OpenCV-main directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'OpenCV-main')))

from opencv import run_opencv, generate_frames  # Import the OpenCV functions

app = Flask(__name__)
CORS(app)  # Enable CORS
socketio = SocketIO(app)

@app.route('/')
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/melding')
def melding():
    return render_template('melding.html')

@app.route('/kiosk')
def kiosk():
    return render_template('kiosk.html')

@app.route('/update_count', methods=['POST'])
def update_count():
    data = request.json
    count = data.get('count')
    socketio.emit('update_count', {'count': count}, broadcast=True)
    return '', 200

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('send_alert')
def handle_send_alert(data):
    emit('receive_alert', data, broadcast=True)

def simulate_bar_status():
    alert_sent = {'bar 1': False, 'bar 2': False, 'bar 3': False}
    last_drunk_time = {'bar 1': 0, 'bar 2': 0, 'bar 3': 0}
    cooldown_period = 30  # Cooldown periode van 30 seconden

    while True:
        current_time = time.time()
        bar_statuses = []

        for bar_id in ['bar 1', 'bar 2', 'bar 3']:
            if current_time - last_drunk_time[bar_id] < cooldown_period:
                status = random.choices(['Het wordt druk', 'Rustig'], weights=[3, 7])[0]
            else:
                status = random.choices(['Druk', 'Het wordt druk', 'Rustig'], weights=[1, 3, 7])[0]
                if status == 'Druk':
                    last_drunk_time[bar_id] = current_time

            bar_statuses.append((bar_id, status))

        for bar_id, status in bar_statuses:
            if status == 'Druk' and not alert_sent[bar_id]:
                bar_number = bar_id.split(' ')[1]  # Haal het cijfer van de bar op
                socketio.emit('send_alert', {'message': f'Te veel klanten bij bar {bar_number}'})
                socketio.emit('send_alert', {'message': 'Betalen bij de kassa'})  # Voeg melding toe
                alert_sent[bar_id] = True
            elif status != 'Druk' and alert_sent[bar_id]:
                alert_sent[bar_id] = False

        time.sleep(10)  # Verlaag de frequentie van statusupdates

if __name__ == '__main__':
    threading.Thread(target=simulate_bar_status).start()
    threading.Thread(target=run_opencv).start()  # Start the OpenCV function in a separate thread
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)