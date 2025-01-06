from flask import Flask, render_template, request, jsonify
from db import init_db, get_counts, insert_count
import threading
from opencv import run_opencv  # Import the OpenCV function
import socket

app = Flask(__name__)

# Initialize the database
init_db()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/db')
def db():
    return render_template('db.html')

@app.route('/count', methods=['GET', 'POST'])
def count():
    return render_template('count.html')

@app.route('/get_counts', methods=['GET'])
def get_counts_route():
    counts = get_counts()
    return jsonify(counts)

@app.route('/update_count', methods=['POST'])
def update_count():
    data = request.get_json()
    count = data.get('count')
    insert_count(count)
    return jsonify({'status': 'success'})

@app.route('/current_count', methods=['GET'])
def current_count():
    counts = get_counts()
    return jsonify({
        'current_count': counts['current_count'],
        'min_count': counts['min_count'],
        'max_count': counts['max_count'],
        'avg_count': counts['avg_count']
    })

if __name__ == '__main__':
    # Get the local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # Print the IP address and port
    print(f"Server running at http://{local_ip}:5000/")
    
    # Start the OpenCV function in a separate thread
    threading.Thread(target=run_opencv).start()
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0')