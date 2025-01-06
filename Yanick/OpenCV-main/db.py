import sqlite3
import time

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

def get_timestamps_and_counts():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT timestamp, count FROM user_count')
    data = c.fetchall()
    conn.close()
    
    timestamps = [time.strftime('%H:%M:%S', time.strptime(ts, '%Y-%m-%d %H:%M:%S')) for ts, _ in data]
    counts = [count for _, count in data]
    
    return timestamps, counts