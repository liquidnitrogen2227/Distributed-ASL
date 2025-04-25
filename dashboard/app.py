from flask import Flask, render_template, jsonify, request  # Added request import
import requests
import time
import threading
import logging
import os
import json
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional

app = Flask(__name__, static_folder='static', template_folder='templates')
logger = logging.getLogger(__name__)

# Configuration
LOAD_BALANCER_URL = os.environ.get('LOAD_BALANCER_URL', 'http://load_balancer:5000')
DB_PATH = os.environ.get('METRICS_DB', '/data/metrics.db')
METRICS_INTERVAL = int(os.environ.get('METRICS_INTERVAL', '5'))  # seconds

# Setup SQLite database for metrics
def setup_database():
    """Set up SQLite database for storing metrics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        timestamp REAL PRIMARY KEY,
        metrics_json TEXT
    )
    ''')
    
    # Create node metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS node_metrics (
        timestamp REAL,
        node_id TEXT,
        cpu_load REAL,
        memory_load REAL,
        response_time REAL,
        healthy INTEGER,
        PRIMARY KEY (timestamp, node_id)
    )
    ''')
    
    # Create request metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS request_metrics (
        timestamp REAL PRIMARY KEY,
        total_requests INTEGER,
        successful_requests INTEGER,
        failed_requests INTEGER,
        algorithm TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Metrics collection thread
def metrics_collector():
    """Thread to periodically collect and store metrics."""
    while True:
        try:
            # Get metrics from load balancer
            response = requests.get(f"{LOAD_BALANCER_URL}/metrics")
            
            if response.status_code == 200:
                metrics = response.json()
                timestamp = time.time()
                
                # Store in SQLite
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Store raw metrics
                cursor.execute(
                    "INSERT OR REPLACE INTO metrics VALUES (?, ?)",
                    (timestamp, json.dumps(metrics))
                )
                
                # Store node metrics
                for node_id, node_data in metrics.get('nodes', {}).items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO node_metrics VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            timestamp,
                            node_id,
                            node_data.get('cpu_load', 0),
                            node_data.get('memory_load', 0),
                            node_data.get('avg_response_time', 0),
                            1 if node_data.get('healthy', False) else 0
                        )
                    )
                
                # Store request metrics
                cursor.execute(
                    "INSERT OR REPLACE INTO request_metrics VALUES (?, ?, ?, ?, ?)",
                    (
                        timestamp,
                        metrics.get('requests_total', 0),
                        metrics.get('requests_success', 0),
                        metrics.get('requests_failed', 0),
                        metrics.get('algorithm', 'unknown')
                    )
                )
                
                conn.commit()
                conn.close()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        time.sleep(METRICS_INTERVAL)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/current-metrics')
def current_metrics():
    """Get current metrics from load balancer."""
    try:
        response = requests.get(f"{LOAD_BALANCER_URL}/metrics")
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f"Failed to get metrics: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical-metrics')
def historical_metrics():
    """Get historical metrics."""
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get time range from query params (default to last hour)
        hours = request.args.get('hours', 1, type=int)
        since = time.time() - (hours * 3600)
        
        # Get node metrics
        cursor.execute(
            '''
            SELECT 
                node_id,
                timestamp,
                cpu_load,
                memory_load,
                response_time,
                healthy
            FROM node_metrics
            WHERE timestamp >= ?
            ORDER BY timestamp
            ''',
            (since,)
        )
        
        node_metrics = {}
        for row in cursor.fetchall():
            node_id = row['node_id']
            if node_id not in node_metrics:
                node_metrics[node_id] = {
                    'timestamps': [],
                    'cpu_load': [],
                    'memory_load': [],
                    'response_time': [],
                    'healthy': []
                }
                
            node_metrics[node_id]['timestamps'].append(row['timestamp'])
            node_metrics[node_id]['cpu_load'].append(row['cpu_load'])
            node_metrics[node_id]['memory_load'].append(row['memory_load'])
            node_metrics[node_id]['response_time'].append(row['response_time'])
            node_metrics[node_id]['healthy'].append(bool(row['healthy']))
            
        # Get request metrics
        cursor.execute(
            '''
            SELECT 
                timestamp,
                total_requests,
                successful_requests,
                failed_requests,
                algorithm
            FROM request_metrics
            WHERE timestamp >= ?
            ORDER BY timestamp
            ''',
            (since,)
        )
        
        request_metrics = {
            'timestamps': [],
            'total_requests': [],
            'successful_requests': [],
            'failed_requests': [],
            'algorithms': []
        }
        
        for row in cursor.fetchall():
            request_metrics['timestamps'].append(row['timestamp'])
            request_metrics['total_requests'].append(row['total_requests'])
            request_metrics['successful_requests'].append(row['successful_requests'])
            request_metrics['failed_requests'].append(row['failed_requests'])
            request_metrics['algorithms'].append(row['algorithm'])
            
        conn.close()
        
        return jsonify({
            'node_metrics': node_metrics,
            'request_metrics': request_metrics
        })
        
    except Exception as e:
        logger.error(f"Error retrieving historical metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/algorithm/<algorithm>', methods=['POST'])
def set_algorithm(algorithm):
    """Change load balancing algorithm."""
    try:
        response = requests.post(f"{LOAD_BALANCER_URL}/algorithm/{algorithm}")
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f"Failed to set algorithm: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Setup database
    setup_database()
    
    # Start metrics collector
    metrics_thread = threading.Thread(target=metrics_collector, daemon=True)
    metrics_thread.start()
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)), debug=True)