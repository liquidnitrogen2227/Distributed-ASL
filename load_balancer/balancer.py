from flask import Flask, request, jsonify
import requests
import time
import logging
import threading
import json
import os
import base64
import math
from typing import Dict, List, Optional, Tuple

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Configuration
HEALTH_CHECK_INTERVAL = 5  # seconds
ALGORITHM = os.environ.get('LOAD_BALANCING_ALGORITHM', 'weighted_round_robin')

class RecognitionNode:
    """Represents a recognition service node."""
    
    def __init__(self, id: str, url: str):
        self.id = id
        self.url = url
        self.healthy = False
        self.cpu_load = 0
        self.memory_load = 0
        self.response_times = []
        self.last_used = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
    def update_metrics(self, cpu: float, memory: float):
        self.cpu_load = cpu
        self.memory_load = memory
        self.healthy = True
        
    def add_response_time(self, time_ms: float):
        self.response_times.append(time_ms)
        if len(self.response_times) > 100:  # Keep last 100 only
            self.response_times.pop(0)
    
    def avg_response_time(self) -> float:
        if not self.response_times:
            return float('inf')
        return sum(self.response_times) / len(self.response_times)
    
    def get_load_score(self) -> float:
        """Calculate a load score (lower is better)."""
        # Customize this formula based on your specific requirements
        return 0.7 * self.cpu_load + 0.3 * self.memory_load
    
    def get_performance_score(self):
        """
        Calculate a comprehensive performance score for this node.
        Lower score is better (node with lowest score gets selected).
        """
        # Start with base score
        score = 0.0
        
        # Apply weights to different metrics
        weights = {
            'cpu_load': 2.0,        # CPU utilization has high impact
            'memory_load': 1.5,     # Memory usage has medium-high impact
            'response_time': 2.0,    # Response time has high impact
            'queue_length': 3.0,     # Queue length has highest impact
            'error_rate': 4.0        # Error rate has very high impact
        }
        
        # Add weighted CPU score (0-100%)
        score += self.cpu_load * weights['cpu_load']
        
        # Add weighted memory score (0-100%)
        score += self.memory_load * weights['memory_load']
        
        # Add weighted response time
        # Normalize response time: assume anything under 50ms is great, over 1000ms is terrible
        avg_time = self.avg_response_time()
        if avg_time != float('inf'):  # Handle nodes with no response time history
            normalized_time = min(100, (avg_time / 10))  # Convert ms to 0-100 scale
            score += normalized_time * weights['response_time']
        else:
            # Penalize nodes with no history, but not too severely
            score += 50 * weights['response_time']
        
        # Add weighted queue length
        if hasattr(self, 'queue_length'):
            # Normalize queue: 0 is perfect, 10+ is bad
            normalized_queue = min(100, self.queue_length * 10)
            score += normalized_queue * weights['queue_length']
        
        # Add weighted error rate
        if hasattr(self, 'error_rate'):
            score += self.error_rate * weights['error_rate']
        
        # Apply a small bonus for nodes that have handled more requests successfully
        # This slightly prefers experienced nodes over newly added ones
        if hasattr(self, 'successful_requests') and self.successful_requests > 0:
            # Small negative adjustment (reduces score)
            experience_bonus = min(20, math.log2(self.successful_requests))
            score -= experience_bonus
        
        return score

class LoadBalancer:
    """Custom load balancer for recognition services."""
    
    def __init__(self):
        self.nodes = {}
        self.algorithm = ALGORITHM
        self.current_index = 0
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'algorithm': self.algorithm,
            'distribution': {}
        }
        
    def register_node(self, node_id: str, url: str):
        """Register a new recognition node."""
        self.nodes[node_id] = RecognitionNode(node_id, url)
        self.metrics['distribution'][node_id] = 0
        logger.info(f"Registered node {node_id} at {url}")
        
    def remove_node(self, node_id: str):
        """Remove a node from the balancer."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed node {node_id}")
    
    def get_healthy_nodes(self):
        """Get all currently healthy nodes."""
        return [node for node in self.nodes.values() if node.healthy]
    
    def select_node(self):
        """Select a node based on the configured algorithm."""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return None
            
        if self.algorithm == 'round_robin':
            return self._round_robin(healthy_nodes)
        elif self.algorithm == 'weighted_round_robin':
            return self._weighted_round_robin(healthy_nodes)
        elif self.algorithm == 'least_connections':
            return self._least_connections(healthy_nodes)
        elif self.algorithm == 'response_time':
            return self._best_response_time(healthy_nodes)
        elif self.algorithm == 'score':  # Add this new algorithm option
            return self._best_score(healthy_nodes)
        else:
            # Default to weighted round robin
            return self._weighted_round_robin(healthy_nodes)
    
    def _round_robin(self, nodes):
        """Simple round-robin selection."""
        self.current_index = (self.current_index + 1) % len(nodes)
        return nodes[self.current_index]
    
    def _weighted_round_robin(self, nodes):
        """Weight nodes by their inverse load."""
        # Calculate weights (higher weight = lower load)
        weights = [(100 - node.get_load_score()) for node in nodes]
        total_weight = sum(weights)
        
        if total_weight <= 0:
            return self._round_robin(nodes)
            
        # Normalize weights
        norm_weights = [w/total_weight for w in weights]
        
        # Choose based on normalized weights
        r = sum(norm_weights[:self.current_index])
        target = (r + 0.1) % 1.0  # Move forward but wrap around
        self.current_index = 0
        cumulative = 0
        
        for i, weight in enumerate(norm_weights):
            cumulative += weight
            if cumulative > target:
                self.current_index = i
                break
                
        return nodes[self.current_index]
    
    def _least_connections(self, nodes):
        """Select the node with the least current usage."""
        return min(nodes, key=lambda node: node.last_used)
    
    def _best_response_time(self, nodes):
        """Select node with best response time."""
        return min(nodes, key=lambda node: node.avg_response_time())
    
    def _best_score(self, nodes):
        """Select the node with the best (lowest) performance score."""
        if not nodes:
            return None
        return min(nodes, key=lambda node: node.get_performance_score())

    def update_metrics_for_request(self, node_id: str, success: bool):
        """Update metrics after a request."""
        self.metrics['requests_total'] += 1
        
        if success:
            self.metrics['requests_success'] += 1
            if node_id in self.metrics['distribution']:
                self.metrics['distribution'][node_id] += 1
        else:
            self.metrics['requests_failed'] += 1

# Initialize load balancer
balancer = LoadBalancer()

# Setup recognition nodes from environment
def setup_nodes():
    """Setup initial nodes from environment variables."""
    nodes = os.environ.get('RECOGNITION_NODES', '')
    if nodes:
        for node_info in nodes.split(','):
            parts = node_info.split(':')
            if len(parts) == 3:  # Format: id:host:port
                node_id, host, port = parts
                url = f"http://{host}:{port}"
                balancer.register_node(node_id, url)

# Health checking thread
def health_check_worker():
    """Periodically check health of all registered nodes."""
    while True:
        for node_id, node in balancer.nodes.items():
            try:
                response = requests.get(f"{node.url}/health", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    node.update_metrics(data.get('cpu', 100), data.get('memory', 100))
                else:
                    node.healthy = False
            except Exception as e:
                node.healthy = False
                logger.warning(f"Health check failed for node {node_id}: {e}")
        
        time.sleep(HEALTH_CHECK_INTERVAL)

@app.route('/register', methods=['POST'])
def register_node():
    """Register a new node."""
    data = request.json
    node_id = data.get('node_id')
    url = data.get('url')
    
    if not node_id or not url:
        return jsonify({'error': 'Missing node_id or url'}), 400
    
    balancer.register_node(node_id, url)
    return jsonify({'status': 'success', 'message': f'Registered node {node_id}'})

@app.route('/process', methods=['POST'])
def process_frame():
    """Process a frame by forwarding to a selected node."""
    start_time = time.time()
    
    # Select node
    node = balancer.select_node()
    
    if not node:
        return jsonify({
            'error': 'No healthy nodes available'
        }), 503
    
    # Forward request
    node.last_used = time.time()
    
    try:
        response = requests.post(
            f"{node.url}/process", 
            json=request.json,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            node.add_response_time(processing_time * 1000)  # Convert to ms
            node.successful_requests += 1
            balancer.update_metrics_for_request(node.id, True)
            
            # Add load balancer info to response
            result['load_balancer'] = {
                'node_id': node.id,
                'processing_time': processing_time,
                'algorithm': balancer.algorithm
            }
            
            return jsonify(result)
        else:
            node.failed_requests += 1
            balancer.update_metrics_for_request(node.id, False)
            return jsonify({
                'error': f'Recognition node returned an error: {response.status_code}'
            }), response.status_code
            
    except Exception as e:
        node.failed_requests += 1
        balancer.update_metrics_for_request(node.id, False)
        return jsonify({
            'error': f'Failed to process frame: {str(e)}'
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return current metrics for dashboard."""
    metrics = balancer.metrics.copy()
    
    # Add node-specific metrics
    metrics['nodes'] = {}
    for node_id, node in balancer.nodes.items():
        # Calculate performance score for display
        performance_score = node.get_performance_score()
        
        metrics['nodes'][node_id] = {
            'healthy': node.healthy,
            'cpu_load': node.cpu_load,
            'memory_load': node.memory_load,
            'avg_response_time': node.avg_response_time(),
            'successful_requests': getattr(node, 'successful_requests', 0),
            'failed_requests': getattr(node, 'failed_requests', 0),
            'performance_score': performance_score,
            'url': node.url
        }
    
    return jsonify(metrics)

@app.route('/algorithm/<algorithm>', methods=['POST'])
def change_algorithm(algorithm):
    """Change the load balancing algorithm."""
    valid_algorithms = ['round_robin', 'weighted_round_robin', 'least_connections', 'response_time', 'score']
    
    if algorithm not in valid_algorithms:
        return jsonify({'error': f'Invalid algorithm. Choose from: {valid_algorithms}'}), 400
    
    balancer.algorithm = algorithm
    balancer.metrics['algorithm'] = algorithm
    
    return jsonify({'status': 'success', 'algorithm': algorithm})

if __name__ == '__main__':
    # Setup initial nodes
    setup_nodes()
    
    # Start health check thread
    health_thread = threading.Thread(target=health_check_worker, daemon=True)
    health_thread.start()
    
    # Start server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))