"""
Abyme Tree Visualizer - Flask Application

Real-time visualization of Abyme recursive model generation trees.
"""

import sys
sys.path.append('/home/lilixing/Abyme/abyme-rllm')

from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from config import config
import os


# Create Flask app
app = Flask(__name__)

# Load configuration
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Enable CORS for development
CORS(app)

# Initialize Flask-SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",  # Allow all origins in development
    async_mode='gevent',
    logger=True,
    engineio_logger=True
)

# Register blueprints
from routes.main import main_bp
app.register_blueprint(main_bp)

# Initialize WebSocket handlers
from routes.websocket import init_websocket_handlers
init_websocket_handlers(socketio)


@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy'}, 200


if __name__ == '__main__':
    print("=" * 60)
    print("Abyme Tree Visualizer")
    print("=" * 60)
    print(f"Environment: {env}")
    print(f"Debug: {app.config['DEBUG']}")
    print("Server starting on http://localhost:5000")
    print("=" * 60)

    # Run with gevent WebSocket support
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to avoid issues with threading
    )
