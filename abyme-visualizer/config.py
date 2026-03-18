"""
Configuration settings for Abyme Visualizer
"""

import os


class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Flask-SocketIO settings
    SOCKETIO_ASYNC_MODE = 'gevent'  # Use gevent for production
    SOCKETIO_LOGGER = True
    SOCKETIO_ENGINEIO_LOGGER = True

    # Throttling settings
    TREE_UPDATE_THROTTLE_INTERVAL = 0.1  # 100ms = ~10 updates/second

    # Tree drawing settings
    TREE_RIGHT_SPACING = 150.0   # Horizontal spacing between nodes
    TREE_DOWN_SPACING = 100.0    # Vertical spacing between levels
    TREE_NODE_RADIUS = 30.0      # Node radius for layout calculations


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SOCKETIO_LOGGER = True
    SOCKETIO_ENGINEIO_LOGGER = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SOCKETIO_LOGGER = False
    SOCKETIO_ENGINEIO_LOGGER = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
