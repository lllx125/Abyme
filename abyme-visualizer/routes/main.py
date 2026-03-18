"""
Main page route for the Abyme visualizer
"""

from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Render the main visualization page."""
    return render_template('index.html')
