"""
ACE-Step Turbo Trainer UI
Flask + Socket.IO application for dataset editing and LoRA fine-tuning.
"""

import os
import sys
import eventlet
eventlet.monkey_patch()

from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS

# Resolve paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ACESTEP_DIR = os.path.normpath(os.path.join(APP_DIR, "..", "ACE-Step"))
WORK_DIR = os.path.join(APP_DIR, "workspace")

# Ensure workspace exists
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "datasets"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "configs"), exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["SECRET_KEY"] = "acestep-trainer-ui"
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB max upload
app.config["ACESTEP_DIR"] = ACESTEP_DIR
app.config["WORK_DIR"] = WORK_DIR

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=200 * 1024 * 1024, async_mode="eventlet")

# Register blueprints
from backend.dataset_api import dataset_bp
from backend.trainer_api import trainer_bp

app.register_blueprint(dataset_bp, url_prefix="/api/dataset")
app.register_blueprint(trainer_bp, url_prefix="/api/trainer")

# Make socketio accessible to blueprints
app.config["SOCKETIO"] = socketio


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@socketio.on("connect", namespace="/training")
def on_training_connect():
    from backend.trainer_service import trainer_service
    status = trainer_service.get_status()
    socketio.emit("training_status", status, namespace="/training")


if __name__ == "__main__":
    print("=" * 60)
    print("  ACE-Step Turbo Trainer UI")
    print(f"  ACE-Step dir: {ACESTEP_DIR}")
    print(f"  Workspace:    {WORK_DIR}")
    print("  Starting on http://127.0.0.1:7870")
    print("=" * 60)
    socketio.run(app, host="127.0.0.1", port=7870, debug=False)
