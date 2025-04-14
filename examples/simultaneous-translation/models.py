from dataclasses import dataclass
from typing import Optional
import threading
import websockets

class ClientSession:
    """Represents a connected client session with its associated state."""
    
    def __init__(self, websocket: websockets.ServerConnection):
        self.websocket = websocket
        self.recorder = None
        self.recorder_thread = None
        self.is_running = True
        self.recorder_ready = threading.Event()
        self.last_vad_stop = None
        self.preferred_voice_gender = "female"  # Default to female voice
        self.language = "en-us"  # Default to English language 