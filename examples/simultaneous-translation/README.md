# RealtimeSTT Server

A real-time speech-to-text (STT) server that transcribes audio in real-time, translates it from Turkish to English, and optionally converts it back to speech using text-to-speech (TTS).

## Features

- Real-time audio transcription using Whisper models
- Translation from Turkish to English using Groq's API
- Optional text-to-speech conversion
- WebSocket communication for real-time updates
- Voice Activity Detection (VAD) for better speech recognition
- Ngrok tunneling for easy access from anywhere

## Project Structure

```
RealtimeSTT_server/
├── config.py              # Configuration and command-line arguments
├── models.py              # Data models for the application
├── server.py              # Main server implementation
├── utils.py               # Utility functions for audio processing
├── services/
│   ├── translation.py     # Translation service using Groq API
│   └── tts.py             # Text-to-speech service
└── index.html             # Web client for interacting with the service
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)

### Running the Server

```bash
python server.py [options]
```

### Command-line Options

- `--silero-sensitivity`: Silero VAD sensitivity (default: 0.4)
- `--language`: Language of transcript (default: auto)
- `--webrtc-sensitivity`: WebRTC VAD sensitivity (default: 2)
- `--post-speech-silence`: Post speech silence duration in seconds (default: 0.4)
- `--realtime-model`: Model type for realtime transcription (default: medium)
- `--model`: Model type for final transcription (default: large-v3-turbo)
- `--beam-size`: Beam size for final transcription (default: 7)
- `--beam-size-realtime`: Beam size for realtime transcription (default: 5)
- `--enable-realtime`: Enable realtime transcription
- `--enable-tts`: Enable text-to-speech conversion

## Usage

1. Start the server with desired options
2. Access the web client using the provided URL
3. Speak into your microphone to see real-time transcription and translation

## API Keys

The server requires API keys for:
- Ngrok (for tunneling)
- Groq (for translation and TTS)

Edit the `config.py` file to update these keys or set them as environment variables.