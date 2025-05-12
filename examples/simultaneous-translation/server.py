import ngrok
import time
import asyncio
import pathlib
import websockets
import json
import logging
import sys
import threading
from aiohttp import web

from RealtimeSTT import AudioToTextRecorder

# Local imports
from config import parse_args, PROVIDER
from utils import decode_and_resample, preprocess_realtime_text
from models import ClientSession
from services.translation import TranslationService
from services.tts import TTSService

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('faster_whisper').setLevel(logging.WARNING)

class AudioServer:
    """Main server class that handles both HTTP and WebSocket connections."""
    
    def __init__(self, args):
        self.clients = {}
        self.main_loop = None
        self.app = web.Application()
        self.setup_routes()
        self.ws_url = None
        self.args = args
        # Start the server with a dummy recorder to initialize whisper properly
        recorder = AudioToTextRecorder(**{
            'faster_whisper_vad_filter': False,
            'spinner': False,
            'use_microphone': False,
            'model': self.args.model,
            'enable_realtime_transcription': self.args.enable_realtime,
            'realtime_model_type': self.args.realtime_model,
        })
        recorder.start()
        recorder.stop()
        recorder.shutdown()
        
        # Initialize services
        self.translation_service = TranslationService()
        if self.args.enable_tts:
            self.tts_service = TTSService(PROVIDER)
            print(f"TTS is enabled with provider: {PROVIDER}")

    def setup_routes(self):
        """Set up HTTP routes for the server."""
        self.app.router.add_get('/', self.handle_client_page)
        self.app.router.add_get('/ws-url', self.handle_ws_url)

    async def handle_client_page(self, request):
        """Handle requests for the main client page."""
        current_dir = pathlib.Path(__file__).parent
        index_path = current_dir / 'index.html'
        return web.FileResponse(index_path)

    async def handle_ws_url(self, request):
        """Handle requests for the WebSocket URL."""
        return web.json_response({'url': self.ws_url})

    def get_recorder_config(self, client_id):
        """Generate the configuration for the RealtimeSTT recorder."""
        def text_detected_callback(text):
            """Callback for when realtime transcription is stabilized."""
            if self.main_loop is not None:
                text = preprocess_realtime_text(text)
                asyncio.run_coroutine_threadsafe(
                    self.send_to_client(client_id, {
                        'type': 'realtime',
                        'text': text
                    }), self.main_loop)
                print(f"\rClient {client_id}: {text}", flush=True, end='')

        def recording_start():
            """Called when VAD detects speech start."""
            logging.debug(f"Recording started for client {client_id}")
            client = self.clients.get(client_id)
            message = {'type': 'recording_start'}
            if client:
                asyncio.run_coroutine_threadsafe(
                    self.send_to_client(client_id, message), self.main_loop)

        def recording_stop():
            """Called when VAD detects speech end."""
            logging.debug(f"Recording stopped for client {client_id}")
            client = self.clients.get(client_id)
            message = {'type': 'recording_stop'}
            if client:
                asyncio.run_coroutine_threadsafe(
                    self.send_to_client(client_id, message), self.main_loop)

        def on_vad_start():
            """Called when VAD detects voice activity start."""
            logging.debug(f"VAD detect start for client {client_id}")
            client = self.clients.get(client_id)
            message = {'type': 'vad_detect_start'}
            if client:
                client.last_vad_stop = None
                asyncio.run_coroutine_threadsafe(
                    self.send_to_client(client_id, message), self.main_loop)

        def on_vad_stop():
            """Called when VAD detects voice activity end."""
            logging.debug(f"VAD detect stopped for client {client_id}")
            client = self.clients.get(client_id)
            message = {'type': 'vad_detect_stop'}
            if client:
                client.last_vad_stop = time.time()
                asyncio.run_coroutine_threadsafe(
                    self.send_to_client(client_id, message), self.main_loop)

        return {
            'silero_deactivity_detection': True,
            'pre_recording_buffer_duration': 1.5,
            'early_transcription_on_silence': self.args.post_speech_silence / 2,
            'print_transcription_time': True,
            'faster_whisper_vad_filter': True,
            'spinner': False,
            'use_microphone': False,
            'model': self.args.model,
            'language': self.args.language,
            'silero_sensitivity': self.args.silero_sensitivity,
            'webrtc_sensitivity': self.args.webrtc_sensitivity,
            'initial_prompt': self.args.initial_prompt,
            'post_speech_silence_duration': self.args.post_speech_silence,
            'min_length_of_recording': 2.0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': self.args.enable_realtime,
            'realtime_processing_pause': 0,
            'realtime_model_type': self.args.realtime_model,
            'on_realtime_transcription_stabilized': text_detected_callback,
            'on_recording_start': recording_start,
            'on_recording_stop': recording_stop,
            'on_vad_start': on_vad_start,
            'on_vad_stop': on_vad_stop,
            'beam_size': self.args.beam_size,
            'debug_mode': True,
            'beam_size_realtime': self.args.beam_size_realtime,
        }

    async def initialize_client(self, client_id):
        """Initialize recorder for a client and wait until it's ready."""
        client = self.clients[client_id]

        # Create and start recorder thread
        client.recorder_thread = threading.Thread(
            target=self.run_recorder,
            args=(client_id,)
        )
        client.recorder_thread.daemon = True
        client.recorder_thread.start()

        # Wait for recorder to be ready
        await asyncio.get_event_loop().run_in_executor(None, client.recorder_ready.wait)

        # Send initialization complete message
        await self.send_to_client(client_id, {
            'type': 'init_complete'
        })

        return client

    async def handle_client(self, websocket, client_id):
        """Handle a connected client."""
        print(f"Client {client_id} connected")

        # Create new client session
        client = ClientSession(websocket=websocket)
        self.clients[client_id] = client

        try:
            # Initialize recorder and wait until it's ready
            client = await self.initialize_client(client_id)

            async for message in websocket:
                if not client.recorder:
                    print(f"Recorder not ready for client {client_id}")
                    continue

                try:
                    # Check if it's a control message (JSON)
                    if message[0] == '{':
                        control_data = json.loads(message)
                        if 'voice_gender' in control_data:
                            client.preferred_voice_gender = control_data['voice_gender']
                            print(f"Client {client_id} set voice gender to: {client.preferred_voice_gender}")
                            continue

                    # Handle audio data
                    metadata_length = int.from_bytes(message[:4], byteorder='little')
                    metadata_json = message[4:4+metadata_length].decode('utf-8')
                    metadata = json.loads(metadata_json)
                    sample_rate = metadata['sampleRate']
                    chunk = message[4+metadata_length:]
                    resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
                    client.recorder.feed_audio(resampled_chunk)
                except Exception as e:
                    print(f"Error processing message for client {client_id}: {e}")
                    continue

        except websockets.exceptions.ConnectionClosed:
            print(f"Client {client_id} disconnected")
        finally:
            await self.cleanup_client(client_id)

    async def generate_and_send_tts(self, client_id: str, text: str) -> None:
        """Generate TTS audio and send it to the client."""
        if client_id not in self.clients or not hasattr(self, 'tts_service'):
            return

        try:
            client = self.clients[client_id]
            
            # Generate audio
            audio_b64, sample_rate, time_taken = await self.tts_service.generate_audio(
                text, client.language, client.preferred_voice_gender
            )
            
            if audio_b64:
                print(f"\033[92mTime taken for TTS: {time_taken:.2f}s\033[92m")
                print(f"Sending TTS audio to client {client_id}")
                
                await self.send_to_client(client_id, {
                    'type': 'tts_audio',
                    'audio': audio_b64,
                    'sample_rate': sample_rate
                })
            else:
                print(f"Failed to generate TTS for client {client_id}")

        except Exception as e:
            print(f"Error in generate_and_send_tts for client {client_id}: {e}")

    def run_recorder(self, client_id):
        """Initialize and run recorder for a client."""
        client = self.clients[client_id]
        try:
            print(f"Initializing RealtimeSTT for client {client_id}...")
            client.recorder = AudioToTextRecorder(**self.get_recorder_config(client_id))
            logger = logging.getLogger('RealtimeSTT')
            logger.setLevel(logging.DEBUG)
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.propagate = False
            print(f"RealtimeSTT initialized for client {client_id}")
            client.recorder_ready.set()

            while client.is_running:
                try:
                    full_sentence = client.recorder.text()
                    if full_sentence and self.main_loop is not None:
                        # Translate the text
                        translated_text, translation_time = self.translation_service.translate(full_sentence)
                        print(f"Original text: {full_sentence}\nTranslated text: {translated_text}")
                        print(f"\033[92mTime taken for translation: {translation_time:.2f}s\033[92m")

                        # Calculate latency if we have a VAD stop time
                        latency_ms = None
                        if client.last_vad_stop is not None:
                            latency_ms = int((time.time() - client.last_vad_stop) * 1000)
                            client.last_vad_stop = None  # Reset for next sentence

                        # Send full sentence to client
                        asyncio.run_coroutine_threadsafe(
                            self.send_to_client(client_id, {
                                'type': 'fullSentence',
                                'text': translated_text,
                                'latency_ms': latency_ms
                            }), self.main_loop)

                        # Generate and send TTS if enabled
                        if self.args.enable_tts:
                            asyncio.run_coroutine_threadsafe(
                                self.generate_and_send_tts(client_id, translated_text),
                                self.main_loop
                            )

                        print(f"\rClient {client_id} Sentence: {translated_text} \033[92m(Latency: {latency_ms}ms)\033[92m")
                except Exception as e:
                    print(f"Error in recorder thread for client {client_id}: {e}")
                    continue
        except Exception as e:
            print(f"Fatal error in recorder thread for client {client_id}: {e}")
            client.is_running = False
            client.recorder_ready.set()  # Prevent deadlock

    async def send_to_client(self, client_id, message):
        """Send a message to a connected client."""
        if client_id in self.clients:
            client = self.clients[client_id]
            try:
                await client.websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                await self.cleanup_client(client_id)

    async def cleanup_client(self, client_id):
        """Clean up resources for a disconnected client."""
        if client_id in self.clients:
            client = self.clients[client_id]
            client.is_running = False
            if client.recorder:
                client.recorder.stop()
                client.recorder.shutdown()
                del client.recorder
            del self.clients[client_id]
            print(f"Client {client_id} disconnected and cleaned up")

    async def main(self):
        """Main entry point for the server."""
        self.main_loop = asyncio.get_running_loop()
        client_counter = 0

        async def client_handler(websocket):
            nonlocal client_counter
            client_counter += 1
            await self.handle_client(websocket, f"client_{client_counter}")

        print("Server started. Press Ctrl+C to stop the server.")

        # Start HTTP server on port 8001 with static domain
        http_tunnel = await ngrok.forward(
            8001,
            "http",
            authtoken_from_env=True
        )
        print(f"HTTP tunnel \"{http_tunnel.url()}\" -> \"http://0.0.0.0:8001\"")

        # Start WebSocket server on port 8002 with random domain
        ws_tunnel = await ngrok.forward(
            8002,
            "http",
            authtoken_from_env=True,
        )
        self.ws_url = ws_tunnel.url().replace('https://', 'wss://')
        print(f"WebSocket tunnel \"{self.ws_url}\" -> \"ws://0.0.0.0:8002\"")

        # Start HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8001)
        await site.start()

        # Start WebSocket server
        ws_server = await websockets.serve(client_handler, "0.0.0.0", 8002)

        print(f"\033[92m\nAccess the demo client on {http_tunnel.url()}\033[92m\n")
        try:
            await asyncio.Future()  # run forever
        except asyncio.CancelledError:
            print("\nShutting down server...")
            await runner.cleanup()
            ws_server.close()
            await ws_server.wait_closed()
            for client_id in list(self.clients.keys()):
                await self.cleanup_client(client_id)


def main():
    """Entry point for the application."""
    args = parse_args()
    print("Starting server, please wait...")
    
    server = AudioServer(args)
    try:
        asyncio.run(server.main())
    except KeyboardInterrupt:
        print("\nServer shutdown initiated by user")


if __name__ == '__main__':
    main()
