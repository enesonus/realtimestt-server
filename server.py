import ngrok
import time
from aiohttp import web
import pathlib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Start the STT server with custom parameters')
    parser.add_argument('--silero-sensitivity', type=float, default=0.4,
                        help='Silero VAD sensitivity (default: 0.4)')
    parser.add_argument('--language', type=str, default="",
                        help='Language of transcript (default: auto)')
    parser.add_argument('--webrtc-sensitivity', type=int, default=2,
                        help='WebRTC VAD sensitivity (default: 2)')
    parser.add_argument('--post-speech-silence', type=float, default=0.4,
                        help='Post speech silence duration in seconds (default: 0.4)')
    parser.add_argument('--realtime-model', type=str, default='medium',
                        choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium',
                                 'medium.en', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo'],
                        help='Model type for realtime transcription (default: medium)')
    parser.add_argument('--model', type=str, default='large-v3-turbo',
                        choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium',
                                 'medium.en', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo'],
                        help='Model type for final transcription (default: large-v3-turbo)')
    parser.add_argument('--beam-size', type=int, default=7,
                        help='Beam size for final transcription (default: 7)')
    parser.add_argument('--beam-size-realtime', type=int, default=5,
                        help='Beam size for realtime transcription (default: 5)')
    parser.add_argument('--enable-realtime', action='store_true',
                        help='Enable realtime transcription')

    args = parser.parse_args()

    print("Starting server, please wait...")
    from RealtimeSTT import AudioToTextRecorder
    import asyncio
    import websockets
    import threading
    import numpy as np
    from scipy.signal import resample
    import json
    import logging
    import sys
    from dataclasses import dataclass
    from typing import Optional
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('faster_whisper').setLevel(logging.WARNING)

    @dataclass
    class ClientSession:
        websocket: websockets.ServerConnection
        recorder: Optional[AudioToTextRecorder] = None
        recorder_thread: Optional[threading.Thread] = None
        is_running: bool = True
        recorder_ready: threading.Event = threading.Event()
        # Track when VAD detects speech end
        last_vad_stop: Optional[float] = None

    class AudioServer:
        def __init__(self, args):
            self.clients = {}
            self.main_loop = None
            self.app = web.Application()
            self.setup_routes()
            self.ws_url = None
            self.args = args

            # Start the server with a dummy recorder to initialize whisper properly
            recorder = AudioToTextRecorder()
            recorder.start()
            recorder.stop()
            recorder.shutdown()
            del recorder
            print("Server initialized")

        def setup_routes(self):
            self.app.router.add_get('/', self.handle_client_page)
            # Add endpoint to get WebSocket URL
            self.app.router.add_get('/ws-url', self.handle_ws_url)

        async def handle_client_page(self, request):
            current_dir = pathlib.Path(__file__).parent
            index_path = current_dir / 'index.html'
            return web.FileResponse(index_path)

        async def handle_ws_url(self, request):
            # Endpoint to get the WebSocket URL
            return web.json_response({'url': self.ws_url})

        def get_recorder_config(self, client_id):
            def text_detected_callback(text):
                if self.main_loop is not None:
                    def preprocess_text(text):
                        text = text.lstrip()
                        if text.startswith("..."):
                            text = text[3:]
                        if text.endswith("...'."):
                            text = text[:-1]
                        if text.endswith("...'"):
                            text = text[:-1]
                        text = text.lstrip()
                        if text:
                            text = text[0].upper() + text[1:]
                        return text
                    text = preprocess_text(text)
                    asyncio.run_coroutine_threadsafe(
                        self.send_to_client(client_id, {
                            'type': 'realtime',
                            'text': text
                        }), self.main_loop)
                    print(f"\rClient {client_id}: {text}", flush=True, end='')

            def recording_start():
                """Called when VAD detects speech start"""
                logging.debug(f"Recording started for client {client_id}")
                client = self.clients.get(client_id)
                message = {'type': 'recording_start'}
                if client:
                    asyncio.run_coroutine_threadsafe(
                        self.send_to_client(client_id, message), self.main_loop)

            def recording_stop():
                """Called when VAD detects speech end"""
                logging.debug(f"Recording stopped for client {client_id}")
                client = self.clients.get(client_id)
                message = {'type': 'recording_stop'}
                if client:
                    asyncio.run_coroutine_threadsafe(
                        self.send_to_client(client_id, message), self.main_loop)

            def on_vad_start():
                logging.debug(f"VAD detect start for client {client_id}")
                client = self.clients.get(client_id)
                message = {'type': 'vad_detect_start'}
                if client:
                    client.last_vad_stop = None
                    asyncio.run_coroutine_threadsafe(
                        self.send_to_client(client_id, message), self.main_loop)

            def on_vad_stop():
                logging.debug(f"VAD detect stopped for client {client_id}")
                client = self.clients.get(client_id)
                message = {'type': 'vad_detect_stop'}
                if client:
                    client.last_vad_stop = time.time()
                    asyncio.run_coroutine_threadsafe(
                        self.send_to_client(client_id, message), self.main_loop)

            return {
                'silero_deactivity_detection': True,
                'pre_recording_buffer_duration': 1.2,
                'print_transcription_time': True,
                'faster_whisper_vad_filter': False,
                'spinner': False,
                'use_microphone': False,
                'model': self.args.model,
                'language': self.args.language,
                'silero_sensitivity': self.args.silero_sensitivity,
                'webrtc_sensitivity': self.args.webrtc_sensitivity,
                'post_speech_silence_duration': self.args.post_speech_silence,
                'min_length_of_recording': 1.1,
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
                'beam_size_realtime': self.args.beam_size_realtime,
            }

        async def initialize_client(self, client_id):
            """Initialize recorder for a client and wait until it's ready"""
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
                        # Read the metadata length (first 4 bytes)
                        metadata_length = int.from_bytes(
                            message[:4], byteorder='little')
                        # Get the metadata JSON string
                        metadata_json = message[4:4 +
                                                metadata_length].decode('utf-8')
                        metadata = json.loads(metadata_json)
                        sample_rate = metadata['sampleRate']
                        # Get the audio chunk following the metadata
                        chunk = message[4+metadata_length:]
                        resampled_chunk = self.decode_and_resample(
                            chunk, sample_rate, 16000)
                        client.recorder.feed_audio(resampled_chunk)
                    except Exception as e:
                        print(
                            f"Error processing message for client {client_id}: {e}")
                        continue

            except websockets.exceptions.ConnectionClosed:
                print(f"Client {client_id} disconnected")
            finally:
                await self.cleanup_client(client_id)

        def run_recorder(self, client_id):
            """Initialize and run recorder for a client"""
            client = self.clients[client_id]
            try:
                print(f"Initializing RealtimeSTT for client {client_id}...")
                client.recorder = AudioToTextRecorder(
                    **self.get_recorder_config(client_id))
                print(f"RealtimeSTT initialized for client {client_id}")
                client.recorder_ready.set()

                while client.is_running:
                    try:
                        full_sentence = client.recorder.text()
                        if full_sentence:
                            # Calculate latency if we have a VAD stop time
                            latency_ms = None
                            if client.last_vad_stop is not None:
                                latency_ms = int(
                                    (time.time() - client.last_vad_stop) * 1000)
                                client.last_vad_stop = None  # Reset for next sentence

                            if self.main_loop is not None:
                                asyncio.run_coroutine_threadsafe(
                                    self.send_to_client(client_id, {
                                        'type': 'fullSentence',
                                        'text': full_sentence,
                                        'latency_ms': latency_ms
                                    }), self.main_loop)
                                print(
                                    f"\rClient {client_id} Sentence: {full_sentence} (Latency: {latency_ms}ms)")
                    except Exception as e:
                        print(
                            f"Error in recorder thread for client {client_id}: {e}")
                        continue
            except Exception as e:
                print(
                    f"Fatal error in recorder thread for client {client_id}: {e}")
                client.is_running = False
                client.recorder_ready.set()  # Prevent deadlock

        async def send_to_client(self, client_id, message):
            if client_id in self.clients:
                client = self.clients[client_id]
                try:
                    await client.websocket.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    await self.cleanup_client(client_id)

        @staticmethod
        def decode_and_resample(audio_data, original_sample_rate, target_sample_rate):
            try:
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                num_original_samples = len(audio_np)
                num_target_samples = int(
                    num_original_samples * target_sample_rate / original_sample_rate)
                resampled_audio = resample(audio_np, num_target_samples)
                return resampled_audio.astype(np.int16).tobytes()
            except Exception as e:
                print(f"Error in resampling: {e}")
                return audio_data

        async def cleanup_client(self, client_id):
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
            self.main_loop = asyncio.get_running_loop()
            client_counter = 0

            async def client_handler(websocket):
                nonlocal client_counter
                client_counter += 1
                await self.handle_client(websocket, f"client_{client_counter}")

            print("Server started. Press Ctrl+C to stop the server.")

            # Start HTTP server on port 8001 with static domain
            http_tunnel = await ngrok.forward(8001,
                                              "http",
                                              authtoken_from_env=True)
            print(
                f"HTTP tunnel \"{http_tunnel.url()}\" -> \"http://localhost:8001\"")

            # Start WebSocket server on port 8002 with random domain
            ws_tunnel = await ngrok.forward(8002,
                                            "http",
                                            authtoken_from_env=True,
                                            )
            self.ws_url = ws_tunnel.url().replace('https://', 'wss://')
            print(
                f"WebSocket tunnel \"{self.ws_url}\" -> \"ws://localhost:8002\"")

            # Start HTTP server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 8001)
            await site.start()

            # Start WebSocket server
            ws_server = await websockets.serve(client_handler, "localhost", 8002)

            print(
                f"\033[92m\nAccess the demo client on {http_tunnel.url()}\033[92m\n")
            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                print("\nShutting down server...")
                await runner.cleanup()
                ws_server.close()
                await ws_server.wait_closed()
                for client_id in list(self.clients.keys()):
                    await self.cleanup_client(client_id)

    # Start the server with command line arguments
    server = AudioServer(args)
    try:
        asyncio.run(server.main())
    except KeyboardInterrupt:
        print("\nServer shutdown initiated by user")
