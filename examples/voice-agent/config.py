import argparse
import os
from dotenv import load_dotenv

load_dotenv()

# Load API keys from environment variables or define them here
# It's strongly recommended to use environment variables for security
NGROK_AUTHTOKEN = os.environ.get('NGROK_AUTHTOKEN')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
DEEPINFRA_API_KEY = os.environ.get('DEEPINFRA_API_KEY')
PROVIDER = os.environ.get('PROVIDER', 'groq')
CHAT_MODEL = os.environ.get('CHAT_MODEL', 'gpt-4.1-mini')

def parse_args():
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
    parser.add_argument('--enable-tts', action='store_true',
                        help='Enable text-to-speech conversion of transcribed text')
    parser.add_argument('--initial_prompt', type=str,
                        default="",
                        help='Initial prompt for the transcription model.')
    args = parser.parse_args()
    return args