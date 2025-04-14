import time
import os
import base64
from openai import OpenAI
from config import GROQ_API_KEY
from voice_mapping import get_voice

class TTSService:
    """Handles text-to-speech conversion using Groq's API."""
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.client = None
        self._initialize_client()
        self._ensure_tmp_directory()
    
    def _initialize_client(self):
        """Initialize the OpenAI client with Groq's API."""
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
    
    def _ensure_tmp_directory(self):
        """Ensure the temporary directory for audio files exists."""
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
    
    async def generate_audio(self, text: str, language: str, voice_gender: str) -> tuple[str, int, float]:
        """
        Generate audio from text.
        
        Args:
            text: The text to convert to speech
            language: The language code
            voice_gender: Preferred gender of the voice
            
        Returns:
            tuple: (base64_encoded_audio, sample_rate, time_taken_in_seconds)
        """
        if not self.client:
            self._initialize_client()
        
        voice_id = get_voice(language, voice_gender)
        start_time = time.time()
        
        try:
            # Generate unique filename
            audio_file_path = f"tmp/audio_{time.time()}.wav"
            
            # Generate audio file
            with self.client.audio.speech.with_streaming_response.create(
                model="playai-tts",
                voice="Angelo-PlayAI",  # Using voice_id would be ideal if the service supports it
                response_format="wav",
                input=text,
            ) as response:
                response.stream_to_file(audio_file_path)
            
            # Encode to base64
            audio_b64 = None
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up file
            os.remove(audio_file_path)
            
            time_taken = time.time() - start_time
            return audio_b64, 24000, time_taken  # 24000 is the sample rate
            
        except Exception as e:
            print(f"TTS error: {e}")
            return None, 0, time.time() - start_time 