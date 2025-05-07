import time
import os
import base64
from openai import OpenAI
from config import GROQ_API_KEY, OPENAI_API_KEY, DEEPINFRA_API_KEY
from voice_mapping import get_voice


class TTSService:
    """Handles text-to-speech conversion using Groq's API."""

    def __init__(self, provider: str = "groq"):
        self.provider = provider
        self.client = None
        self._initialize_client()
        self._ensure_tmp_directory()
    
    def _initialize_client(self):
        """Initialize the OpenAI client with Groq's API."""
        if self.provider == "groq":
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=GROQ_API_KEY
            )
        elif self.provider == "openai":
            self.client = OpenAI(
                api_key=OPENAI_API_KEY
            )
        elif self.provider == "deepinfra":
            self.client = OpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url="https://api.deepinfra.com/v1/openai"
            )
        else:
            raise ValueError(
                "Unsupported TTS provider. Supported providers: groq, openai, deepinfra.")

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
            if self.provider == "groq":
                self.generate_audio_groq(voice_gender, text, audio_file_path)
            elif self.provider == "openai":
                self.generate_audio_openai(voice_gender, text, audio_file_path)
            elif self.provider == "deepinfra":
                self.generate_audio_deepinfra(
                    voice_gender, text, audio_file_path)

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

    def generate_audio_groq(self, voice_gender, text, output_path):
        tts_name = "Angelo-PlayAI" if voice_gender == "male" else "Jennifer-PlayAI"

        with self.client.audio.speech.with_streaming_response.create(
            model="playai-tts",
            voice=tts_name,  # Using voice_id would be ideal if the service supports it
            response_format="wav",
            input=text,
        ) as response:
            response.stream_to_file(output_path)
        return output_path

    def generate_audio_deepinfra(self, voice_gender, text, output_path):
        tts_name = "am_fenrir" if voice_gender == "male" else "af_heart"

        with self.client.audio.speech.with_streaming_response.create(
            model="hexgrad/Kokoro-82M",
            voice=tts_name,  # Using voice_id would be ideal if the service supports it
            response_format="mp3",
            input=text,
        ) as response:
            response.stream_to_file(output_path)
        return output_path

    def generate_audio_openai(self, voice_gender, text, output_path):
        tts_name = "ash" if voice_gender == "male" else "sage"

        with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=tts_name,
                response_format="mp3",
                input=text
        ) as response:
            response.stream_to_file(output_path)
        return output_path

    def generate_audio_stream(self, text: str, voice_gender: str):
        """
        Generate audio and stream it as it comes from the OpenAI API.
        This method can only be used when the provider is 'openai'.

        Args:
            text: The text to convert to speech.
            voice_gender: Preferred gender of the voice ("male" or "female").

        Yields:
            bytes: Chunks of audio data (MP3 format).
        
        Raises:
            ValueError: If the provider is not 'openai'.
            # Re-raises exceptions from the OpenAI API client.
        """
        if self.provider != "openai":
            raise ValueError("Streaming is only supported for the OpenAI TTS provider.")

        if not self.client:
            # This ensures the client is initialized, using self.provider.
            # If self.provider is "openai", it will set up the OpenAI client.
            self._initialize_client()

        # Voice selection specific to OpenAI, as in generate_audio_openai
        tts_name = "ash" if voice_gender == "male" else "sage"
        model_name = "gpt-4o-mini-tts"  # Consistent with generate_audio_openai
        audio_format = "mp3"          # Consistent with generate_audio_openai

        try:
            # Create a streaming response context manager
            streaming_response_context = self.client.audio.speech.with_streaming_response.create(
                model=model_name,
                voice=tts_name,
                input=text,
                response_format=audio_format
            )
            # Asynchronously enter the context and iterate over byte chunks
            with streaming_response_context as response:
                for chunk in response.iter_bytes(4096):
                    yield chunk
        except Exception as e:
            print(f"OpenAI TTS streaming error: {e}") # Log the error
            raise # Re-raise to allow the caller to handle
