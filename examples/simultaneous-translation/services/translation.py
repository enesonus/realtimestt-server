import time
from openai import OpenAI
from config import GROQ_API_KEY

class TranslationService:
    """Handles translation of text from Turkish to English using Groq's API."""
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client with Groq's API."""
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        
    def translate(self, text: str) -> tuple[str, float]:
        """
        Translate text from Turkish to English.
        
        Args:
            text: The Turkish text to translate
            
        Returns:
            tuple: (translated_text, time_taken_in_seconds)
        """
        if not self.client:
            self._initialize_client()
            
        start_time = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "Translate the given Turkish text to English. Never output text other than the translation itself. Make sure your translation is properly aligned with the meaning of the original text. Do not add any additional information or context. Don't try to translate people's names and company names."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.2,
                max_completion_tokens=1024,
                top_p=0.95,
                stream=False,
            )
            
            translated_text = completion.choices[0].message.content
            time_taken = time.time() - start_time
            
            return translated_text, time_taken
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text, time.time() - start_time  # Return original text on error 