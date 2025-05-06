import time
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL


class ChatService:
    """Handles chat completion using Groq's API."""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client with Groq's API."""
        self.client = OpenAI(
            # base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        
    def get_chat_response(self, text: str) -> tuple[str, float]:
        """
        Get a chat response for the given text.
        
        Args:
            text: The user's input text.
            
        Returns:
            tuple: (chat_response_text, time_taken_in_seconds)
        """
        if not self.client:
            self._initialize_client()
            
        start_time = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful voice assistant. Respond to the user's query concisely and naturally, as if you were speaking.
                        Keep your responses very brief. Your responses are maximum 30 words, you can not exceed this. You only speak english"""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.7,
                max_tokens=2048,
                top_p=1.0,
                stream=False,
            )
            
            response_text = completion.choices[0].message.content
            time_taken = time.time() - start_time
            
            return response_text, time_taken
            
        except Exception as e:
            print(f"Chat completion error: {e}")
            return "I'm sorry, I encountered an error.", time.time() - start_time
