import time
from typing import Generator
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, SYSTEM_PROMPT


class ChatService:
    """Handles chat completion using Groq's API."""

    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.client = None
        self.history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the OpenAI client with Groq's API."""
        self.client = OpenAI(
            # base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )

    def get_chat_response(self, text: str):
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

        self.history.append({"role": "user", "content": text})

        try:
            completion = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=self.history,
                temperature=0.7,
                max_tokens=4096,
                top_p=1.0,
                stream=True,
            )
            init_text = ""
            remaining_text = ""
            total_text = ""
            word_list = []
            n_words = 5
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content and len(content) > 0:
                    total_text += content
                    word_list = total_text.strip(" ").split()
                    word_count = len(word_list)
                    if (word_count >= n_words and
                       (" ".join(word_list[n_words:]).find(".") > -1) and
                            init_text == ""):
                        init_text = total_text
                        time_taken = time.time() - start_time
                        yield init_text, time_taken

            self.history.append(
                {"role": "assistant", "content": total_text})
            remaining_text = total_text[len(init_text):]
            yield remaining_text, time.time() - start_time

        except Exception as e:
            print(f"Chat completion error: {e}")
            return "I'm sorry, I encountered an error.", time.time() - start_time
