import os
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv, find_dotenv

from common.retry import retry_with_backoff

load_dotenv(find_dotenv(), override = True)
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

_RETRYABLE = [APIError]

class GEMINI:
    """
    A wrapper class for interacting with the Gemini API
    Attributes:
        MODEL_NAME (str): The name of the Gemini model being used (e.g., "gemini-2.5-flash").
        client (OpenAI): The client instance for making API calls to Gemini.
    """
    MODEL_NAME = ''
    def __init__(self, modelname):
        self.MODEL_NAME = modelname
        self.client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
        self.chat = self.client.chats.create(model=modelname)

        self.current_input_tokens = 0
        self.current_output_tokens = 0

    @retry_with_backoff(_RETRYABLE)
    def prompt(self, prompt):
        """
        Sends a prompt to the Gemini model and retrieves the model's response.

        Args:
            prompt (str): The input prompt to send to the Gemini model.

        Returns:
            str: The content of the model's response extracted from the API result.
        """

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=prompt,
        )

        if response.usage_metadata:
            self.current_input_tokens += response.usage_metadata.prompt_token_count
            self.current_output_tokens += response.usage_metadata.candidates_token_count

        return response.text

    @retry_with_backoff(_RETRYABLE)
    def chat_with_model(self, message):
        """
        Engage in a chat with the model using the ongoing session history.

        Args:
            message (str): The new message to send to the model.

        Returns:
            str: The model's response to the chat.
        """
        response = self.chat.send_message(message)
        if response.usage_metadata:
            self.current_input_tokens += response.usage_metadata.prompt_token_count
            self.current_output_tokens += response.usage_metadata.candidates_token_count

        return response.text
    def reset_chat(self):
        """
        Resets the chat session by creating a new chat instance.
        
        This clears all previous conversation history and resets token trackers.
        """
        self.chat = self.client.chats.create(model=self.MODEL_NAME)
        self.current_output_tokens = 0
        self.current_input_tokens = 0

    def get_and_reset_turn_tokens(self):
        """
        Retrieves the tracked input and output tokens for the current turn,
        then resets the counters to zero.

        Returns:
            tuple: (input_tokens, output_tokens)
        """
        in_tok = self.current_input_tokens
        out_tok = self.current_output_tokens
        self.current_input_tokens = 0
        self.current_output_tokens = 0
        return in_tok, out_tok

    def get_chat_history(self):
        """
        Retrieves the chat history of the current session.

        Returns:
            list: A list of message objects exchanged in the chat session.
        """
        return self.chat.get_history()

    def inject_history(self, user_prompt, model_response):
        """
        Injects a simulated conversational turn into the conversation history.
        This ensures that the model is aware of the cached context when generating
        subsequent follow-up responses.

        Args:
            user_prompt (str): The simulated user question.
            model_response (str): The simulated system response.
        """
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)]
        )
        model_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=model_response)]
        )

        history = self.get_chat_history()
        history.extend([user_content, model_content])
        self.chat = self.client.chats.create(model=self.MODEL_NAME, history=history)