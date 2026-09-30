import os
from openai import OpenAI, OpenAIError, RateLimitError, InternalServerError
from dotenv import load_dotenv, find_dotenv
from common.retry import retry_with_backoff

load_dotenv(find_dotenv(), override = True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

_RETRYABLE = [RateLimitError, InternalServerError, OpenAIError]

class ChatGPT:
    """
    A wrapper class for interacting with the OpenAI GPT API.

    Attributes:
        MODEL_NAME (str): The name of the GPT model being used (e.g., "gpt-3.5-turbo").
        client (OpenAI): The client instance for making API calls to OpenAI.
    """
    MODEL_NAME = ''

    def __init__(self, modelname):
        self.MODEL_NAME = modelname
        self.client = OpenAI()
        self.messages = []

        self.current_input_tokens = 0
        self.current_output_tokens = 0

    @retry_with_backoff(_RETRYABLE)
    def prompt(self, prompt):
        """
        Sends a single-turn prompt to the model and returns the response.

        Args:
            prompt (str): The input text to send to the model.

        Returns:
            str: The model's response.
        """
        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        if response.usage:
            self.current_input_tokens += response.usage.prompt_tokens
            self.current_output_tokens += response.usage.completion_tokens

        return response.choices[0].message.content.strip()

    @retry_with_backoff(_RETRYABLE)
    def chat_with_model(self, message):
        """
        Sends a message in an ongoing conversation (with history).

        Args:
            message (str): The new input message to append to the conversation.

        Returns:
            str: The model's response.
        """
        self.messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=self.messages,
        )
        if response.usage:
            self.current_input_tokens += response.usage.prompt_tokens
            self.current_output_tokens += response.usage.completion_tokens

        reply = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

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

    def reset_chat(self):
        """
        Clears the chat history and resets the token counters to zero.
        """
        self.messages = []
        self.current_input_tokens = 0
        self.current_output_tokens = 0
        
    def get_chat_history(self):
        """
        Returns the current chat history.

        Returns:
            list: The list of message dictionaries exchanged in the session.
        """
        return self.messages

    def inject_history(self, user_prompt, model_response):
        """
        Injects a simulated conversational turn into the conversation history.
        This ensures that the model knows the cached context when generating follow-up responses.

        Args:
            user_prompt (str): The user's input to inject.
            model_response (str): The model's corresponding response to inject.
        """
        self.messages.append({"role": "user", "content": user_prompt})
        self.messages.append({"role": "assistant", "content": model_response})