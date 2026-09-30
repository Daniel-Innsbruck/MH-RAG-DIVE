import os
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


class GEMINI:
    """
    A wrapper class for interacting with the Google Gemini API.

    Attributes:
        MODEL_NAME (str): The name of the Gemini model being used.
        client (genai.Client): The client instance for making API calls to Google.
        chat (genai.Chat): The active chat session object managing conversation history.
        safety_settings (list): Configured safety thresholds (set to BLOCK_NONE).
    """
    MODEL_NAME = ''

    def __init__(self, modelname):
        self.MODEL_NAME = modelname
        self.client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
        self.chat = self.client.chats.create(model=modelname)
        # configure Safety Settings (HARM_BLOCK_THRESHOLD_BLOCK_NONE)
        self.safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]
    def prompt(self, prompt):
        """
        Sends a single-turn prompt to the Gemini model and returns the response.

        Constructs an API request configuration with the specified prompt, sends it
        to the GPT model, and parses the response to extract the model's reply.

        Args:
            prompt (str): The input prompt to send to the GPT model.

        Returns:
            str: The content of the model's response extracted from the API result.
        """

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=self.safety_settings
            )
        )

        return response.text

    def chat_with_model(self, message):
        """
        Engage in a chat with the model using a series of messages.

        Args:
            message (str): The message to send to the model in the ongoing conversation.
        Returns:
            str: The model's response to the chat.
        """
        response = self.chat.send_message(message)

        return response.text

    def reset_chat(self):
        """
        Resets the chat session by creating a new chat instance.

        This clears all previous conversation history and starts fresh.
        """
        self.chat = self.client.chats.create(model=self.MODEL_NAME)

    def get_chat_history(self):
        """
        Retrieves the chat history of the current session.

        Returns:
            list: A list of messages exchanged in the chat session.
        """
        return self.chat.get_history()