"""
Client for communicating with the standard RAG evaluation target.
"""

import requests
import logging
from common.retry import retry_with_backoff

logger = logging.getLogger(__name__)

class RAGClient:
    """
    API client for interfacing with the standard testing RAG pipeline.
    Handles dynamic configuration, thread memory management, and querying.
    """
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def get_config(self) -> dict:
        """
        Retrieves the current configuration of the target RAG system.

        Returns:
            dict: The current system configuration.
        """
        res = requests.get(f"{self.base_url}/config", headers=self.headers, timeout=10)
        res.raise_for_status()
        return res.json()

    def update_config(self, new_settings: dict) -> dict:
        """
        Partially updates the target RAG system's configuration (e.g., retrieval strategy).

        Args:
            new_settings (dict): The configuration to apply.

        Returns:
            dict: Confirmation of the applied settings.
        """
        payload = {"rag_settings": new_settings}
        res = requests.patch(f"{self.base_url}/config", headers=self.headers, json=payload, timeout=10)
        res.raise_for_status()
        return res.json()

    def get_thread_id(self) -> str:
        """
        Generates a new conversation thread ID on the target system.

        Returns:
            str: The newly created thread ID.
        """
        res = requests.get(f"{self.base_url}/thread", headers=self.headers, timeout=10)
        res.raise_for_status()
        return res.json().get("thread_id")

    def delete_thread(self, thread_id: str) -> bool:
        """
        Deletes the conversation thread and its associated history from the target system memory.

        Args:
            thread_id (str): The identifier of the thread to delete.

        Returns:
            bool: True if deleted successfully, False if an error occurred.
        """
        try:
            res = requests.delete(f"{self.base_url}/thread/{thread_id}", headers=self.headers, timeout=10)
            res.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Konnte Thread {thread_id} nicht auf dem Server löschen: {e}")
            return False

    @retry_with_backoff(exceptions=(requests.exceptions.RequestException,), max_retries=10, backoff_base=2.0)
    def ask_rag(self, question: str, thread_id: str) -> dict:
        """
        Submits a query to the target RAG system within a specific conversation thread.

        Args:
            question (str): The user query to evaluate.
            thread_id (str): The conversation thread context.

        Returns:
            dict: The system's response payload containing the answer and retrieved context.
        """
        payload = {"question": question, "thread_id": thread_id}
        res = requests.post(f"{self.base_url}/rag", headers=self.headers, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()