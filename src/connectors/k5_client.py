"""
Client for communicating with the K5 RAG system (Industry Use Case).
"""

import logging
import requests
import json
from common.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class K5RAGClient:
    """
    API client for interfacing with the domain-specific K5 RAG system.
    Handles session management, authentication, and specific response parsing
    required for the industrial evaluation use case.
    """
    def __init__(self, base_url: str, internal_token: str, service_name: str, tenant_name: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

        # Authentication headers
        self.headers = {
            "X-Internal-Token": internal_token,
            "X-Service-Name": service_name,
            "X-Tenant-Name": tenant_name,
            "Content-Type": "application/json",
        }

    # ==========================================
    # Mock methods for pipeline compatibility
    # ==========================================
    def get_config(self) -> dict:
        """
        K5 RAG does not offer a dynamic configuration API.
        Mocked for pipeline compatibility.
        """
        logger.info("[K5-Client] get_config() called (Mock)")
        return {"system": "k5_industry_rag"}

    def update_config(self, new_settings: dict) -> dict:
        """
        K5 RAG is not dynamically reconfigured by the pipeline.
        Mocked and ignored for pipeline compatibility.
        """
        logger.info(f"[K5-Client] update_config() ignored: {new_settings}")
        return {"status": "mock_updated"}

    # ==========================================
    # Core dialog methods
    # ==========================================

    def get_thread_id(self) -> str:
        """
        Starts a new K5 session and returns the session_id as 'thread_id'.

        Returns:
            str: The unique session identifier.
        """
        res = requests.post(f"{self.base_url}session/start", headers=self.headers, json={}, timeout=self.timeout)
        res.raise_for_status()
        session_id = res.json()["session_id"]
        logger.info(f"[K5-Client] New session started: {session_id}")
        return session_id

    def delete_thread(self, thread_id: str) -> bool:
        """
        Ends the K5 session (equivalent to deleting the thread memory).

        Args:
            thread_id (str): The active session identifier.

        Returns:
            bool: True if the session was successfully ended, False otherwise.
        """
        if not thread_id:
            return False

        try:
            payload = {"session_id": thread_id}
            res = requests.post(f"{self.base_url}session/end", headers=self.headers, json=payload, timeout=self.timeout)
            res.raise_for_status()
            logger.info(f"[K5-Client] Session {thread_id} ended successfully.")
            return True
        except Exception as e:
            logger.warning(f"[K5-Client] Could not end session {thread_id}: {e}")
            return False

    def _parse_source_content(self, source_item: dict) -> str:
        """
        extracts text content from a K5 source object.
        Handles potentially nested JSON strings within the content field.
        """
        content = source_item.get("content", "")
        if isinstance(content, str) and content.strip().startswith("{"):
            try:
                parsed_json = json.loads(content)
                return parsed_json.get("Inhalt", content)
            except (json.JSONDecodeError, TypeError):
                pass
        return content

    @retry_with_backoff(exceptions=(requests.exceptions.RequestException,), max_retries=10, backoff_base=2.0)
    def ask_rag(self, question: str, thread_id: str) -> dict:
        """
        Sends the question to the K5 chat endpoint within the active session.
        Transforms the response to match the evaluation pipeline's expected format.

        Args:
            question (str): The user query to evaluate.
            thread_id (str): The current active session ID.

        Returns:
            dict: The formatted response containing 'answer' and 'context'.
        """
        payload = {
            "session_id": thread_id,
            "request_text": question
        }

        res = requests.post(f"{self.base_url}chat", headers=self.headers, json=payload, timeout=self.timeout)
        res.raise_for_status()
        chat_data = res.json()

        answer = chat_data.get("answer", "")
        raw_sources = chat_data.get("sources", [])
        parsed_sources = [self._parse_source_content(s) for s in raw_sources]

        return {
            "answer": answer,
            "context": parsed_sources
        }