import os
import logging
import chromadb
import random
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv(), override=True)

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB using Google's Gemini models.
    """
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model='gemini-embedding-001',
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                )
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

class ChromaConnector:
    """
    Manages the connection to the Evaluation Vector Database (V_eval).

    This class provides the necessary retrieval methods for the Context Discoverer
    (finding initial chunks and semantic neighbors) and handles the deterministic
    context expansion during ongoing conversations.
    """
    def __init__(self, path: str, collection_name="wikipedia_eval_chunks"):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Critical Error: ChromaDB path does not exist: '{path}'")

        self.gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_collection(name=collection_name)

    def get_gemini_embedding(self, text):
        """
        Generates Gemini embedding for semantic search queries.

        Args:
            text (str): The text to embed.

        Returns:
            list: The vector representation of the text.
        """
        response = self.gemini_client.models.embed_content(
            model='gemini-embedding-001',
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )
        return response.embeddings[0].values

    def get_random_chunk(self):
        """
        Samples an arbitrary starting chunk from V_eval.

        Returns:
            dict: The randomly selected chunk, or None if the database is empty.
        """
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return None

        random_idx = random.randint(0, total_chunks - 1)
        results = self.collection.get(
            include=['metadatas', 'documents'],
            limit=1,
            offset=random_idx
        )

        if not results['ids']:
            return None

        return {
            'id': results['ids'][0],
            'text_snippet': results['documents'][0],
            'article_id': results['metadatas'][0]['parent_id'],
            'chunk_index': results['metadatas'][0]['chunk_index']
        }

    def get_bridging_chunk(self, query_text, exclude_article_id, top_k=5):
        """
        Executes a filtered semantic search to find a candidate bridging chunk.

        Enforces the constraint that the candidate must originate from a different
        source document than the initial chunk to ensure a proper multi-hop context.

        Args:
            query_text (str): The text snippet of the initial chunk.
            exclude_article_id (str): The parent document ID to exclude.
            top_k (int): Number of semantic neighbors to retrieve before randomizing.

        Returns:
            dict: A candidate bridging chunk, or None if no valid neighbors exist.
        """
        query_vector = self.get_gemini_embedding(query_text)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"article_id": {"$ne": exclude_article_id}}
        )

        if not results['ids'] or not results['ids'][0]:
            return None

        idx = random.randint(0, len(results['ids'][0]) - 1)
        return {
            'id': results['ids'][0][idx],
            'text_snippet': results['documents'][0][idx],
            'article_id': results['metadatas'][0][idx]['parent_id'],
            'chunk_index': results['metadatas'][0][idx]['chunk_index']
        }

    def get_adjacent_chunk(self, article_id, current_indices, total_chunks, direction="below"):
        """
        Retrieves the adjacent document chunk based on the preferred semantic direction.
        Automatically reverses direction if a document boundary is reached.

        Args:
            article_id (str): The parent document identifier.
            current_indices (list): A list of chunk indices currently in the active context.
            total_chunks (int): The total number of chunks in the parent document.
            direction (str): The preferred expansion direction ('above' or 'below').

        Returns:
            dict: The adjacent chunk dictionary, or None if the document is fully loaded.
        """
        if len(current_indices) >= total_chunks:
            return None

        min_idx = min(current_indices)
        max_idx = max(current_indices)

        target_idx = None

        if direction == "below":
            if max_idx + 1 < total_chunks:
                target_idx = max_idx + 1
            elif min_idx - 1 >= 0:
                target_idx = min_idx - 1
        elif direction == "above":
            if min_idx - 1 >= 0:
                target_idx = min_idx - 1
            elif max_idx + 1 < total_chunks:
                target_idx = max_idx + 1

        if target_idx is None:
            return None

        results = self.collection.get(
            where={"$and": [{"parent_id": article_id}, {"chunk_index": target_idx}]},
            include=['metadatas', 'documents']
        )

        if not results['ids']:
            return None

        return {
            'id': results['ids'][0],
            'text_snippet': results['documents'][0],
            'article_id': results['metadatas'][0]['parent_id'],
            'chunk_index': results['metadatas'][0]['chunk_index'],
            'total_chunks': results['metadatas'][0].get('total_chunks', total_chunks)
        }

    def get_chunk_by_id(self, article_id, chunk_index):
        """
        Retrieves a specific chunk by its parent article ID and chunk index.

        Args:
            article_id (str): The parent document identifier.
            chunk_index (int): The sequential position of the chunk.

        Returns:
            dict: The requested chunk dictionary, or None if not found.
        """
        results = self.collection.get(
            where={"$and": [{"parent_id": article_id}, {"chunk_index": chunk_index}]},
            include=['metadatas', 'documents']
        )

        if not results['ids']:
            return None

        return {
            'id': results['ids'][0],
            'text_snippet': results['documents'][0],
            'article_id': results['metadatas'][0]['parent_id'],
            'chunk_index': results['metadatas'][0]['chunk_index'],
            'total_chunks': results['metadatas'][0].get('total_chunks', 1)
        }