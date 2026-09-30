import os
import json
import uuid
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError

from common.retry import retry_with_backoff

class ConversationCacher:
    """
    Implements the caching mechanism to reduce the computational cost
    of dynamic user simulation.

    The cache is organized as a dynamic tree structure. Initial multi-hop questions
    and source chunks are stored as root nodes. As RAG responses diverge, the tree
    expands dynamically by creating new branches. Semantic similarity search (cosine similarity)
    is used to identify matching conversational states.
    """
    def __init__(self, base_dir="./data/dialogue_cache", tau=0, safeguard=True):
        """
        Initializes the Conversation Cacher.

        Args:
            base_dir (str): The root directory for the cache files.
            tau (float): The semantic similarity threshold for identifying cache hits.
            safeguard (bool): If True, validates cached turns against the current dialogue history before reusing.
        """
        self.base_dir = base_dir
        self.roots_dir = os.path.join(base_dir, "roots")
        self.trees_dir = os.path.join(base_dir, "trees")
        self.tau = tau
        self.safeguard = safeguard

        # Create cache directories if they do not exist
        os.makedirs(self.roots_dir, exist_ok=True)
        os.makedirs(self.trees_dir, exist_ok=True)

        # Gemini Client for embedding RAG responses
        self.gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

        # Temporary in-memory cache for the current conversation (committed only upon success)
        self.current_temp_tree_nodes = []

        # Tracks the current path/node in the dialogue tree
        self.active_parent_id = None

    @retry_with_backoff(APIError)
    def _get_embedding(self, text):
        response = self.gemini_client.models.embed_content(
            model='gemini-embedding-001',
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # ==========================================
    # ROOT BOOTSTRAPPING (Initial Turn)
    # ==========================================
    def get_random_root(self, current_alpha=None, current_beta=None, current_dir=None):
        """
        Loads a random initial root (Turn 0) from the cache.

        Two operational modes:
        1. Tuning-Mode (Parameters = None): Ignores expansion configurations and browses all cached roots.
        2. Strict-Mode (Parameters set): Returns only roots that exactly match the current context expansion config.
        """
        root_files = [f for f in os.listdir(self.roots_dir) if f.endswith('.json')]
        if not root_files:
            return None

        import random
        random.shuffle(root_files)

        for file in root_files:
            with open(os.path.join(self.roots_dir, file), 'r', encoding='utf-8') as f:
                root_data = json.load(f)

            # MODE 1
            if current_alpha is None and current_beta is None and current_dir is None:
                return root_data

            # MODE 2
            config = root_data.get("expansion_config", {})
            if (config.get("alpha") == current_alpha and
                    config.get("beta") == current_beta and
                    config.get("direction") == current_dir):
                return root_data

        return None

    def get_all_root_filenames(self):
        """Returns a sorted list of all cached root filenames."""
        if not os.path.exists(self.roots_dir):
            return []
        return sorted([f for f in os.listdir(self.roots_dir) if f.endswith('.json')])

    def get_root_by_filename(self, filename):
        """Loads a specific root node via its filename."""
        filepath = os.path.join(self.roots_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def prepare_new_root(self, chunk_a, chunk_b, t_bridge, initial_question, alpha, beta, direction):
        """
        Prepares a newly generated multi-hop interaction for caching as a root node.
        Includes hyperparameters governing dialogue progression to maintain a consistent tree structure.
        """
        root_id = str(uuid.uuid4())
        root_data = {
            "root_id": root_id,
            "chunk_a": chunk_a,
            "chunk_b": chunk_b,
            "t_bridge": t_bridge,
            "initial_question": initial_question,
            "expansion_config": {
                "alpha": alpha,
                "beta": beta,
                "direction": direction
            }
        }
        return root_id, root_data

    def save_root(self, root_data):
        """Persists the root node to disk."""
        filepath = os.path.join(self.roots_dir, f"{root_data['root_id']}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(root_data, f, ensure_ascii=False, indent=2)

    def start_new_path(self, root_id):
        """Initializes the active pointer for a new dialogue path originating from a root node."""
        self.active_parent_id = root_id
        self.clear_temp_stage()

    # ==========================================
    # TREE CACHING
    # ==========================================
    def find_cache_hit(self, root_id, turn_index, current_rag_answer):
        """
        Searches for a semantically similar RAG response in the active branch of the dialogue tree.

        Computes the semantic similarity of the target system response to all cached
        response keys at the current node. The conversation continues from the cached
        response with the highest similarity.

        Args:
            root_id (str): The ID of the root node originating this dialogue tree.
            turn_index (int): The current conversational turn.
            current_rag_answer (str): The response just generated by the target RAG system.

        Returns:
            tuple: (best_matching_node_bundle, similarity_score, number_of_candidate_nodes)
        """
        tree_path = os.path.join(self.trees_dir, f"{root_id}.json")
        if not os.path.exists(tree_path):
            return None, 0.0, 0

        with open(tree_path, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)

        turn_key = str(turn_index)
        if turn_key not in tree_data:
            return None, 0.0, 0

        valid_nodes = [n for n in tree_data[turn_key] if n.get('parent_id') == self.active_parent_id]

        num_candidates = len(valid_nodes)
        if not valid_nodes:
            return None, 0.0, 0

        current_embedding = self._get_embedding(current_rag_answer)

        best_match = None
        highest_sim = 0.0

        for cached_node in valid_nodes:
            sim = self._cosine_similarity(current_embedding, cached_node['rag_answer_embedding'])
            if sim > highest_sim:
                highest_sim = sim
                best_match = cached_node

        if highest_sim >= self.tau:
            print(f"    --> [CACHE HIT] Similarity: {highest_sim:.4f} >= {self.tau}")
            return best_match, highest_sim, num_candidates
        else:
            print(f"    --> [CACHE MISS] Best similarity: {highest_sim:.4f} < {self.tau}")
            return None, highest_sim, num_candidates

    def stage_tree_node(self, turn_index, rag_answer, next_question_bundle):
        """
        Stages a generated turn in the in-memory temporary cache.
        This branch will only be permanently added to the tree upon successful completion
        of the entire dialogue.
        """
        new_node_id = str(uuid.uuid4())

        self.current_temp_tree_nodes.append({
            "node_id": new_node_id,
            "parent_id": self.active_parent_id,
            "turn_index": str(turn_index),
            "rag_answer": rag_answer,
            "rag_answer_embedding": self._get_embedding(rag_answer),
            "next_question_bundle": next_question_bundle
        })
        self.active_parent_id = new_node_id

    def commit_tree(self, root_id):
        """
        Saves all staged nodes for the current successful dialogue path to disk.
        """
        if not self.current_temp_tree_nodes:
            return

        tree_path = os.path.join(self.trees_dir, f"{root_id}.json")
        tree_data = {}
        if os.path.exists(tree_path):
            with open(tree_path, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)

        for node in self.current_temp_tree_nodes:
            turn_key = node['turn_index']
            if turn_key not in tree_data:
                tree_data[turn_key] = []

            # omit duplicate entries in case the exact same rag response already exists in current child nodes.
            exists = any(n['rag_answer'] == node['rag_answer'] for n in tree_data[turn_key])
            if not exists:
                tree_data[turn_key].append(node)

        with open(tree_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)

        # Clear the temporary cache after successful commit
        self.clear_temp_stage()

    def clear_temp_stage(self):
        """Empties the temporary node cache (e.g. after generation errors)."""
        self.current_temp_tree_nodes = []