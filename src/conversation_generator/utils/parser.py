import logging
import json
import re
from typing import Dict, Optional, Any


logger = logging.getLogger(__name__)

class LLMResponseParser:

    def parse_and_validate(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse and validate the Conversation Generator (CG) response in one step

        Args:
            response (str): Raw LLM response

        Returns:
            Optional[Dict[str, Any]]: Parsed data if valid, None otherwise
        """
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            json_str = match.group(1) if match else response.strip()

            json_data = json.loads(json_str)

            if isinstance(json_data, dict):
                normalized = {}
                for k, v in json_data.items():
                    key = k.lower().replace(" ", "_")
                    if isinstance(v, str):
                        normalized[key] = v.strip()
                    elif isinstance(v, (bool, int)):
                        normalized[key] = v
                    else:
                        normalized[key] = str(v)

                if 'question' in normalized and 'rag_input' not in normalized:
                    normalized['rag_input'] = normalized['question']

                is_initial_turn = 'bridging_topic' in normalized or 'thematic_link' in normalized or str(normalized.get('type', '')).lower() == 'initial'

                if is_initial_turn:
                    required_fields = ['rag_input', 'question', 'answer', 'type', 'logic_type', 'multi_hop_flag', 'bridging_topic']
                else:
                    required_fields = ['rag_input', 'question', 'answer', 'type']

                missing = [field for field in required_fields if field not in normalized]

                if not missing:
                    if is_initial_turn:
                        try:
                            normalized['multi_hop_flag'] = int(normalized['multi_hop_flag'])
                        except (ValueError, TypeError, KeyError):
                            normalized['multi_hop_flag'] = 1

                    return normalized
                else:
                    logger.warning(f"Missing fields: {missing}")

        except json.JSONDecodeError:
            logger.error("Error: Response is not valid JSON.")
            logger.debug(f"Response text: {response}")
        except Exception as e:
            logger.error(f"Unexpected parsing error: {e}")

        return None