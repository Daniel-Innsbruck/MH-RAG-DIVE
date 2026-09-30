import re
import json
from typing import Dict, Optional

class LLMResponseParser:
    """
ds (e.g., question, answer, logic_type)
        and extracts the complete output bundle. For the initial turn, the bundle is
        extended to include the bridging topic synthesized from the candidate se
    """
    def parse_and_validate(self, response: str) -> Optional[Dict[str, str]]:
        """
        Parses the output of the Conversation Generator (CG).

        Validates the presence of required fields (e.g., question, answer, logic_type)
        and extracts the complete output bundle. For the initial turn, the bundle is
        extended to include the bridging topic synthesized from the candidate set.

        Args:
            response (str): The raw string output from the CG LLM.

        Returns:
            dict: The normalized dictionary of the generated turn, or None if parsing fails.
        """
        try:
            if response.strip().startswith("```json"):
                response = re.sub(r"^```json\s*", "", response.strip())
                response = re.sub(r"\s*```$", "", response.strip())

            json_data = json.loads(response)

            if isinstance(json_data, dict):
                normalized = {}
                for k, v in json_data.items():
                    key = k.lower().replace(" ", "_")
                    normalized[key] = str(v).strip() if isinstance(v, str) else str(v)

                if 'question' in normalized and 'rag_input' not in normalized:
                    normalized['rag_input'] = normalized['question']

                is_initial_turn = 'bridging_topic' in normalized or str(normalized.get('type', '')).lower() == 'initial'

                if is_initial_turn:
                    required_fields = ['rag_input', 'question', 'answer', 'type', 'logic_type', 'multi_hop_flag',
                                       'bridging_topic']
                else:
                    required_fields = ['rag_input', 'question', 'answer', 'type', 'logic_type', 'multi_hop_flag']

                if all(field in normalized for field in required_fields):
                    # Safety cast for multi_hop_flag to ensure it is an integer (1 or 0)
                    try:
                        normalized['multi_hop_flag'] = int(normalized['multi_hop_flag'])
                    except (ValueError, TypeError):
                        normalized['multi_hop_flag'] = 0
                    return {field: normalized[field] for field in required_fields}
                else:
                    missing = [field for field in required_fields if field not in normalized]
                    print(f"Parser Warning: Missing fields in generation: {missing}")

        except json.JSONDecodeError:
            print("Parser Error: Generator LLM did not return valid JSON.")

        return None

    def parse_and_validate_validation(self, response: str) -> Optional[Dict]:
        """
        Parses the output of the Conversation Validator (CV).

        Strictly looks for the validation schema: {"correct": true/false, "reason": "..."}.
        Ensures the CV's decision on query validity and consistency is correctly captured.

        Args:
            response (str): The raw string output from the CV LLM.

        Returns:
            dict: The normalized validation dictionary, or None if parsing fails.
        """
        try:
            if response.strip().startswith("```json"):
                response = re.sub(r"^```json\s*", "", response.strip())
                response = re.sub(r"\s*```$", "", response.strip())

            json_data = json.loads(response)
            if isinstance(json_data, dict):
                normalized = {}
                for k, v in json_data.items():
                    key = k.lower().replace(" ", "_")
                    # Keep booleans as booleans, strip strings
                    normalized[key] = v.strip() if isinstance(v, str) else v

                required_fields = ['correct', 'reason']

                if all(field in normalized for field in required_fields):
                    # Safety catch: if the LLM returned "true" as a string instead of a boolean
                    if isinstance(normalized['correct'], str):
                        normalized['correct'] = normalized['correct'].lower() == 'true'

                    return {field: normalized[field] for field in required_fields}

        except json.JSONDecodeError:
            pass

        print("Warning: Incomplete CV validation response received")
        return None