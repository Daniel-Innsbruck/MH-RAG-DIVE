VALIDATION_PROMPTS = {''
    'validate_init_prompt': '''
        Validate the following initial prompt.  
        It contains a `RAG_input` used to test a Retrieval-Augmented Generation (rag_to_be_tested) system.  
        This should represent the start of a conversation with the rag_to_be_tested system.  

        Requirements:  
        1. The context of the RAG_input should be related to the provided document.  
        2. The question must be exactly the same as the `RAG_input`.  
        3. The answer must be correct.  
        4. Multi-Hop Integrity: The answer MUST require synthesising information from BOTH provided snippets (Context A and Context B). It must not be answerable by only one snippet and must not be a compound question (e.g., "What is X and what is Y?").        5. Query Type Alignment: The logical reasoning used must strictly match the declared `logic_type`.

        Input to validate:  
        {question}  
        Provided Documents (Contains Context A and Context B):
        {document}

        If any of the above requirements are not met, return the reason and set `"correct"` to `false`.  
        If all requirements are met, set `"correct"` to `true`.  

        Return the result as JSON in the following format. Perform the audit steps first before reaching your final verdict:
        {{
            "audit_fact_A": "What specific fact MUST be retrieved from Context A to answer this? (If answerable without A, write 'None')",
            "audit_fact_B": "What specific fact MUST be retrieved from Context B to answer this? (If answerable without B, write 'None')",
            "is_compound": "Is this just two separate questions glued together? (Yes/No)",
            "correct": true/false,
            "reason": "Explanation if incorrect (e.g. 'Fails multi-hop because Fact B is not needed'), otherwise empty"
        }}
    ''',
    'validate_follow_up_prompt': '''
         Validate the follow-up input against the rules below.

         1. RAG_input
            Must be clear, specific, and contextually connected. It must build on the prior turn and use conversational pronouns (e.g., this, it, that). Never mention documents.

        2. Question (1:1 Standalone Form)
            Must be a perfect standalone translation of the rag_input. It must retain the EXACT SAME difficulty, information content, and meaning as the rag_input, but be fully understandable without the conversation history.
            - It must resolve all vague pronouns.
            - It must resolve all dangling references and implicit comparisons. If the rag_input uses relative terms (e.g., "different", "alternative", "similar", "the other", or "instead of"), the standalone question must explicitly state the baseline of comparison using the conversation history.            
         
         3. Type
            Must be one of: Follow-up, Clarification, Correction, Comparative. Must match its declared conversational type.
    
         4. Answer
            - The Answer must be factually correct based on the active context.
            - STRICT REJECTION: You MUST reject the input if the ground-truth answer relies on the ABSENCE of information ("The text does not mention...", "There is no information...", "Does not state", or "No, the provided text..."). The question must be positively answerable.
    
        5. Question Integrity & Premise Leakage (CRITICAL)
            - NO COMPOUND QUESTIONS: It must NOT be two separate questions joined together.
            - STRICT ENTITY CHECK: You must check how the standalone 'question' handles entities compared to the Conversation History:
              a) If an entity's true name was ALREADY revealed in the history, the 'question' MUST use that exact name. Reject artificially maintained puzzles (e.g., using "the town that flooded" if the history already established it is "Charles City").
              b) If the name has NOT been revealed in the history (Premise Leakage), the 'question' MUST NOT leak it. It must reuse the descriptive clues to maintain the multi-hop difficulty.
         Input to validate: 
         {question}
    
         Conversation History: 
         {conversation_history}
    
         Currently Active Context (Use this for factual verification):
         {active_context}
    
         Return the result as JSON in the following format:
         {{
            "correct": true/false,
            "reason": "Explanation of which requirement failed and why (leave empty if correct)"
         }}
    '''
}