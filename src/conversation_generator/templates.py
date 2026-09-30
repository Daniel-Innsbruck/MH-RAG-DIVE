CONVERSATION_PROMPTS = {
    'init_multihop_prompt': '''
        You are an expert evaluator of conversational AI systems, specialising in assessing Retrieval-Augmented Generation (rag_to_be_tested) models dynamically.

        Your task: Write the FIRST question to start a conversation with a RAG system. 
        You must create a Multi-Hop question based on the provided bridging topics and contexts.

        **Your Persona:** {Role}
        **Bridging Topics:** {t_bridge}

        **Context A:** 
        {chunk_a}

        **Context B:** 
        {chunk_b}

        ### RULES FOR THE QUESTION
        1  **Answerability:** It must be possible to give the right answer to the questions from the provided context without additional assumptions. Be specific.
        2. **Strict Multi-Hop:** The question MUST require synthesising facts from BOTH Context A and B. It must be impossible to answer using only one context.
        3. **No Compound Questions:** Do NOT ask two separate questions joined by "and" (e.g., "What is X, and what is Y?"). Ask a single, cohesive question.
        4. **Natural Persona:** Sound like a human naturally starting a chat based on your Persona. Easy English Vocabulary is preferred. Avoid stiff, academic exam-style phrasing. NEVER mention "Context A" or "Context B".
        5. **Scope:** The answer to your question must be entirely contained within the provided snippets.
        6. **No Premise Leakage:** Do not explicitly summarise the facts from both Context A and Context B in your question. A true multi-hop question asks for the connection WITHOUT giving it away. 
           - BAD: "Given that flour prices have tripled [Context B], how did the Iran attack [Context A] cause this?"
           - GOOD: "What direct impact did the recent escalation with Iran have on the pricing of basic baking ingredients like flour?"
                    
        ### PRECISION RULES
        
        - **No Vague Temporal Language:** NEVER use "recently", "lately", "not long ago", etc. Use **specific dates, months, or years** from the contexts (e.g., "in March 2024", "after the 14 April strike").
        - **Specific Answers:** The ground-truth answer must contain concrete facts (dates, numbers, names) that narrow down the possible reference ground truths to precisely the references sources that are given to you to formulate the question.
        
        ### CONFIGURATION
        - **Logic Type:** Choose the reasoning required: 'inference' (connecting premises), 'comparison' (comparing entities), or 'temporal' (timelines/sequences). 
        - **First Turn Rule:** Because this is the very first message, `rag_input` and `question` MUST be strictly identical.
        
        ### OUTPUT FORMAT (Strict JSON)
        {{
            "rag_input": "Your initial, natural question matching your Persona.",
            "question": "Exactly the same as rag_input.",
            "answer": "The comprehensive ground-truth answer combining facts from both contexts.",
            "type": "Initial",
            "logic_type": "inference / comparison / temporal",
            "multi_hop_flag": 1,
            "bridging_topic": "A short summary of the specific bridging topic you used."
        }}
    ''',

    'rephrase_init_prompt': '''
        This was incorrect. Reason: {reason}

        Generate a completly new QA Set again, following this exact format without any additional text:

        OUTPUT FORMAT (exact JSON):
        {{
            "rag_input": "The initial natural question sent to the RAG system, phrased in your assigned Role",
            "question": "Exactly the same as rag_input",
            "answer": "The combined, comprehensive ground-truth answer",
            "type": "Initial",
            "logic_type": "The logical reasoning type you chose (inference, comparison, or temporal)",
            "multi_hop_flag": 1,
            "bridging_topic": "A single string summarizing the bridging topic(s) you used"
        }}
    ''',
    'follow_up_prompt': '''
        This was the provided answer from the rag_to_be_tested system:

        **{RAG_answer}**

        {expanding_context}

        Your task is to write the **next turn in the conversation** — a natural follow-up question that assumes a shared conversational context, but **does not refer to any documents, sources, or retrieval process.**

        You are evaluating how well the system can maintain internal consistency, resolve ambiguities, and reason based on prior conversation turns.

        Choose one of the following Types:
            - **Follow-up:** Builds naturally on the previous answer.
            - **Clarification:** Resolves an ambiguity or digs deeper.
            - **Correction:** Rectifies a misunderstanding (do not use if system just didn't know).
            - **Comparative:** Compares aspects from the context.

        ### 1. THE CONVERSATIONAL INPUT (`rag_input`)
        - **Natural Chat:** Must be a concise, context-aware follow-up in persona, using pronouns like 'this', 'it', or 'that show'.
        - **MULTI-HOP PREFERENCE:** Prefer multi-hop questions when a logical connection exists between Context A and Context B that is not yet apparent from the dialogue history.
        - **NO COMPOUND QUESTIONS:** Do not ask two separate questions joined by "and".
        
        ### 2. THE DIALOGUE-AGNOSTIC QUESTION (`question`)
        You must translate the `rag_input` into a strict standalone format for later single-turn metric evaluation. It must be fully understandable without the conversation history.
        - **Resolve Vagueness:** You must explicitly ground all dangling references or relative terms (e.g., if using "different", "similar", or "instead of", state exactly what is being compared based on the history).
        - **THE ENTITY NAMING RULE (CRITICAL):** 
            - If an entity's true name (e.g., 'New York') was ALREADY revealed in the conversation history, you MUST use that exact name. 
            - If the name has NOT been revealed in the chat yet (e.g., the RAG system failed to find it previously), you MUST retain the descriptive puzzle (e.g., 'the city that flooded in 2008'). NEVER leak unmentioned names.

        ### 3. THE ANSWER (`answer`)
        - **POSITIVE GROUNDING (CRITICAL):** The question MUST be answerable with explicit, affirmative facts from the provided context. DO NOT ask questions where the answer is "The text does not mention this" or "No information is provided". If the text doesn't explicitly state it, you cannot ask it.        
        
        OUTPUT FORMAT (exact JSON):
        {{
            "rag_input": "The natural conversational input.",
            "question": "The dialogue-agnostic, standalone version according to the rules above.",
            "answer": "The expected, comprehensive ground-truth answer.",
            "type": "Follow-up / Clarification / Correction / Comparative",
            "required_NEW_fact_from_doc_A": "State exact new fact needed, or write 'None'",
            "required_NEW_fact_from_doc_B": "State exact new fact needed, or write 'None'"
        }}
    ''',
    'rephrase_follow_up_prompt': '''
        This was incorrect. Reason: {reason}

        Do it again, following this exact format without any additional text:
        OUTPUT FORMAT (exact JSON):
        {{
            "rag_input": "The natural conversational input.",
            "question": "The dialogue-agnostic, standalone version according to the rules above.",
            "answer": "The expected, comprehensive ground-truth answer.",
            "type": "Follow-up / Clarification / Correction / Comparative",
            "required_NEW_fact_from_doc_A": "State exact new fact needed, or write 'None'",
            "required_NEW_fact_from_doc_B": "State exact new fact needed, or write 'None'"
        }}
    '''
}