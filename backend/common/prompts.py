# ============================================================================
# LABELS AND PLACEHOLDERS
# ============================================================================

FACTUAL_LABEL = 'True'
NON_FACTUAL_LABEL = 'False'
NEI_LABEL = 'Not Enough Info'

STATEMENT_PLACEHOLDER = '[STATEMENT]'
KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

FACT_CHECK_SYSTEM_PROMPT = 'You are a fact-checking agent responsible for verifying the accuracy of claims.'


# ============================================================================
# VERIFICATION PROMPTS
# ============================================================================

FINAL_ANSWER_OR_NEXT_SEARCH_PROMPT = f"""\
Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points from web search.
2. **CRITICAL RULE: You MUST ONLY use information from the KNOWLEDGE provided. DO NOT use your internal training knowledge, especially for:**
   - Current events, positions, or facts ("hiện nay", "hiện tại", "now", "current")
   - Political positions, leadership roles, or appointments
   - Recent events or changes in status
   - Any facts that could have changed over time
3. **TEMPORAL REASONING:**
   - **FIRST: Carefully extract ALL dates mentioned in KNOWLEDGE (ngày X/Y, tháng Z năm YYYY)**
   - **Pay special attention to year (năm) - if "ngày 20/1" appears, look for "năm 2025" or "năm 2026" in the same text**
   - For "current" or "now" questions, ONLY trust sources with the MOST RECENT dates
   - If KNOWLEDGE shows dated information (e.g., "đến tháng 10/2023"), recognize it as OUTDATED
   - If KNOWLEDGE contains conflicting information from different times, ALWAYS prefer the newest
4. **DECISION PROCESS:**
   - If KNOWLEDGE is empty or insufficient → Issue search query
   - If KNOWLEDGE directly contradicts your internal knowledge → Trust KNOWLEDGE, not your training data
   - If KNOWLEDGE clearly supports/refutes the claim → Make final decision
5. First, provide your reasoning in Vietnamese (tiếng Việt):
   - **IMPORTANT: First sentence must identify and extract the exact year from the evidence (e.g., "Theo nguồn, sự kiện xảy ra vào ngày 20/1 năm 2025")**
   - Identify and quote relevant dates from KNOWLEDGE with FULL year information
   - Compare temporal information if there are conflicts
   - Explain which source is most recent and why you trust it
   - Write 2-3 clear paragraphs explaining your analysis
   - DO NOT include any JSON, code blocks, or special formatting in your explanation

6. After your Vietnamese explanation, on a new line, output ONLY the JSON decision:
   - If KNOWLEDGE is empty or insufficient: {{"search_query": "Your search query"}}
   - If KNOWLEDGE clearly supports the claim: {{"final_answer": "{FACTUAL_LABEL}"}}
   - If KNOWLEDGE clearly refutes the claim: {{"final_answer": "{NON_FACTUAL_LABEL}"}}
   - If evidence is inconclusive: {{"final_answer": "{NEI_LABEL}"}}
   - If you need more information: {{"search_query": "Your search query"}}

7. Format example:
   [Your Vietnamese explanation here - 2-3 paragraphs]
   
   {{"search_query": "Chủ tịch quốc hội Việt Nam hiện nay"}}

KNOWLEDGE:
{KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{STATEMENT_PLACEHOLDER}
"""


MUST_HAVE_FINAL_ANSWER_PROMPT = f"""\
Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points from web search.
2. **CRITICAL: You MUST base your decision ONLY on the KNOWLEDGE provided. DO NOT use your internal training knowledge.**
3. **TEMPORAL REASONING (for "current"/"now" questions):**
   - **FIRST: Extract ALL dates with COMPLETE year information from KNOWLEDGE**
   - **Example: If you see "ngày 20/1", look for "năm 2025" or "năm 2026" nearby in the text**
   - If sources have different dates, TRUST THE MOST RECENT ONE
   - Ignore outdated information (e.g., "đến tháng 10/2023" is old if newer info exists)
   - If KNOWLEDGE contradicts your training data, TRUST KNOWLEDGE
4. **DECISION MAKING:**
   - Base verdict ONLY on what KNOWLEDGE explicitly states
   - If KNOWLEDGE directly confirms the claim → True
   - If KNOWLEDGE directly refutes the claim → False
   - If KNOWLEDGE is insufficient → Not Enough Info
5. First, provide your reasoning in Vietnamese (tiếng Việt):
   - **IMPORTANT: First sentence must identify the exact year from evidence (e.g., "Sự kiện xảy ra ngày 20/1/2025, không phải 2026")**
   - Quote relevant dates WITH FULL YEAR from KNOWLEDGE
   - Explain temporal reasoning if applicable
   - Show how KNOWLEDGE supports your decision
   - Write 2-3 clear paragraphs explaining your analysis
   - DO NOT include any JSON, code blocks, or special formatting in your explanation

6. After your Vietnamese explanation, on a new line, output ONLY the JSON decision:
   - True: {{"final_answer": "{FACTUAL_LABEL}"}}
   - False: {{"final_answer": "{NON_FACTUAL_LABEL}"}}
   - Not enough evidence: {{"final_answer": "{NEI_LABEL}"}}

7. Format example:
   [Your Vietnamese explanation here - 2-3 paragraphs]
   
   {{"final_answer": "{FACTUAL_LABEL}"}}

KNOWLEDGE:
{KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{STATEMENT_PLACEHOLDER}
"""

def format_verification_prompt(statement: str, knowledge: str, require_answer: bool = False) -> str:
    """
    Format verification prompt with statement and knowledge.
    
    Args:
        statement: The claim to verify
        knowledge: The knowledge/evidence gathered
        require_answer: If True, use prompt that requires final answer (no more searches)
    
    Returns:
        Formatted prompt ready for LLM
    """
    template = MUST_HAVE_FINAL_ANSWER_PROMPT if require_answer else FINAL_ANSWER_OR_NEXT_SEARCH_PROMPT
    
    prompt = template.replace(STATEMENT_PLACEHOLDER, statement)
    prompt = prompt.replace(KNOWLEDGE_PLACEHOLDER, knowledge if knowledge else 'N/A')
    
    return prompt
