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
2. **WHEN TO USE KNOWLEDGE vs INTERNAL TRAINING:**
   a) **Basic Facts (geography, history, science):** If KNOWLEDGE is empty or doesn't address the claim, you CAN use internal knowledge
      - Example: "Hà Nội là thủ đô Việt Nam" - well-known geographic fact
      - Example: "Việt Nam thống nhất năm 1975" - historical fact
   
   b) **Current Events ("hiện nay", "current"):** You MUST ONLY use KNOWLEDGE, NOT internal training
      - Political positions, leadership roles (may have changed)
      - Recent appointments or elections
      - Any facts that could have changed recently
   
   c) **When KNOWLEDGE contradicts your internal knowledge:**
      - If KNOWLEDGE is from trusted sources → TRUST KNOWLEDGE over your training
      - If KNOWLEDGE seems unreliable → State "Not Enough Info"
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
5. **CHAIN-OF-THOUGHT REASONING** - First, provide your step-by-step reasoning in Vietnamese:
   Step 1 - EXTRACT TEMPORAL INFO:
   - **CRITICAL: Scan ENTIRE knowledge text and extract ALL year mentions (e.g., "năm 2023", "năm 2024", "năm 2025", "năm 2026")**
   - **CRITICAL: For dates like "20/1" or "7/1", look for the year in the SAME SOURCE/paragraph**
   - Quote exact dates with years: "Theo Nguồn X: 'ngày 20/1/2025...'" or "Theo Nguồn Y: 'đến tháng 10 năm 2023...'"
   
   Step 2 - IDENTIFY MOST RECENT:
   - If multiple years found → Identify which is NEWEST (2026 > 2025 > 2024 > 2023...)
   - State explicitly: "Nguồn mới nhất: Nguồn X (năm 2025), Nguồn cũ: Nguồn Y (năm 2023)"
   
   Step 3 - MAKE DECISION:
   - For "current/now" questions → ONLY use the NEWEST source, IGNORE old data
   - Compare what the NEWEST source says vs the STATEMENT
   - Explain why you trust/distrust the evidence
   
   Write 2-3 clear paragraphs following these steps
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
2. **WHEN TO USE KNOWLEDGE:**
   - **Basic Facts:** If KNOWLEDGE is empty, you CAN use internal knowledge for well-known facts
   - **Current Events:** MUST use KNOWLEDGE only, NOT internal training
   - **Contradictions:** TRUST KNOWLEDGE from reliable sources over your training data
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
5. **CHAIN-OF-THOUGHT REASONING** - First, provide your step-by-step reasoning in Vietnamese:
   Step 1 - EXTRACT ALL TEMPORAL EVIDENCE:
   - **CRITICAL: List ALL years mentioned in KNOWLEDGE (e.g., "Nguồn 1: năm 2025", "Nguồn 2: năm 2023")**
   - For each source, quote the exact date with year
   
   Step 2 - RANK BY RECENCY:
   - Order sources by date (newest to oldest)
   - For "current/now" claims → State: "Nguồn mới nhất (năm YYYY) nói: ..."
   
   Step 3 - VERIFY AGAINST STATEMENT:
   - Compare what the NEWEST/MOST RELEVANT source states
   - Does it match the STATEMENT? → True
   - Does it contradict? → False
   - Not enough info? → Not Enough Info
   
   Write 2-3 clear paragraphs following these steps
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
