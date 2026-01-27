FACTUAL_LABEL = 'True'
NON_FACTUAL_LABEL = 'False'
NEI_LABEL = 'Not Enough Info'

STATEMENT_PLACEHOLDER = '[STATEMENT]'
KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'

FACT_CHECK_SYSTEM_PROMPT = 'You are a fact-checking agent responsible for verifying the accuracy of claims.'

FINAL_ANSWER_OR_NEXT_SEARCH_PROMPT = f"""\
You verify claims using KNOWLEDGE from search. Follow these rules:

**When to use KNOWLEDGE vs Internal Knowledge:**
- Basic facts (geography, history): Use internal knowledge if KNOWLEDGE empty
- Current events ("hiện nay"): MUST use KNOWLEDGE only
- Contradictions: Trust KNOWLEDGE over training data

**Temporal Reasoning:**
1. Extract ALL dates/years from KNOWLEDGE
2. For "current" claims → use NEWEST source only
3. Ignore outdated information

**Decision:**
- Insufficient → {{"search_query": "..."}}
- Supports → {{"final_answer": "{FACTUAL_LABEL}"}}
- Refutes → {{"final_answer": "{NON_FACTUAL_LABEL}"}}
- Unclear → {{"final_answer": "{NEI_LABEL}"}}

**Format:**
[Brief Vietnamese explanation - 1 paragraph]

{{"final_answer": "..."}} or {{"search_query": "..."}}

KNOWLEDGE:
{KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{STATEMENT_PLACEHOLDER}
"""


MUST_HAVE_FINAL_ANSWER_PROMPT = f"""\
Make FINAL decision on claim using KNOWLEDGE.

**Rules:**
- Basic facts: Use internal knowledge if KNOWLEDGE empty
- Current events: MUST use KNOWLEDGE only  
- Temporal: Use newest source for "current" claims
- Trust KNOWLEDGE over training data

**Output:**
[Brief Vietnamese explanation]

{{"final_answer": "{FACTUAL_LABEL}"}} or {{"final_answer": "{NON_FACTUAL_LABEL}"}} or {{"final_answer": "{NEI_LABEL}"}}

KNOWLEDGE:
{KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{STATEMENT_PLACEHOLDER}
"""

def format_verification_prompt(statement: str, knowledge: str, require_answer: bool = False) -> str:
    template = MUST_HAVE_FINAL_ANSWER_PROMPT if require_answer else FINAL_ANSWER_OR_NEXT_SEARCH_PROMPT
    prompt = template.replace(STATEMENT_PLACEHOLDER, statement)
    prompt = prompt.replace(KNOWLEDGE_PLACEHOLDER, knowledge if knowledge else 'N/A')
    return prompt
