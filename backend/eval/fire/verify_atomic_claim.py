"""
Rates a single atomic claim for accuracy.
For each atomic claim, the process would be to prompt the model think of the search term to obtain relevant information,
and then let the model decide if the information is enough to make a judgement or the model needs to continue searching.
"""

import dataclasses
import torch
from typing import Any, Union, List, Optional, Dict, Tuple
from common import modeling, shared_config, utils
from eval.fire import config as fire_config
from eval.fire import query_serper
from eval.fire.query_serper import SerperAPI
from sentence_transformers import SentenceTransformer, util

try:
    from common.vietnamese_utils import preprocessor
    from common.query_deduplication import deduplicator as query_deduplicator
    from common.evidence_validator import validator as evidence_validator
    from common.confidence import calibrator as confidence_calibrator
    from common.database import db as fact_check_db
    from common.claim_analyzer import get_analyzer
    from common.prompts import (
        FACTUAL_LABEL,
        NON_FACTUAL_LABEL,
        NEI_LABEL,
        STATEMENT_PLACEHOLDER,
        KNOWLEDGE_PLACEHOLDER,
        FINAL_ANSWER_OR_NEXT_SEARCH_PROMPT,
        MUST_HAVE_FINAL_ANSWER_PROMPT,
        format_verification_prompt
    )
    VIETNAMESE_SUPPORT = True
except ImportError as e:
    print(f"Vietnamese components not available: {e}")
    VIETNAMESE_SUPPORT = False
    # Fallback labels if prompts import fails
    FACTUAL_LABEL = 'True'
    NON_FACTUAL_LABEL = 'False'
    NEI_LABEL = 'Not Enough Info'
    STATEMENT_PLACEHOLDER = '[STATEMENT]'
    KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'

device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)


@dataclasses.dataclass()
class GoogleSearchResult:
    query: str
    result: str
    link: str = ''


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


def call_search(
        search_query: str,
        search_type: str = fire_config.search_type,
        num_searches: int = fire_config.num_searches,
        serper_api_key: str = shared_config.serper_api_key,
        search_postamble: str = '',
        atomic_claim: str = '',
        with_links: bool = False,
) -> Union[str, list]:
    """Call Google Search to get the search result with Vietnamese enhancements."""
    
    if VIETNAMESE_SUPPORT and atomic_claim:
        cached = fact_check_db.get_cached_search(search_query)
        if cached and not with_links:
            return cached
    
    if VIETNAMESE_SUPPORT:
        is_dup, similarity = query_deduplicator.check_and_add(search_query)
        if is_dup and not with_links:
            return "[Duplicate query - skipped to reduce API costs]"
    
    if VIETNAMESE_SUPPORT:
        try:
            enhanced_query = query_serper.enhance_vietnamese_query(search_query)
            search_query = enhanced_query
        except Exception as e:
            print(f"Query enhancement failed, using original: {e}")
    
    search_query += f' {search_postamble}' if search_postamble else ''

    if search_type == 'serper':
        if VIETNAMESE_SUPPORT and hasattr(query_serper, 'VietnameseSerperAPI'):
            serper_searcher = query_serper.VietnameseSerperAPI(serper_api_key, k=num_searches)
            base_searcher = serper_searcher.base_api
        else:
            base_searcher = query_serper.SerperAPI(serper_api_key, k=num_searches)
        
        if with_links:
            # Get raw results and extract link from first result
            raw_results = base_searcher._google_serper_api_results(search_query, search_type='search', num=num_searches)
            result_key = base_searcher.result_key_for_type['search']
            
            link = ''
            if result_key in raw_results and len(raw_results[result_key]) > 0:
                link = raw_results[result_key][0].get('link', '')
            
            return link
        else:
            if VIETNAMESE_SUPPORT and hasattr(query_serper, 'VietnameseSerperAPI'):
                result = serper_searcher.run(search_query, claim=atomic_claim, k=num_searches)
            else:
                result = base_searcher.run(search_query, k=num_searches)
            
            if VIETNAMESE_SUPPORT and atomic_claim:
                fact_check_db.cache_search(search_query, result)
            
            return result
    else:
        raise ValueError(f'Unsupported search type: {search_type}')

def get_sentence_similarity(new_sent, sentences, threshold=0.9):
    if len(sentences) == 0:
        return 0
    single_embedding  = sbert_model.encode(new_sent, convert_to_tensor=True).to(device)
    list_embeddings = sbert_model.encode(sentences, convert_to_tensor=True).to(device)
    similarities = util.cos_sim(single_embedding, list_embeddings)

    count_above_threshold = sum(1 for i in range(len(sentences)) if similarities[0][i].item() > threshold)
    return count_above_threshold

def final_answer_or_next_search(
        atomic_claim: str,
        past_searches: List[GoogleSearchResult],
        model: modeling.Model,
        diverse_prompt: bool = False,
        tolerance: int = 4,
) -> Tuple[Union[FinalAnswer, GoogleSearchResult, None, str], Optional[Dict]]:
    """Get the next query from the model.
    atomic_claim: The claim that we need to verify.
    past_searches: The search results from the previous searches.
    model: The backbone language model we choose.
    diverse_prompt: Whether to use diverse prompt or not.
    tolerance: The number of similar queries or search results to tolerate before early stopping.
    """

    # Sort searches by temporal relevance for current claims
    if VIETNAMESE_SUPPORT and past_searches:
        # Check if current claim using pre-computed analysis if available
        # This is passed via global/closure or re-computed lightweight
        try:
            analyzer = get_analyzer()
            # Try to get from cache first (very fast)
            analysis = analyzer.analyze_claim(atomic_claim)
            is_current_claim = analysis.get('temporal_sensitivity') == 'high'
        except:
            # Fallback to simple check
            is_current_claim = any(keyword in atomic_claim.lower() for keyword in 
                                   ["hiện nay", "hiện tại", "now", "current", "bây giờ", "hôm nay"])
        
        if is_current_claim:
            sorted_searches = []
            for search in past_searches:
                validation = evidence_validator.validate_evidence(
                    evidence_text=search.result,
                    source_url=search.link if hasattr(search, 'link') and search.link else search.query,
                    claim=atomic_claim
                )
                search.validation = validation
                sorted_searches.append((search, validation.get('temporal_score', 0.5)))
            
            # Sort by temporal score (most recent first) for current claims
            past_searches = [s[0] for s in sorted(sorted_searches, key=lambda x: x[1], reverse=True)]
    
    # Format knowledge with date metadata
    knowledge_parts = []
    for i, s in enumerate(past_searches, 1):
        knowledge_text = f"[Nguồn {i}]: {s.result}"
        if hasattr(s, 'validation') and s.validation.get('evidence_date'):
            knowledge_text = f"[Nguồn {i} - Ngày: {s.validation['evidence_date']}]: {s.result}"
        knowledge_parts.append(knowledge_text)
    
    knowledge = '\n\n'.join(knowledge_parts)
    knowledge = 'N/A' if not knowledge else knowledge
    full_prompt = format_verification_prompt(atomic_claim, knowledge, require_answer=False)
    full_prompt = utils.strip_string(full_prompt)

    query_history = [item.query for item in past_searches]
    search_history = [item.result for item in past_searches]

    if diverse_prompt:
        if len(query_history) >= 2:
            full_prompt += "Please pay attention to optimizing the query to make it more diverse and the retrieved knowledge is as different as possible."

        if len(search_history) >= tolerance - 1 and get_sentence_similarity(search_history[-1],
                                                                            search_history[-(tolerance - 1):-1],
                                                                            threshold=0.9) >= tolerance - 2:
            full_prompt += "\n\nPlease note! We have detected multiple very similar contents in the Knowledge section. Please optimize your query so that the retrieved knowledge is as different as possible."

        if len(query_history) >= tolerance - 1 and get_sentence_similarity(query_history[-1],
                                                                           query_history[-(tolerance - 1):-1],
                                                                           threshold=0.9) >= tolerance - 2:
            full_prompt += "\nPlease note that we have detected very similar content many times in the past query history. Please pay attention to optimizing the query to make it more diverse."

    model_response, usage = model.generate(full_prompt)

    answer_or_next_query = utils.extract_json_from_output(model_response)
    if answer_or_next_query is None:
        return None, None
    elif 'final_answer' in answer_or_next_query:
        return FinalAnswer(response=model_response, answer=answer_or_next_query['final_answer']), usage

    elif 'search_query' in answer_or_next_query:
        query = answer_or_next_query['search_query']
        if len(query_history) >= (tolerance-1) and get_sentence_similarity(query, query_history[-(tolerance-1):], threshold=0.9) >= tolerance-1:
            return '_Early_Stop', usage
        if len(search_history) >= tolerance and get_sentence_similarity(search_history[-1], search_history[-tolerance:-1],
                                                                    threshold=0.9) >= tolerance - 1:
            return '_Early_Stop', usage

        search_result = call_search(answer_or_next_query['search_query'], atomic_claim=atomic_claim)
        link = call_search(answer_or_next_query['search_query'], atomic_claim=atomic_claim, with_links=True)
        
        return GoogleSearchResult(query=answer_or_next_query['search_query'], result=search_result, link=link), usage
    else:
        print(f"Unexpected output: {answer_or_next_query}")
        return None, None
    
def must_get_final_answer(
        atomic_fact: str,
        searches: List[GoogleSearchResult],
        model: modeling.Model,
) -> Tuple[Optional[FinalAnswer], Optional[Dict]]:
    '''
    Handles cases where the model does not return a valid answer and re.sub cannot parse the answer.
    '''
    """At the end, the LLM must make a decision."""
    
    # Sort searches by temporal relevance for current claims
    if VIETNAMESE_SUPPORT and searches:
        # Use analyzer for consistency
        try:
            analyzer = get_analyzer()
            analysis = analyzer.analyze_claim(atomic_fact)
            is_current_claim = analysis.get('temporal_sensitivity') == 'high'
        except:
            # Fallback
            is_current_claim = any(keyword in atomic_fact.lower() for keyword in 
                                   ["hiện nay", "hiện tại", "now", "current", "bây giờ", "hôm nay"])
        
        if is_current_claim:
            sorted_searches = []
            for search in searches:
                validation = evidence_validator.validate_evidence(
                    evidence_text=search.result,
                    source_url=search.link if hasattr(search, 'link') and search.link else search.query,
                    claim=atomic_fact
                )
                sorted_searches.append((search, validation.get('temporal_score', 0.5), validation))
            sorted_searches = sorted(sorted_searches, key=lambda x: x[1], reverse=True)
            searches = [s[0] for s in sorted_searches]
            
            # Add date metadata
            knowledge_parts = []
            for i, (search, _, validation) in enumerate(sorted_searches, 1):
                knowledge_text = f"[Nguồn {i}]: {search.result}"
                if validation.get('evidence_date'):
                    knowledge_text = f"[Nguồn {i} - Ngày: {validation['evidence_date']}]: {search.result}"
                knowledge_parts.append(knowledge_text)
            knowledge = '\n\n'.join(knowledge_parts)
        else:
            knowledge = '\n'.join([search.result for search in searches])
    else:
        knowledge = '\n'.join([search.result for search in searches])
    
    full_prompt = format_verification_prompt(atomic_fact, knowledge, require_answer=True)
    full_prompt = utils.strip_string(full_prompt)

    try:
        model_response, usage = model.generate(full_prompt)
        
        if not model_response:
            return None, None
        
        answer = utils.extract_json_from_output(model_response)
        if not answer:
            return None, None
        
        if 'final_answer' in answer:
            final_answer = answer['final_answer']

        if final_answer in [FACTUAL_LABEL, NON_FACTUAL_LABEL, NEI_LABEL]:
            return FinalAnswer(response=model_response, answer=final_answer), usage
        else:
            return None, None
    except Exception as e:
        print(f"Error in must_get_final_answer: {e}")
        return None, None


def verify_atomic_claim(
        atomic_claim: str,
        rater: modeling.Model,
        max_steps: int = fire_config.max_steps,
        max_retries: int = fire_config.max_retries,
        diverse_prompt: bool = fire_config.diverse_prompt,
        tolerance: int = fire_config.max_tolerance,
) -> Tuple[Optional[FinalAnswer], Dict[str, Any], Optional[Dict]]:
    '''
    We verify the atomic_claims by interactively calling the tools.
    :param atomic_claim: The claim that we need to verify.
    :param rater: The backbone language model we choose.
    :param max_steps: The maximum step for calling tools.
    :param max_retries: The maximum tryouts for the LLM call for each step
    :return: FinalAnswer or None, search results, usage of tokens for verifying one atomic claim.
    '''
    
    # Analyze claim once at the beginning (cache for reuse)
    claim_analysis = None
    if VIETNAMESE_SUPPORT:
        try:
            analyzer = get_analyzer(rater)
            claim_analysis = analyzer.analyze_claim(atomic_claim, model=rater)
            print(f"Claim analysis: {claim_analysis.get('claim_type')}, sensitivity: {claim_analysis.get('temporal_sensitivity')}")
        except Exception as e:
            print(f"Claim analysis failed: {e}, will use fallback")
    
    preprocessed_claim = atomic_claim
    if VIETNAMESE_SUPPORT:
        try:
            processed_result = preprocessor.preprocess_claim(atomic_claim)
            if isinstance(processed_result, dict):
                preprocessed_claim = processed_result.get('normalized', atomic_claim)
            else:
                preprocessed_claim = str(processed_result)
            print(f"Preprocessed claim: {preprocessed_claim[:100]}...")
        except Exception as e:
            print(f"Preprocessing failed, using original claim: {e}")
            preprocessed_claim = atomic_claim
    
    search_results = []
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
    }

    stop_search = False
    # Increased from 1 to 2 for better accuracy - require at least 2 evidence sources
    min_searches = 2
    
    for step in range(max_steps):
        answer_or_next_search, num_tries = None, 0
        while not answer_or_next_search and num_tries <= max_retries:
            answer_or_next_search, usage = final_answer_or_next_search(atomic_claim, search_results, rater,
                                                                        diverse_prompt=diverse_prompt, tolerance=tolerance)
            if usage is not None:
                total_usage['input_tokens'] += usage['input_tokens']
                total_usage['output_tokens'] += usage['output_tokens']
            if answer_or_next_search == '_Early_Stop':
                stop_search = True
                break
            num_tries += 1
        if stop_search:
            break
        if answer_or_next_search is None:
            print(f'Maximum tryouts passed, still no answer or next search found.')
            break
        elif isinstance(answer_or_next_search, GoogleSearchResult):
            search_results.append(answer_or_next_search)
        elif isinstance(answer_or_next_search, FinalAnswer):
            if len(search_results) < min_searches:
                print(f"LLM tried to answer without search. Forcing search... (Step {step+1}/{max_steps})")
                
                # Use pre-analyzed claim data (already computed)
                if VIETNAMESE_SUPPORT and claim_analysis:
                    try:
                        # Use LLM-suggested query (more intelligent)
                        default_query = claim_analysis.get('suggested_query', '')
                        
                        # Add temporal context if needed
                        if claim_analysis.get('requires_recent_info'):
                            from datetime import datetime
                            current_year = datetime.now().year
                            default_query = f"{default_query} {current_year}"
                        elif claim_analysis.get('has_specific_date'):
                            # For date-specific claims, add "khi nào" or "ngày bầu"
                            import re
                            year_match = re.search(r'\b(19|20)\d{2}\b', atomic_claim)
                            if year_match:
                                default_query = f"{default_query} ngày bầu {year_match.group()}"
                            else:
                                default_query = f"{default_query} khi nào"
                        
                        print(f"Auto-generated query (LLM-analyzed): {default_query}")
                    except Exception as e:
                        print(f"Using fallback query: {e}")
                        # Fallback to simple entity extraction
                        try:
                            processed = preprocessor.preprocess_claim(atomic_claim)
                            entities = [e['text'] for e in processed.get('entities', [])[:3]]
                            default_query = ' '.join(entities) if entities else ' '.join(atomic_claim.split()[:6])
                        except:
                            default_query = ' '.join(atomic_claim.split()[:6])
                elif VIETNAMESE_SUPPORT:
                    # No claim_analysis, use fallback
                    try:
                        processed = preprocessor.preprocess_claim(atomic_claim)
                        entities = [e['text'] for e in processed.get('entities', [])[:3]]
                        default_query = ' '.join(entities) if entities else ' '.join(atomic_claim.split()[:6])
                    except:
                        default_query = ' '.join(atomic_claim.split()[:6])
                else:
                    # Non-Vietnamese: simple extraction
                    default_query = ' '.join(atomic_claim.split()[:6])
                
                search_result_text = call_search(default_query, atomic_claim=atomic_claim)
                link = call_search(default_query, atomic_claim=atomic_claim, with_links=True)
                
                search_results.append(GoogleSearchResult(query=default_query, result=search_result_text, link=link))
                continue
            
            if VIETNAMESE_SUPPORT:
                try:
                    evidence_scores = []
                    for search in search_results:
                        claim_text = preprocessed_claim if isinstance(preprocessed_claim, str) else str(preprocessed_claim)
                        
                        validation = evidence_validator.validate_evidence(
                            evidence_text=search.result,
                            source_url=search.query,
                            claim=claim_text
                        )
                        evidence_scores.append(validation['overall_score'])
                    
                    avg_evidence_quality = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.5
                    
                    claim_text = preprocessed_claim if isinstance(preprocessed_claim, str) else str(preprocessed_claim)
                    confidence = confidence_calibrator.calculate_confidence(
                        verdict=answer_or_next_search.answer,
                        iterations=len(search_results),
                        max_iterations=max_steps,
                        claim_length=len(claim_text.split()),
                        evidence_count=len(search_results),
                        evidence_quality=avg_evidence_quality
                    )
                    
                    is_confident = confidence_calibrator.is_confident(
                        confidence,
                        claim_complexity=len(claim_text.split())
                    )
                    
                    verdict_label = confidence_calibrator.get_verdict_label(
                        answer_or_next_search.answer,
                        confidence
                    )
                    
                    fact_check_db.save_verification(
                        claim=atomic_claim,
                        verdict=answer_or_next_search.answer,
                        confidence=confidence,
                        reasoning=answer_or_next_search.response,
                        model=rater.__class__.__name__ if hasattr(rater, '__class__') else 'unknown',
                        searches=[{'query': s.query, 'result': s.result} for s in search_results]
                    )
                    
                    
                except Exception as e:
                    import traceback
                    print(f"Vietnamese enhancements failed: {e}")
                    traceback.print_exc()
            
            search_dicts = {
                'google_searches': [dataclasses.asdict(s) for s in search_results]
            }
            return answer_or_next_search, search_dicts, total_usage

    final_answer, num_tries = None, 0
    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer, usage = must_get_final_answer(preprocessed_claim, searches=search_results, model=rater)
        if usage is not None:
            total_usage['input_tokens'] += usage['input_tokens']
            total_usage['output_tokens'] += usage['output_tokens']
    
    if VIETNAMESE_SUPPORT and final_answer:
        try:
            evidence_scores = []
            for search in search_results:
                validation = evidence_validator.validate_evidence(
                    evidence_text=search.result,
                    source_url=search.query,
                    claim=preprocessed_claim
                )
                evidence_scores.append(validation['overall_score'])
            
            avg_evidence_quality = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.5
            confidence = confidence_calibrator.calculate_confidence(
                verdict=final_answer.answer,
                iterations=len(search_results),
                max_iterations=max_steps,
                claim_length=len(preprocessed_claim.split()),
                evidence_count=len(search_results),
                evidence_quality=avg_evidence_quality
            )
            
            verdict_label = confidence_calibrator.get_verdict_label(
                final_answer.answer,
                confidence
            )
            
            fact_check_db.save_verification(
                claim=atomic_claim,
                verdict=final_answer.answer,
                confidence=confidence,
                reasoning=final_answer.response if hasattr(final_answer, 'response') else '',
                model=rater.__class__.__name__ if hasattr(rater, '__class__') else 'unknown',
                searches=[{'query': s.query, 'result': s.result} for s in search_results]
            )
            
        except Exception as e:
            print(f"Vietnamese final enhancements failed: {e}")
    
    search_dicts = {
        'google_searches': [dataclasses.asdict(s) for s in search_results]
    }
    return final_answer, search_dicts, total_usage

if __name__ == '__main__':
    pass