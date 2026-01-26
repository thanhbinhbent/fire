from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from common.modeling import Model
    from common import shared_config
    from common.vietnamese_utils import preprocessor
    from common.database import db
    from common.confidence import calibrator
    from common.query_deduplication import deduplicator
    from common.evidence_validator import validator
    from eval.fire.verify_atomic_claim import verify_atomic_claim
    VIETNAMESE_SUPPORT = True
except ImportError as e:
    print(f"Import error: {e}")
    VIETNAMESE_SUPPORT = False
    verify_atomic_claim = None

rater = None

try:
    os.environ["OPENAI_API_KEY"] = shared_config.openai_api_key
    os.environ["ANTHROPIC_API_KEY"] = shared_config.anthropic_api_key
    os.environ["SERPER_API_KEY"] = shared_config.serper_api_key
    os.environ["GROQ_API_KEY"] = shared_config.groq_api_key
    os.environ["GEMINI_API_KEY"] = shared_config.gemini_api_key
    
    rater = Model(
        model_name=shared_config.default_model_name,
        temperature=shared_config.default_temperature,
        max_tokens=shared_config.default_max_tokens
    )
    print(f"Model: {shared_config.default_model_name}, Temp: {shared_config.default_temperature}")
except Exception as e:
    print(f"Model init failed: {e}")


class FactCheckRequest(BaseModel):
    claim: str
    model: Optional[str] = None


class FactCheckResponse(BaseModel):
    verdict: str
    explanation: str
    sources: Optional[List[Dict]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Vietnamese Fact Checking API",
        "model": shared_config.default_model_name,
        "vietnamese_support": VIETNAMESE_SUPPORT,
    }


@app.post("/api/check", response_model=FactCheckResponse)
async def check_fact(request: FactCheckRequest):
    if not rater:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        model_instance = rater
        
        if request.model and request.model != shared_config.default_model_name:
            try:
                model_instance = Model(
                    model_name=request.model,
                    temperature=shared_config.default_temperature,
                    max_tokens=shared_config.default_max_tokens
                )
            except Exception as e:
                print(f"Failed to load custom model: {e}, using default")
        
        processed = None
        entities = []
        if VIETNAMESE_SUPPORT:
            try:
                processed = preprocessor.preprocess_claim(request.claim)
                entities = [e['text'] for e in processed['entities']]
            except Exception as e:
                print(f"Preprocessing error: {e}")
        
        if verify_atomic_claim and VIETNAMESE_SUPPORT:
            print(f"Starting verification: {request.claim}")
            
            final_answer, search_results, usage = verify_atomic_claim(
                atomic_claim=request.claim,
                rater=model_instance,
                max_steps=4,  # Increased from 3 for better accuracy
                max_retries=2,
                diverse_prompt=True,
                tolerance=3  # Increased from 2 to allow more diverse searches
            )
            
            google_searches = search_results.get('google_searches', [])
            print(f"Verification complete. Searches: {len(google_searches)}")
            
            if final_answer:
                verdict = final_answer.answer
                raw_response = final_answer.response
                
                verdict_map = {
                    "True": "Đúng",
                    "False": "Sai",
                    "Not Enough Info": "Chưa đủ thông tin",
                    "NOT ENOUGH INFO": "Chưa đủ thông tin",
                    "Supported": "Đúng",
                    "Refuted": "Sai",
                    "NEI": "Chưa đủ thông tin",
                }
                verdict_vn = verdict_map.get(verdict, "Chưa rõ")
                
                explanation = raw_response
                
                import re
                lines = explanation.split('\n')
                cleaned_lines = [line for line in lines if not re.match(r'^\s*\{[^}]*"(final_answer|search_query)"[^}]*\}\s*$', line)]
                explanation = '\n'.join(cleaned_lines).strip()
                
                sources = []
                for idx, search in enumerate(google_searches):
                    validation = validator.validate_evidence(
                        evidence_text=search.get('result', ''),
                        source_url=search.get('query', ''),
                        claim=request.claim
                    )
                    
                    result_text = search.get('result', '')
                    
                    # Use direct link from search result if available, otherwise try to extract from text
                    source_url = search.get('link', '')
                    if not source_url:
                        import re
                        url_match = re.search(r'https?://[^\s]+', result_text)
                        source_url = url_match.group(0) if url_match else f"https://google.com/search?q={search.get('query', '')}"
                    
                    # Extract title more intelligently
                    title = None
                    # Try to extract domain from parentheses like "(Chinhphu.vn)"
                    domain_match = re.match(r'\(([^)]+)\)\s*[-–—]', result_text)
                    if domain_match:
                        title = domain_match.group(1)
                    else:
                        # Try to get text before " - ", ". " or just first 50 chars
                        if ' - ' in result_text:
                            title = result_text.split(' - ')[0]
                        elif '. ' in result_text:
                            title = result_text.split('. ')[0]
                        else:
                            title = result_text[:50]
                    
                    # Clean up and limit title length
                    title = title.strip('()')
                    title = title[:100] if len(title) > 100 else title
                    
                    sources.append({
                        "url": source_url,
                        "title": title or f"Search result #{idx+1}",
                        "snippet": result_text[:200] + "..." if len(result_text) > 200 else result_text,
                        "credibility": validation.get('credibility_score', 0.5),
                        "relevance": validation.get('relevance_score', 0.5),
                    })
                
                evidence_quality = sum(s['credibility'] * s['relevance'] for s in sources) / len(sources) if sources else 0.5
                confidence_data = calibrator.calculate_confidence(
                    verdict=verdict,
                    iterations=len(search_results),
                    max_iterations=3,
                    claim_length=len(request.claim.split()),
                    evidence_count=len(search_results),
                    evidence_quality=evidence_quality
                )
                
                verdict_label = calibrator.get_verdict_label(verdict, confidence_data)
                
                metadata = {
                    "preprocessing": {
                        "normalized": processed['normalized'] if processed else request.claim,
                        "entities": entities,
                        "token_count": processed['token_count'] if processed else len(request.claim.split()),
                    },
                    "verification": {
                        "searches": len(search_results),
                        "tokens_used": usage.get('input_tokens', 0) + usage.get('output_tokens', 0) if usage else 0,
                        "raw_verdict": verdict,
                    },
                    "vietnamese_support": VIETNAMESE_SUPPORT,
                }
                
                return FactCheckResponse(
                    verdict=verdict_label,
                    explanation=explanation,
                    sources=sources,
                    confidence=confidence_data,
                    metadata=metadata
                )
            else:
                raise HTTPException(status_code=500, detail="Verification failed")
        
        else:
            raise HTTPException(status_code=503, detail="Verification not available")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model": shared_config.default_model_name,
        "model_loaded": rater is not None,
    }


@app.get("/api/stats")
async def get_stats():
    stats = {
        "model": shared_config.default_model_name,
        "vietnamese_support": VIETNAMESE_SUPPORT,
    }
    
    if VIETNAMESE_SUPPORT:
        try:
            stats["deduplication"] = deduplicator.get_stats()
        except Exception:
            pass
    
    return stats


if __name__ == "__main__":
    import uvicorn
    print("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

