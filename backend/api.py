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
    print(f"Some Vietnamese components not available: {e}")
    VIETNAMESE_SUPPORT = False
    verify_atomic_claim = None

app = FastAPI(title="Vietnamese Fact Checking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    print(f"Model initialized: {shared_config.default_model_name}")
    print(f"Temperature: {shared_config.default_temperature}")
    print(f"Max tokens: {shared_config.default_max_tokens}")
except Exception as e:
    print(f"Model initialization failed: {e}")


class FactCheckRequest(BaseModel):
    claim: str
    model: Optional[str] = None


class FactCheckResponse(BaseModel):
    verdict: str
    explanation: str
    sources: Optional[List[Dict]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


@app.get("/")
async def root():
    return {
        "message": "Vietnamese Fact Checking API",
        "version": "2.0.0 (Enhanced Architecture)",
        "model": shared_config.default_model_name,
        "temperature": shared_config.default_temperature,
        "vietnamese_support": VIETNAMESE_SUPPORT,
        "features": [
            "Three-layer Vietnamese optimization",
            "Embedding-based query deduplication",
            "Dynamic confidence calibration",
            "Vietnamese-aware evidence validation"
        ]
    }


@app.post("/api/check", response_model=FactCheckResponse)
async def check_fact(request: FactCheckRequest):
    """
    Kiểm tra tính chính xác của thông tin bằng tiếng Việt.
    
    THREE-LAYER VIETNAMESE ARCHITECTURE:
    1. Preprocessing: Normalize, extract entities
    2. Query Generation: Deduplicate, enhance for Vietnamese
    3. Evidence Validation: Score credibility and relevance
    """
    if not rater:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        model_instance = rater
        if request.model:
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
                print(f"Preprocessed: {processed['normalized']}")
                print(f"🏷️ Entities: {entities}")
            except Exception as e:
                print(f"Preprocessing error: {e}")
        
        if verify_atomic_claim and VIETNAMESE_SUPPORT:
            print(f"🔍 Starting verification for: {request.claim}")
            
            final_answer, search_results, usage = verify_atomic_claim(
                atomic_claim=request.claim,
                rater=model_instance,
                max_steps=3,
                max_retries=2,
                diverse_prompt=True,
                tolerance=2
            )
            
            google_searches = search_results.get('google_searches', [])
            print(f"📊 Verification complete. Searches: {len(google_searches)}")
            
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
                    import re
                    url_match = re.search(r'https?://[^\s]+', result_text)
                    source_url = url_match.group(0) if url_match else f"https://google.com/search?q={search.get('query', '')}"
                    
                    title_match = result_text.split('.')[0] if '.' in result_text else search.get('query', '')
                    title = title_match[:100] if len(title_match) > 100 else title_match
                    
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
                raise HTTPException(status_code=500, detail="Verification failed - no final answer")
        
        else:
            verdict = "Đúng (Chắc chắn)"
            explanation = f"Mock response - Claim đã được xử lý: {processed['normalized'] if processed else request.claim}"
            confidence = 0.75
            
            sources = [
                {
                    "url": "https://vnexpress.net",
                    "title": "Example source (Mock)",
                    "snippet": "This is mock data. Real verification not available.",
                    "credibility": 1.0,
                    "relevance": 0.8,
                }
            ]
            
            metadata = {
                "preprocessing": {
                    "normalized": processed['normalized'] if processed else request.claim,
                    "entities": entities,
                    "token_count": processed['token_count'] if processed else len(request.claim.split()),
                },
                "vietnamese_support": VIETNAMESE_SUPPORT,
                "mode": "mock"
            }
            
            return FactCheckResponse(
                verdict=verdict,
                explanation=explanation,
                sources=sources,
                confidence=confidence,
                metadata=metadata
            )
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "model": shared_config.default_model_name,
        "temperature": shared_config.default_temperature,
        "max_tokens": shared_config.default_max_tokens,
        "model_loaded": rater is not None,
    }
    
    if VIETNAMESE_SUPPORT:
        status["vietnamese_components"] = {
            "preprocessor": "loaded",
            "database": "loaded",
            "calibrator": "loaded",
            "deduplicator": "loaded",
            "validator": "loaded",
        }
    
    return status


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    stats = {
        "model": shared_config.default_model_name,
        "temperature": shared_config.default_temperature,
        "vietnamese_support": VIETNAMESE_SUPPORT,
    }
    
    if VIETNAMESE_SUPPORT:
        try:
            stats["deduplication"] = deduplicator.get_stats()
            stats["database"] = {
                "path": db.db_path,
                "recent_verifications": len(db.get_recent_verifications(10))
            }
        except Exception as e:
            stats["error"] = str(e)
    
    return stats


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Vietnamese Fact Checking API...")
    print(f"📍 Model: {shared_config.default_model_name}")
    print(f"Temperature: {shared_config.default_temperature}")
    print(f"🇻🇳 Vietnamese Support: {VIETNAMESE_SUPPORT}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

