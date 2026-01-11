from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent directory to path to import fire modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.fire.verify_atomic_claim import verify_claim

app = FastAPI(title="Fact Checking API")

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FactCheckRequest(BaseModel):
    claim: str

class FactCheckResponse(BaseModel):
    verdict: str  # "Đúng", "Sai", "Chưa rõ"
    explanation: str
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "Fact Checking API is running"}

@app.post("/api/check", response_model=FactCheckResponse)
async def check_fact(request: FactCheckRequest):
    """
    Kiểm tra tính chính xác của một thông tin
    """
    try:
        # Call FIRE verification system
        result = await verify_claim(request.claim)
        
        # Map result to Vietnamese verdict
        verdict_map = {
            "supported": "Đúng",
            "refuted": "Sai",
            "not enough info": "Chưa rõ"
        }
        
        verdict = verdict_map.get(result.get("label", "not enough info"), "Chưa rõ")
        
        return FactCheckResponse(
            verdict=verdict,
            explanation=result.get("explanation", "Không có giải thích"),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.5)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking fact: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
