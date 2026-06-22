from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field 
from transformers import pipeline 
from typing import Dict, List 
from contextlib import asynccontextmanager 
import torch 
import logging 
import time 
from datetime import datetime, timezone 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sentiment-api")
MODEL_ID = "chabdullah0566/my-sentiment-analyzer" # <-- replace with your Phase 5 repo
ml_models: Dict[str, object] = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model: {MODEL_ID} ...")
    try:
        ml_models["classifier"] = pipeline(
          "sentiment-analysis",
          model=MODEL_ID,
          device=0 if torch.cuda.is_available() else -1,
        )
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        ml_models["classifier"] = None
    yield
    ml_models.clear()
app = FastAPI(
    title="Sentiment Analysis API",
    description="A DistilBERT-based sentiment classifier.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50, description="List of texts (max 50)")

class SentimentResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start 
    logger.info(f'{request.method} {request.url.path} -> {response.status_code} ({duration:.3f}s)')
    return response 

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API", "docs": "/docs", "health": "/health"}

@app.get("/health")
async def health_check():
    return {
       "status": "healthy" if ml_models.get("classifier") is not None else "degraded",
        "model_loaded": ml_models.get("classifier") is not None,
        "model_id": MODEL_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    classifier = ml_models.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health and server logs.")
    try:
        result = classifier(request.text)[0]
        probs = {result['label']: round(result['score'], 4)}
        other_label = 'NEGATIVE' if result['label'] == 'POSITIVE' else 'POSITIVE'
        probs[other_label] = round(1 - result['score'], 4)
        return SentimentResponse(
            text=request.text[:200],
            label=result['label'],
            confidence=round(result['score'], 4),
            probabilities=probs,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal error during prediction.")

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    classifier = ml_models.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health and server logs.")
    try:
        results = classifier(request.texts)
        return {
            "results": [
                {"text": text[:100], "label": r['label'], "confidence": round(r['score'], 4)}
                for text, r in zip(request.texts, results)
            ],
            "total": len(results),
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal error during batch prediction.")
if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)