"""
FastAPI service for safety classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from safety_classifier import safety_classifier

app = FastAPI(title="Safety Classification API", version="1.0.0")

class ClassificationRequest(BaseModel):
    items: str
    alpha: float = 0.05

class ClassificationResponse(BaseModel):
    results: List[float]
    classified_items: List[Dict[str, Any]]

@app.post("/classify_items", response_model=ClassificationResponse)
def classify_items(request: ClassificationRequest):
    """
    Classify items as safe or unsafe using zero-shot classification
    """
    try:
        # Split items by $$
        candidates = request.items.split("$$")
        
        if not candidates or not candidates[0].strip():
            raise HTTPException(status_code=400, detail="No items provided")
        
        # Classify each item
        candidates_labels = ["dangerous", "safe"]
        probabilistic_outputs = []
        classified_items = []
        
        for i, item in enumerate(candidates):
            item = item.strip()
            if not item:
                continue
                
            try:
                result = safety_classifier.pipe(item, candidates_labels)
                dangerous_score = result["scores"][0]
                probabilistic_outputs.append(dangerous_score)
                
                # Create classified item info
                classified_item = {
                    "item": item,
                    "dangerous_score": round(dangerous_score, 3),
                    "safe_score": round(1.0 - dangerous_score, 3),
                    "is_dangerous": dangerous_score >= (1.0 - request.alpha),
                    "confidence": round(max(dangerous_score, 1.0 - dangerous_score), 3)
                }
                classified_items.append(classified_item)
                
            except Exception as e:
                print(f"Error classifying item '{item}': {e}")
                probabilistic_outputs.append(0.0)  # Default to safe
                classified_items.append({
                    "item": item,
                    "dangerous_score": 0.0,
                    "safe_score": 1.0,
                    "is_dangerous": False,
                    "confidence": 1.0,
                    "error": str(e)
                })
        
        # Filter results based on alpha threshold
        filtered_results = [score for score in probabilistic_outputs if score >= (1.0 - request.alpha)]
        
        return ClassificationResponse(
            results=filtered_results,
            classified_items=classified_items
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify_objects", response_model=List[Dict[str, Any]])
def classify_objects(objects_data: List[Dict[str, Any]], alpha: float = 0.05):
    """
    Classify a list of objects with coordinates as safe or unsafe
    """
    try:
        if not safety_classifier.initialized:
            raise HTTPException(status_code=503, detail="Safety classifier not available")
        
        classified_objects = safety_classifier.classify_objects(objects_data, alpha)
        return classified_objects
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "classifier_initialized": safety_classifier.initialized
    }

if __name__ == "__main__":
    print("Starting Safety Classification API...")
    print("API will be available at: http://localhost:8001")
    print("API documentation at: http://localhost:8001/docs")
    
    uvicorn.run(
        "safety_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
