from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .inference import VulnerabilityAnalyzer
from fastapi.responses import HTMLResponse
import os


class CodeRequest(BaseModel):
    """Request model for code to be analyzed."""
    code: str
    threshold: float = 0.6

class LinePrediction(BaseModel):
    """Response model for a single line's prediction."""
    line: int
    vulnerable: int

class AnalysisResponse(BaseModel):
    """Response model for the entire analysis."""
    predictions: List[LinePrediction]


app = FastAPI(
    title="Line Vulnerability Analyzer API",
    description="API to analyzce code snippets for vulnerability",
    version="1.0.0",
)

analyzer = VulnerabilityAnalyzer()

# Path to the templates directory
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

@app.get("/", summary="Root Endpoint", description="A simple endpoint to check if the API is running")
async def read_gui():
    """ Serves the main HTML for the application. """
    with open(os.path.join(templates_dir, 'index.html'), 'r') as file:
        html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)

@app.post("/analyze", response_model=AnalysisResponse, summary="Analyze Code Snippet", tags=["API"])
async def analyze_code(request: CodeRequest):
    """Analyzes the provided code snippet for vulnerabilities."""
    predictions = analyzer.predict(request.code, threshold=request.threshold)
    line_predictions = [LinePrediction(**pred) for pred in predictions]
    return AnalysisResponse(predictions=line_predictions)





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)