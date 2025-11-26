"""
Simple FastAPI app for testing without TensorFlow
"""
from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import os
import logging
from PIL import Image
import numpy as np
from simple_processor import generate_simple_dream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Simple DeepDream Generator (Test Version)",
    version="1.0.0",
    description="Simple image effects for testing without TensorFlow"
)

# Mount static files
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Templates
templates = Jinja2Templates(directory="../templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0 (Simple Test Mode)",
        "note": "Running without TensorFlow - using simple image effects"
    }

@app.get("/api")
def api_info():
    return {
        "message": "Simple DeepDream Generator (Test Version)",
        "available_styles": ["psychedelic", "artistic", "surreal", "fractal"],
        "note": "This is a test version without TensorFlow"
    }

@app.get("/styles")
def get_available_styles():
    """Get all available styles"""
    return {
        "styles": {
            "psychedelic": {"description": "Colorful and vibrant effects"},
            "artistic": {"description": "Painting-like artistic effects"},
            "surreal": {"description": "Dream-like color shifts"},
            "fractal": {"description": "Pattern-enhanced effects"}
        }
    }

@app.post("/generate/")
async def generate_dream(
    file: UploadFile = File(...),
    style: str = Form("psychedelic")
):
    """Generate simple dream effects"""
    try:
        # Read and process image
        input_image = Image.open(file.file).convert("RGB")
        
        # Resize if too large
        max_size = 800
        if max(input_image.size) > max_size:
            input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Apply simple dream effect
        result_image = generate_simple_dream(input_image, style)
        
        # Return as PNG
        buf = io.BytesIO()
        result_image.save(buf, format="PNG", quality=95)
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)