from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import io
import os
import logging
import uuid
from PIL import Image
import numpy as np
import tensorflow as tf
from typing import Optional, List
from .models.enhanced_dreamer import EnhancedDeepDreamer, DreamStyleManager
from .utils.image_utils import ImagePreprocessor, PostProcessor, QualityAnalyzer

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app_config = {
    "title": "Enhanced DeepDream Hallucinations Generator",
    "version": "2.0.0",
    "description": "Advanced AI-powered hallucination generation with multiple neural architectures and dream styles"
}

if ENVIRONMENT == "production":
    app_config["docs_url"] = None  # Disable docs in production
    app_config["redoc_url"] = None

app = FastAPI(**app_config)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models for API
class DreamRequest(BaseModel):
    style: str = "psychedelic"
    model_type: str = "efficientnet"
    intensity: float = 1.0
    use_octave: bool = False
    steps: int = 100
    blend_original: float = 0.7

class BatchDreamRequest(BaseModel):
    style: str = "psychedelic"
    model_type: str = "efficientnet"
    intensity: float = 1.0
    use_octave: bool = False

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint for Docker and AWS load balancers"""
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "version": "1.0.0"
    }

@app.get("/api")
def api_info():
    return {
        "message": "Welcome to Enhanced DeepDream Hallucinations Generator API",
        "environment": ENVIRONMENT,
        "version": "2.0.0",
        "available_models": ["efficientnet", "resnet", "densenet"],
        "available_styles": DreamStyleManager.list_styles(),
        "features": [
            "Multi-scale dreaming",
            "Octave processing",
            "Advanced post-processing",
            "Batch processing",
            "Quality analysis"
        ]
    }

@app.get("/styles")
def get_available_styles():
    """Get all available dream styles and their descriptions"""
    styles = {}
    for style_name in DreamStyleManager.list_styles():
        config = DreamStyleManager.get_style_config(style_name)
        styles[style_name] = {
            "description": config["description"],
            "recommended_model": config["recommended_model"],
            "intensity_range": config["intensity_range"]
        }
    return {"styles": styles}

@app.post("/generate/")
async def generate_dream(
    file: UploadFile = File(...),
    style: str = Form("psychedelic"),
    model_type: str = Form("efficientnet"),
    intensity: float = Form(1.0),
    use_octave: bool = Form(False),
    steps: int = Form(100),
    blend_original: float = Form(0.7)
):
    """Generate enhanced deep dream with advanced controls"""
    try:
        # Validate inputs
        if style not in DreamStyleManager.list_styles():
            raise HTTPException(status_code=400, detail=f"Invalid style. Available: {DreamStyleManager.list_styles()}")
        
        if model_type not in ["efficientnet", "resnet", "densenet"]:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use: efficientnet, resnet, or densenet")
        
        # Read and preprocess image
        input_image = Image.open(file.file).convert("RGB")
        input_array = np.array(input_image)
        
        # Prepare image for processing
        processed_img = ImagePreprocessor.prepare_image(file.file, target_size=(512, 512))
        
        # Initialize dreamer
        dreamer = EnhancedDeepDreamer(model_type=model_type, dream_style=style)
        
        # Generate dream
        if use_octave:
            result = dreamer.octave_dream(processed_img, octaves=4)
        else:
            result = dreamer.multi_scale_dream(processed_img)
        
        # Post-process
        result = PostProcessor.apply_color_enhancement(result, saturation=1.0 + intensity * 0.3)
        result = PostProcessor.apply_artistic_filter(result, "smooth")
        
        # Blend with original if requested
        if blend_original < 1.0:
            original_resized = tf.image.resize(processed_img, result.shape[:2])
            result = PostProcessor.blend_with_original(original_resized, result, blend_original)
        
        # Convert to PIL and return
        output_image = Image.fromarray(result.numpy())
        
        buf = io.BytesIO()
        output_image.save(buf, format="PNG", quality=95)
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error in enhanced generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dream generation failed: {str(e)}")

@app.post("/analyze/")
async def analyze_dream_quality(
    original: UploadFile = File(...),
    dreamed: UploadFile = File(...)
):
    """Analyze the quality of a dreamed image compared to original"""
    try:
        # Load images
        original_img = np.array(Image.open(original.file).convert("RGB"))
        dreamed_img = np.array(Image.open(dreamed.file).convert("RGB"))
        
        # Resize to same dimensions if needed
        if original_img.shape != dreamed_img.shape:
            target_size = min(original_img.shape[:2])
            original_img = tf.image.resize(original_img, [target_size, target_size]).numpy()
            dreamed_img = tf.image.resize(dreamed_img, [target_size, target_size]).numpy()
        
        # Analyze quality
        quality_metrics = QualityAnalyzer.calculate_dream_score(original_img, dreamed_img)
        
        return JSONResponse(content={
            "analysis": quality_metrics,
            "recommendation": _get_quality_recommendation(quality_metrics["quality_score"])
        })
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/batch/")
async def batch_generate_dreams(
    files: List[UploadFile] = File(...),
    style: str = Form("psychedelic"),
    model_type: str = Form("efficientnet"),
    intensity: float = Form(1.0)
):
    """Generate dreams for multiple images"""
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
        
        results = []
        dreamer = EnhancedDeepDreamer(model_type=model_type, dream_style=style)
        
        for i, file in enumerate(files):
            try:
                # Process each image
                input_image = Image.open(file.file).convert("RGB")
                processed_img = ImagePreprocessor.prepare_image(file.file, target_size=(512, 512))
                
                # Generate dream
                result = dreamer.multi_scale_dream(processed_img)
                result = PostProcessor.apply_color_enhancement(result, saturation=1.0 + intensity * 0.3)
                
                # Save to buffer
                output_image = Image.fromarray(result.numpy())
                buf = io.BytesIO()
                output_image.save(buf, format="PNG", quality=95)
                buf.seek(0)
                
                # Encode as base64 for JSON response
                import base64
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "image_data": img_b64,
                    "status": "success"
                })
                
            except Exception as img_error:
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "error": str(img_error),
                    "status": "failed"
                })
        
        return JSONResponse(content={
            "results": results,
            "total_processed": len([r for r in results if r["status"] == "success"]),
            "total_failed": len([r for r in results if r["status"] == "failed"])
        })
        
    except Exception as e:
        logger.error(f"Error in batch endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

def _get_quality_recommendation(score: float) -> str:
    """Get recommendation based on quality score"""
    if score >= 0.8:
        return "Excellent dream quality with rich patterns and details"
    elif score >= 0.6:
        return "Good dream quality, consider increasing intensity for more effects"
    elif score >= 0.4:
        return "Moderate dream quality, try different style or model"
    else:
        return "Low dream quality, consider preprocessing image or adjusting parameters"
