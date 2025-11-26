"""
Simple PIL-Only Deep Dream Generator - No Complex Dependencies!
Works perfectly with any Python version.
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import io
import os
from PIL import Image, ImageEnhance, ImageFilter
import uvicorn

# Simple PIL-based dream processor
class SimpleDreamProcessor:
    """Pure PIL-based dream effects - no external dependencies needed!"""
    
    def __init__(self):
        print("🎨 Simple PIL Dream Processor initialized!")
    
    def psychedelic_effect(self, image, intensity=1.0):
        """Create psychedelic effects using PIL"""
        # Color enhancement
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.5 + intensity)
        
        # Contrast boost
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3 + intensity * 0.5)
        
        # Apply blur for dreamy effect
        image = image.filter(ImageFilter.GaussianBlur(radius=intensity))
        
        # Edge enhancement
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        return image
    
    def artistic_effect(self, image, intensity=1.0):
        """Create artistic painting-like effects"""
        # Smooth the image
        image = image.filter(ImageFilter.SMOOTH_MORE)
        
        # Enhance details
        image = image.filter(ImageFilter.DETAIL)
        
        # Color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.8 + intensity * 0.3)
        
        # Slight blur for painting effect
        image = image.filter(ImageFilter.GaussianBlur(radius=intensity * 0.5))
        
        return image
    
    def surreal_effect(self, image, intensity=1.0):
        """Create surreal dream-like effects"""
        # High contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0 + intensity)
        
        # Sharp edges
        image = image.filter(ImageFilter.SHARPEN)
        
        # Color shift
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(2.5 + intensity * 0.5)
        
        # Motion blur effect
        image = image.filter(ImageFilter.GaussianBlur(radius=intensity * 0.8))
        
        return image
    
    def fractal_effect(self, image, intensity=1.0):
        """Create fractal-like repetitive patterns"""
        # Multiple filter applications
        for i in range(int(2 + intensity)):
            image = image.filter(ImageFilter.FIND_EDGES)
            image = image.filter(ImageFilter.SMOOTH)
            
            # Color enhancement
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.5)
        
        return image
    
    def generate_dream(self, image, style="psychedelic", intensity=1.0):
        """Generate dream effects based on style"""
        if style == "psychedelic":
            return self.psychedelic_effect(image, intensity)
        elif style == "artistic":
            return self.artistic_effect(image, intensity)
        elif style == "surreal":
            return self.surreal_effect(image, intensity)
        elif style == "fractal":
            return self.fractal_effect(image, intensity)
        else:
            return self.psychedelic_effect(image, intensity)

# Initialize the app
app = FastAPI(
    title="Simple PIL Dream Generator",
    description="Pure PIL-based image hallucination effects - Works everywhere!",
    version="1.0.0"
)

# Initialize processor
dream_processor = SimpleDreamProcessor()

# Static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page"""
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple PIL Dream Generator</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .container { text-align: center; }
                input, select, button { margin: 10px; padding: 10px; }
                #result { margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎨 Simple PIL Dream Generator</h1>
                <p>No complex dependencies - Pure PIL effects!</p>
                
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="imageFile" accept="image/*" required>
                    <br>
                    <select id="style">
                        <option value="psychedelic">Psychedelic</option>
                        <option value="artistic">Artistic</option>
                        <option value="surreal">Surreal</option>
                        <option value="fractal">Fractal</option>
                    </select>
                    <br>
                    <label>Intensity: <input type="range" id="intensity" min="0.1" max="3.0" step="0.1" value="1.0"></label>
                    <span id="intensityValue">1.0</span>
                    <br>
                    <button type="submit">Generate Dream</button>
                </form>
                
                <div id="result"></div>
            </div>
            
            <script>
                document.getElementById('intensity').addEventListener('input', function() {
                    document.getElementById('intensityValue').textContent = this.value;
                });
                
                document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData();
                    formData.append('file', document.getElementById('imageFile').files[0]);
                    formData.append('style', document.getElementById('style').value);
                    formData.append('intensity', document.getElementById('intensity').value);
                    
                    const button = document.querySelector('button');
                    button.textContent = 'Processing...';
                    button.disabled = true;
                    
                    try {
                        const response = await fetch('/generate', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const blob = await response.blob();
                            const url = URL.createObjectURL(blob);
                            document.getElementById('result').innerHTML = 
                                `<img src="${url}" style="max-width: 100%; border-radius: 10px;">`;
                        } else {
                            throw new Error('Processing failed');
                        }
                    } catch (error) {
                        document.getElementById('result').innerHTML = 
                            `<p style="color: red;">Error: ${error.message}</p>`;
                    } finally {
                        button.textContent = 'Generate Dream';
                        button.disabled = false;
                    }
                });
            </script>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "processor": "PIL-only (no complex dependencies)",
        "styles": ["psychedelic", "artistic", "surreal", "fractal"]
    }

@app.get("/styles")
async def get_styles():
    """Get available styles"""
    return {
        "styles": [
            {"name": "psychedelic", "description": "Colorful, enhanced reality effects"},
            {"name": "artistic", "description": "Painting-like artistic effects"},
            {"name": "surreal", "description": "Surreal, dream-like transformations"},
            {"name": "fractal", "description": "Repetitive, pattern-based effects"}
        ]
    }

@app.post("/generate")
async def generate_dream(
    file: UploadFile = File(...),
    style: str = Form("psychedelic"),
    intensity: float = Form(1.0)
):
    """Generate simple PIL-based dream effects"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        input_image = Image.open(file.file).convert("RGB")
        
        # Resize if too large (keep it reasonable)
        max_size = 800
        if max(input_image.size) > max_size:
            input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Apply dream effect
        result_image = dream_processor.generate_dream(input_image, style, intensity)
        
        # Return as PNG
        buf = io.BytesIO()
        result_image.save(buf, format="PNG", quality=95)
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Simple PIL Dream Generator...")
    print("📝 This version works with any Python version!")
    print("🎨 Pure PIL effects - no complex dependencies needed")
    
    # Try different ports
    for port in [5000, 8000, 3000, 8080]:
        try:
            print(f"🌐 Trying to start server on http://localhost:{port}")
            uvicorn.run(app, host="0.0.0.0", port=port)
            break
        except Exception as e:
            print(f"❌ Port {port} failed: {e}")
            continue