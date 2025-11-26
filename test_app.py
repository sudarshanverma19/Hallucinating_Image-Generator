"""
Deep Dreams Hallucination Generator - Original Working Version
Creates beautiful image effects using PIL
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import io
import os
from PIL import Image, ImageEnhance, ImageFilter
import uvicorn

def generate_simple_dream(image, style="psychedelic"):
    """Generate simple dream effects using PIL only"""
    
    if style == "psychedelic":
        # Enhanced colors and contrast
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Apply filters
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        image = image.filter(ImageFilter.SMOOTH)
        
    elif style == "artistic":
        # Artistic painting effect
        image = image.filter(ImageFilter.SMOOTH_MORE)
        image = image.filter(ImageFilter.DETAIL)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.8)
        
    elif style == "surreal":
        # Surreal dream-like effect
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        image = image.filter(ImageFilter.SHARPEN)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(2.5)
        
    elif style == "fractal":
        # Fractal-like effect
        for i in range(3):
            image = image.filter(ImageFilter.FIND_EDGES)
            image = image.filter(ImageFilter.SMOOTH)
            
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.5)
    
    return image

# Initialize FastAPI app
app = FastAPI(
    title="Simple Deep Dream Generator",
    description="PIL-based image hallucination effects",
    version="1.0.0"
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Deep Dream Generator</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .container { text-align: center; }
                input, select, button { margin: 10px; padding: 10px; }
                #result { margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎨 Simple Deep Dream Generator</h1>
                <p>Create beautiful hallucination effects!</p>
                
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
                    <button type="submit">Generate Dream</button>
                </form>
                
                <div id="result"></div>
            </div>
            
            <script>
                document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData();
                    formData.append('file', document.getElementById('imageFile').files[0]);
                    formData.append('style', document.getElementById('style').value);
                    
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
    return {"status": "healthy", "version": "PIL-only simple version"}

@app.get("/styles")
async def get_styles():
    """Get available dream styles"""
    return {
        "styles": [
            {"name": "psychedelic", "description": "Colorful, enhanced reality"},
            {"name": "artistic", "description": "Painting-like effects"},
            {"name": "surreal", "description": "Dream-like surreal effects"},
            {"name": "fractal", "description": "Repetitive patterns"}
        ]
    }

@app.post("/generate")
async def generate_dream(
    file: UploadFile = File(...),
    style: str = Form("psychedelic")
):
    """Generate simple dream effects"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
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
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Simple Deep Dream Generator...")
    print("🎨 PIL-only effects - no complex dependencies!")
    uvicorn.run(app, host="0.0.0.0", port=5000)