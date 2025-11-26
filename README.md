# Enhanced DeepDream Hallucinations Generator

## 🧠 Project Overview

This is an advanced AI-powered image hallucination generator that creates stunning, psychedelic dream-like transformations using multiple state-of-the-art neural architectures. The project features enhanced deep learning models including EfficientNetB7, ResNet152V2, and DenseNet201 with advanced processing techniques.

## ✨ Enhanced Features

### 🎨 Multiple Dream Styles
- **Psychedelic**: Vibrant, colorful patterns with high contrast
- **Artistic**: Painterly effects with balanced composition  
- **Surreal**: Abstract, dream-like transformations
- **Fractal**: Self-similar patterns with octave processing

### 🏗️ Neural Architectures
- **EfficientNetB7**: State-of-the-art efficiency and quality (Recommended)
- **ResNet152V2**: Deep residual networks for artistic effects
- **DenseNet201**: Dense connections for surreal transformations

### 🔧 Advanced Processing
- **Multi-scale dreaming**: Process images at multiple resolutions for richer effects
- **Octave processing**: Fractal-like, self-similar pattern generation
- **Quality analysis**: Real-time metrics for dream quality assessment
- **Batch processing**: Handle multiple images simultaneously
- **Advanced post-processing**: Color enhancement and artistic filters

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/sudarshanverma19/Hallucinating_Image-Generator.git
cd Hallucinating_Image-Generator
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. **Access the application**
- Main App: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

2. **For development with hot reload**
```bash
docker-compose --profile dev up --build
```

## 🌐 API Endpoints

### Enhanced Generation
```http
POST /generate/
```
**Parameters:**
- `file`: Image file (required)
- `style`: Dream style - `psychedelic`, `artistic`, `surreal`, `fractal`
- `model_type`: Neural architecture - `efficientnet`, `resnet`, `densenet`
- `intensity`: Effect intensity (0.1 - 2.0)
- `use_octave`: Enable octave processing (boolean)
- `steps`: Processing steps (50 - 300)
- `blend_original`: Original image blend ratio (0 - 1.0)

### Style Information
```http
GET /styles
```
Returns available dream styles and their configurations.

### Batch Processing
```http
POST /batch/
```
Process multiple images simultaneously (max 10 images).

## 👨‍💻 Author

**Sudarshan Verma**
- GitHub: [@sudarshanverma19](https://github.com/sudarshanverma19)
- Project: [Enhanced DeepDream Hallucinations Generator](https://github.com/sudarshanverma19/Hallucinating_Image-Generator)

---

**⚡ Ready to create amazing hallucinations? Upload an image and watch the magic happen!**
