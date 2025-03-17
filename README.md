# 🎨 AI Image Generation with FLUX

Welcome to an exciting world of AI-powered image generation! This project leverages the powerful FLUX model from black-forest-labs to create stunning, high-quality images from text descriptions. Whether you're an artist, developer, or just curious about AI art, this tool provides an intuitive interface for your creative endeavors!

## ✨ Features

- 🖼️ Generate high-quality images from text descriptions
- 🎯 Multiple aspect ratio support (Square, Landscape, Portrait)
- 🔄 Auto-update system with version tracking
- 🎚️ Configurable generation parameters
- 🎮 User-friendly Gradio interface
- 📊 Token-based usage system (optional)
- 💾 Automatic model management and caching
- 🛠️ Memory optimization options for different hardware setups
- 🪄 Smart prompt refinement with Ollama integration

## 🚀 Installation

Getting started is super easy! Just follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/blndev/AI-ImageGeneration-Flux.git
cd AI-ImageGeneration-Flux
```

2. Create your configuration:
```bash
cp .env.example .env
```

3. Edit `.env` file and set your HuggingFace token:
```bash
HUGGINGFACE_TOKEN=your_token_here
```

> 📝 **Note**: You'll need to accept the license agreement for FLUX.1-schnell at [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## 🎮 Running the Application

The project includes a convenient `run.sh` script that handles:
- ✅ Environment setup
- ✅ Dependency management
- ✅ Version checking
- ✅ Application startup

Simply execute:
```bash
chmod +x run.sh
./run.sh
```

## ⚙️ Configuration

Customize your experience through environment variables in `.env`:

### 🎛️ General Settings
- `LOG_LEVEL`: Set logging detail (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `GRADIO_SHARED`: Enable public Gradio link
- `NO_AI`: Development mode without AI processing

### 🎫 Token System
- `INITIAL_GENERATION_TOKEN`: Starting tokens for new users
- `NEW_TOKEN_WAIT_TIME`: Minutes between token refreshes
- `ALLOW_UPLOAD`: Enable image sharing for token rewards

### 🪄 Prompt Magic
- `OLLAMA_SERVER`: Custom Ollama server location (default: localhost)
- `OLLAMA_MODEL`: Model for prompt enhancement (default: artifish/llama3.2-uncensored)

### 🖼️ Generation Settings
- `GENERATION_MODEL`: Choose your model (default: black-forest-labs/FLUX.1-dev)
- `GENERATION_STEPS`: Number of generation steps
- `GENERATION_GUIDANCE`: Guidance scale for generation
- `GENERATION_ASPECT_*`: Configure output dimensions

### 💾 Memory Optimization
- `GPU_ALLOW_XFORMERS`: Enable memory-efficient attention
- `GPU_ALLOW_ATTENTION_SLICING`: Split calculations for lower memory usage
- `GPU_ALLOW_MEMORY_OFFLOAD`: Use CPU memory for model handling

### 🎯 Output Configuration
- `MODEL_DIRECTORY`: Location for downloaded models
- `OUTPUT_DIRECTORY`: Where to save generated images

## 🤝 Contributing

We love contributions! Feel free to:
- Submit bug reports
- Propose new features
- Create pull requests

## 📜 License

This project is licensed under the terms included in the LICENSE file.

---

Happy Creating! 🎨✨
