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

### 📝 Model Configuration (modelconfig.json)

The `modelconfig.json` file supports a powerful inheritance system that allows you to define model configurations hierarchically. This helps reduce redundancy and makes it easier to manage multiple model configurations that share common settings.

#### Inheritance Example
```json
{
    "Model": "default",
    "Description": "default configuration flux-schnell",
    "Parent": "flux1:schnell"
}
```

In this example:
1. The "default" configuration inherits from "flux1:schnell"
2. "flux1:schnell" inherits from "flux"
3. "flux" is a base configuration with no parent

This creates an inheritance chain where:
- Base settings are defined in the "flux" configuration
- "flux1:schnell" extends these settings with specific parameters for the FLUX.1-schnell model
- The "default" configuration can override or extend these settings further

Properties are inherited recursively, meaning a child configuration will receive all properties from its parent, grandparent, and so on. Child configurations can:
- Override inherited properties by redefining them
- Add new properties not present in the parent
- Inherit complex nested objects (like Aspect_Ratio, Generation settings, etc.)

This system makes it easy to:
- Maintain consistent base settings across multiple configurations
- Create specialized configurations with minimal duplication
- Quickly switch between different model setups while preserving common settings

#### Complete Configuration Example
```json
{
    "Model": "model-name",                    // Name used to reference this config
    "Path": "huggingface-repo/model-name",   // HuggingFace path or local file
    "Description": "Model description",       // Optional description
    "Parent": "parent-model",                // Optional parent config to inherit from
    "Generation": {
        "steps": 40,                         // Number of generation steps
        "guidance": 4.0                      // Guidance scale for generation
    },
    "Aspect_Ratio": {
        "Square": "1024x1024",              // Square image dimensions
        "Landscape": "1152x768",            // Landscape image dimensions
        "Portrait": "768x1152"              // Portrait image dimensions
    },
    "Embeddings": {
        "positive": [
            {
                "name": "embedding-name",    // Name of the embedding
                "source": "embedding.pt",    // Path to embedding file
                "keyword": "trigger-word"    // Word that triggers this embedding
            }
        ],
        "negative": []                      // Negative embeddings list
    },
    "Loras": [
        {
            "name": "lora-name",            // Name of the LoRA
            "src": "lora.safetensors",      // Path to LoRA file
            "trigger": "trigger-word",       // Word that activates this LoRA
            "weight": 1.0,                  // LoRA weight (0.0 to 1.0)
            "inject_when": [                // Keywords that trigger LoRA injection
                "keyword1",
                "keyword2"
            ]
        }
    ],
    "Examples": [                           // Optional example prompts
        [
            "A majestic mountain landscape at sunset",  // Positive prompt
            "ugly, blurry",                            // Negative prompt
            "▤ Landscape",                             // Aspect ratio (▤ Landscape, ▯ Portrait, □ Square)
            1                                          // Weight/priority
        ],
        [
            "Professional portrait in modern office",
            "bad anatomy, deformed",
            "▯ Portrait",
            1
        ]
    ]
}
```

This configuration shows all available settings that can be defined for a model. Settings can be:
- Inherited from a parent configuration
- Partially overridden in child configurations
- Extended with additional settings
- Left undefined to use defaults

## 🤝 Contributing

We love contributions! Feel free to:
- Submit bug reports
- Propose new features
- Create pull requests

## 📜 License

This project is licensed under the terms included in the LICENSE file.

---

Happy Creating! 🎨✨
