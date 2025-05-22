# AI-ImageGeneration Tools

This directory contains utility tools for the AI-ImageGeneration project. These tools help with testing models, converting images, and generating analytics data for development purposes.

## Tools Overview

### 1. check_models.py

This tool automatically tests and evaluates different AI image generation models by generating sample images based on prompts defined in `prompts.txt`.

**Key Features:**
- Scans for models in the directory structure defined in `.env` (see `.env.example`)
- Generates test images for each model using prompts from `prompts.txt`
- Automatically detects model type (SD 1.5, SDXL, Flux) and adjusts parameters accordingly
- Supports different model configurations including Flux, Hyper, and Flux-schnell variants
- Configurable number of images per prompt

**Usage:**
```bash
# Set up your .env file first (copy from .env.example)
cp .env.example .env
# Edit .env to configure model directories and output settings
nano .env
# Run the tool
python check_models.py
```

**Configuration (via .env):**
- `MODEL_DIRECTORY`: Path to the directory containing model files
- `MODEL_FILTER`: Optional file containing allowed model names
- `CACHE_DIR`: Directory for HuggingFace cache
- `OUTPUT_DIRECTORY`: Where generated test images will be saved
- `IMAGES`: Number of images to generate per prompt
- `PROMPTS`: Path to the prompts file (default: prompts.txt)

### 2. convert_images.py

A utility for batch converting images to PNG format with optional resizing.

**Key Features:**
- Converts images to PNG format with high quality
- Preserves directory structure when processing folders
- Supports recursive processing of directories
- Optional image resizing while maintaining aspect ratio
- Handles various image formats (JPG, JPEG, PNG, GIF, BMP, WEBP, TIFF)

**Usage:**
```bash
# Basic usage (converts all images in a directory)
python convert_images.py /path/to/images

# Specify output directory
python convert_images.py /path/to/images --output-dir /path/to/output

# Resize images (maximum dimension will be 1024px)
python convert_images.py /path/to/images --max-size 1024
```

### 3. generate_dummy_analytics.py

Generates simulated analytics data for development and testing of the monitoring dashboard.

**Key Features:**
- Creates realistic usage patterns for the application
- Simulates user sessions, image generations, and token usage
- Publishes metrics to Prometheus for visualization in Grafana
- Configurable generation intervals and data variety

**Usage:**
```bash
# Run the analytics generator
python generate_dummy_analytics.py
```

**Generated Metrics:**
- Active user sessions
- New session creation events
- Image generation events with timing data
- User token/credit updates

## Configuration Files

### prompts.txt

Contains example prompts used by the `check_models.py` tool to generate test images. Each line represents one prompt.

### .env.example

Example environment configuration file. Copy this to `.env` and modify the values to configure the tools.

## Getting Started

1. Clone the repository
2. Copy `.env.example` to `.env` and configure as needed
3. Install required dependencies (from main project's requirements.txt)
4. Run the desired tool