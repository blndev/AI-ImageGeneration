# Version 1.2

## New Features
- Link sharing for Credits
- Upload of training material for Credits
- NSFW detection for generated output and blocking / censorship if active
- Model Configuration with inheritance support

## Internal
- complete refactoring of the User Interface and generation classes

## Bugfixes
- enhanced error handling in all classes
- fix generation issues with wrong parameters
- fix upload of generated images possible 


# Version 1.1

## New Features
- Model Configuration with inheritance support
- SDXL and SD1.5 support
- Prompt Magic
- NSFW check for prompt and generation


# 🚀 Release Notes - Version 1.0

## 🌟 Initial Release of AI Image Generation with FLUX

We're thrilled to announce the first stable release of our AI Image Generation platform! This release brings together powerful features and user-friendly interfaces to make AI image generation accessible to everyone.

### ✨ Key Features

#### 🎨 Core Generation Capabilities
- Integration of FLUX.1 models from black-forest-labs
- Support for multiple aspect ratios (1024x1024, 1152x768, 768x1152)
- Configurable generation parameters for fine-tuned outputs
- Intuitive text-to-image generation interface

#### 🛠️ Technical Infrastructure
- Robust auto-update system with version tracking
- User-friendly Gradio web interface

#### 📊 User Access Management
- Credit-based usage system
- Configurable Credit refresh intervals
- Optional image sharing capabilities
- User session management

#### 💾 Resource Management
- Automatic model downloading and caching
- Configurable output directory for generated images
- Smart memory optimization options:
  - xformers support
  - Attention slicing
  - Memory offloading capabilities

#### 🔧 Configuration System
- Environment-based configuration
- Flexible logging levels

### 🏗️ Technical Requirements
- Python 3.x (tested with 3.12)
- Git for version management
- HuggingFace account with accepted model license
- CUDA-compatible GPU (recommended)

### 📝 Documentation
- Comprehensive README with setup instructions
- Detailed environment variable documentation
- Clear running instructions

### 🔒 Security Features
- Environment-based secret management
- Configurable access controls
- Credit-based usage limitations

---

Thank you for choosing our AI Image Generation platform! We're excited to see the amazing creations you'll make with these tools! 🎨✨

For any issues or suggestions, please feel free to submit them through our issue tracker.
