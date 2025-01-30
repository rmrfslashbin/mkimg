# Stable Diffusion Prompt Generator

A Python tool that uses Anthropic's Claude to generate optimized prompts for Stability AI's image generation models. The tool processes user prompts through Claude to create well-structured prompts for Stable Diffusion, generates images, and maintains metadata for reproducibility.

## Features

- Uses Claude to generate optimized Stable Diffusion prompts
- Supports Stable Diffusion 3.0 and 3.5 models
- Flexible configuration via YAML, .env files, or CLI parameters
- Comprehensive logging and error handling
- Metadata tracking for prompt history and image generation parameters
- Support for generating multiple images from a single prompt
- Multiple input methods: stdin or file
- Multiple output image formats: JPEG, PNG, WebP
- Environment-specific configuration overrides

## Requirements

- Python 3.x (system installed version)
- Poetry for dependency management
- Anthropic API key
- Stability AI API key

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install dependencies using Poetry
poetry install
```

## Configuration

The tool supports three configuration methods with the following precedence (highest to lowest):

1. CLI parameters
   - Always override other settings
   - Useful for one-off changes and testing

2. Environment Variables (.env file)
   - Override YAML config settings
   - Good for sensitive data like API keys
   - Can be loaded from .env file or set in shell

3. YAML Configuration File
   - Base configuration
   - Default values if not set elsewhere
   - Best for version-controlled settings

Example of precedence:
```bash
# config.yaml sets output_dir to "./output"
# .env sets OUTPUT_DIR to "./env_output"
# CLI uses --output-dir "./cli_output"
# Result: "./cli_output" will be used
```

When multiple configuration sources define the same setting, the highest precedence source is used and others are ignored.

### Configuration Options

```yaml
# Example config.yaml
anthropic:
  api_key: "your-anthropic-api-key"
  model: "claude-3-sonnet"  # default model
  
stability:
  api_key: "your-stability-api-key"
  model: "stable-diffusion-v3"  # default model
  
output:
  format: "png"  # default format
  directory: "./output"  # default output directory
  
logging:
  level: "INFO"
  file: "app.log"
```

### Environment Variables

```plaintext
ANTHROPIC_API_KEY=your-anthropic-api-key
STABILITY_API_KEY=your-stability-api-key
MODEL_ANTHROPIC=claude-3-sonnet
MODEL_STABILITY=stable-diffusion-v3
OUTPUT_FORMAT=png
OUTPUT_DIR=./output
LOG_LEVEL=INFO
LOG_FILE=app.log
```

## Usage

### Command Line Interface

```bash
usage: main.py [-h] [-c CONFIG] [-e ENV] [-i INPUT] [-o OUTPUT_DIR] 
               [-f {png,jpeg,webp}] [-n COUNT] [-m METADATA] 
               [--anthropic-model MODEL] [--stability-model MODEL] 
               [-v] [--debug]

Image Generation Pipeline using Claude and Stable Diffusion

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to YAML config file
  -e ENV, --env ENV     Path to .env file
  -i INPUT, --input INPUT
                        Path to input prompt file (if not using stdin)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory for generated images and metadata
  -f {png,jpeg,webp}, --format {png,jpeg,webp}
                        Output image format (default: png)
  -n COUNT, --count COUNT
                        Number of images to generate (default: 1)
  -m METADATA, --metadata METADATA
                        Path to metadata YAML to skip Claude
  --anthropic-model MODEL
                        Anthropic model to use (overrides config)
  --stability-model MODEL
                        Stability model to use (overrides config)
  -v, --verbose         Increase output verbosity
  --debug              Enable debug logging
  --dry-run           Validate configuration and prompts without API calls
  --timeout SECONDS   API timeout in seconds (default: 60)
  --parallel          Enable parallel processing for multiple images
  --retry ATTEMPTS    Number of retry attempts for failed API calls (default: 3)
  --continue-on-error Continue processing remaining images if one fails
  --seed SEED         Override seed value from metadata file
  --override-model    Override model specified in metadata file

Environment Variables:
  The following environment variables can be set in a .env file or shell:
    ANTHROPIC_API_KEY   Anthropic API key
    STABILITY_API_KEY   Stability API key
    MODEL_ANTHROPIC     Anthropic model name
    MODEL_STABILITY     Stability model name
    OUTPUT_FORMAT       Default image format
    OUTPUT_DIR         Default output directory
    LOG_LEVEL          Logging level
    LOG_FILE           Log file path

### Basic Usage

```bash
# Process prompt from stdin
echo "A serene mountain landscape at sunset" | python main.py

# Process prompt from file
python main.py --input prompt.txt

# Generate multiple images
python main.py --input prompt.txt --count 3

# Use specific configuration file
python main.py --config custom-config.yaml

# Override output directory
python main.py --output-dir ./custom-output

# Use existing metadata file (skip Claude)
python main.py --metadata existing-image-metadata.yaml

# Dry run to validate configuration
python main.py --input prompt.txt --dry-run

# Generate multiple images in parallel
python main.py --input prompt.txt --count 3 --parallel

# Reuse metadata with overrides
python main.py --metadata previous-run.yaml --seed 12345 --override-model "stable-diffusion-v3.5"
```

### Output Structure

```plaintext
output/
├── image_001.png
├── image_001.yaml
├── image_002.png
├── image_002.yaml
└── app.log
```

### Metadata Structure

```yaml
# Example image_001.yaml
timestamp: "2025-01-30T14:30:00Z"
original_prompt: "A serene mountain landscape at sunset"
generated_prompt: "Create a breathtaking mountain landscape..."
model_info:
  anthropic:
    model: "claude-3-sonnet"
    version: "1.0"
  stability:
    model: "stable-diffusion-v3"
    version: "3.0"
image_parameters:
  width: 1024
  height: 1024
  steps: 50
  cfg_scale: 7.0
  format: "png"
  seed: 123456
image_verification:
  checksum_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  size_bytes: 1048576
  verification_time: "2025-01-30T14:30:05Z"
status:
  success: true
  generation_time: 5.2
```

## System Prompt

The system prompt for Claude is stored in `prompts/system_prompt.md` and can be customized at runtime using mustache templates.

## Project Structure

```plaintext
.
├── pyproject.toml
├── poetry.lock
├── README.md
├── prompts/
│   └── system_prompt.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── prompt_generator.py
│   ├── image_generator.py
│   ├── metadata.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── cli.py
├── tests/
│   └── __init__.py
└── examples/
    ├── config.yaml
    └── prompt_examples.txt
```

## Safety and Validation

### Pre-flight Checks
The tool performs several validation steps before making any API calls:
- Configuration validation
  - Required parameters present
  - API keys format verification
  - Directory permissions
  - File write permissions
- Prompt sanitization
  - Content safety checks
  - Length validation
  - Format verification
- Resource availability
  - Output directory exists/creation
  - Sufficient disk space
  - File access permissions

### Security Features
- API key handling
  - Warning when keys provided via CLI
  - Secure storage recommendations
- Image verification
  - SHA-256 checksum generation and verification
  - Size validation
  - Format verification
- Input sanitization
  - Prompt content filtering
  - Metadata validation
  - File path sanitization

## Error Handling

The tool implements comprehensive error handling for:
- API authentication failures
- Rate limiting
- Network errors
- Invalid configurations
- File I/O errors
- Invalid prompt formats
- Image generation failures

All errors are logged with appropriate context and presented to the user with clear resolution steps.

### Retry Mechanism
- Automatic retry for transient failures
- Configurable retry attempts (default: 3)
- Exponential backoff
- Rate limit handling
- Timeout handling (default: 60 seconds)

### Partial Failure Handling
- Continue processing remaining images on failure
- Individual error reporting per image
- Detailed failure logging
- Recovery suggestions

## Logging

Uses Python's `logging` module with structured logging:
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Log rotation
- Separate log streams for application and API calls
- Contextual information including request IDs and timestamps

## Performance Features

### Parallel Processing
- Concurrent image generation for multiple images
- Configurable concurrency limits
- Progress tracking
- Resource usage monitoring

### Timeout Management
- Configurable API timeouts (default: 60 seconds)
- Individual timeouts for Claude and Stability
- Connection timeout handling
- Response timeout handling

## Development

### Setting up Development Environment

```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run linting
poetry run flake8
poetry run black .
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## Limitations

- No interactive mode
- No support for Stability AI models other than SD 3.0 and 3.5
- No support for Stability AI endpoints other than `generate`
- No containerization
- No custom system prompts for Claude
- No batch processing of multiple different prompts
- No caching or optimization of similar prompts

## Future Enhancements

- Support for additional Stability AI models and endpoints
- Batch processing of multiple prompts
- Interactive mode
- Custom system prompts for Claude
- Containerization
- Prompt effectiveness analysis
- Image post-processing options

## Specification Files
See also these files for more details:
- [tool_spec.md](tool_spec.md)
- [system_prompt.md](system_prompt.md)
