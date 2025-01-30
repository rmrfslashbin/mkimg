# Image Generation Tool Specification

<tool>
A tool for analyzing user prompts and generating detailed specifications for Stable Diffusion image generation.

## Properties

The tool must return a JSON object with the following schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["status", "analysis", "generation"],
  "properties": {
    "status": {
      "type": "object",
      "required": ["success", "errors", "warnings"],
      "properties": {
        "success": {
          "type": "boolean",
          "description": "Overall success status of the prompt analysis"
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["code", "message", "severity"],
            "properties": {
              "code": {
                "type": "string",
                "description": "Error code identifier"
              },
              "message": {
                "type": "string",
                "description": "Detailed error message"
              },
              "severity": {
                "type": "string",
                "enum": ["fatal", "error", "warning", "info"],
                "description": "Severity level of the error"
              },
              "details": {
                "type": "object",
                "description": "Additional error-specific details"
              }
            }
          }
        },
        "warnings": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {
              "code": {
                "type": "string",
                "description": "Warning code identifier"
              },
              "message": {
                "type": "string",
                "description": "Detailed warning message"
              },
              "suggestion": {
                "type": "string",
                "description": "Optional suggestion for addressing the warning"
              }
            }
          }
        }
      }
    },
    "analysis": {
      "type": "object",
      "required": ["subject", "style", "technical", "challenges"],
      "properties": {
        "subject": {
          "type": "object",
          "required": ["primary", "elements"],
          "properties": {
            "primary": {
              "type": "string",
              "description": "Main subject of the image"
            },
            "elements": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "Key elements identified in the prompt"
            }
          }
        },
        "style": {
          "type": "object",
          "required": ["primary", "influences"],
          "properties": {
            "primary": {
              "type": "string",
              "description": "Primary artistic style"
            },
            "influences": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "Additional style influences"
            }
          }
        },
        "technical": {
          "type": "object",
          "required": ["composition", "lighting", "color"],
          "properties": {
            "composition": {
              "type": "string",
              "description": "Compositional approach"
            },
            "lighting": {
              "type": "string",
              "description": "Lighting setup"
            },
            "color": {
              "type": "string",
              "description": "Color scheme"
            }
          }
        },
        "challenges": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Potential challenges identified"
        }
      }
    },
    "generation": {
      "type": "object",
      "required": ["prompt", "negative_prompt", "parameters"],
      "properties": {
        "prompt": {
          "type": "string",
          "description": "Generated prompt for Stable Diffusion"
        },
        "negative_prompt": {
          "type": "string",
          "description": "Generated negative prompt"
        },
        "parameters": {
          "type": "object",
          "required": ["width", "height", "cfg_scale", "steps"],
          "properties": {
            "width": {
              "type": "integer",
              "minimum": 512,
              "maximum": 1024,
              "description": "Image width"
            },
            "height": {
              "type": "integer",
              "minimum": 512,
              "maximum": 1024,
              "description": "Image height"
            },
            "cfg_scale": {
              "type": "number",
              "minimum": 1,
              "maximum": 20,
              "description": "Classifier free guidance scale"
            },
            "steps": {
              "type": "integer",
              "minimum": 10,
              "maximum": 150,
              "description": "Number of inference steps"
            },
            "seed": {
              "type": "integer",
              "description": "Optional seed for reproducibility"
            }
          }
        }
      }
    }
  }
}
```

## Error Codes

### Fatal Errors (F***)
- F001: Invalid prompt structure
- F002: Content policy violation
- F003: Unsupported image type
- F004: Invalid parameter values

### Errors (E***)
- E001: Missing required element
- E002: Ambiguous subject description
- E003: Conflicting style elements
- E004: Technical parameter out of range

### Warnings (W***)
- W001: Complex composition
- W002: Challenging lighting setup
- W003: Potentially difficult detail level
- W004: Style combination complexity

## Example Response

```json
{
  "status": {
    "success": true,
    "errors": [],
    "warnings": [
      {
        "code": "W002",
        "message": "Complex lighting setup may require additional guidance",
        "suggestion": "Consider simplifying the lighting requirements"
      }
    ]
  },
  "analysis": {
    "subject": {
      "primary": "mountain landscape",
      "elements": ["mountains", "sunset", "lake", "trees"]
    },
    "style": {
      "primary": "digital painting",
      "influences": ["impressionism", "fantasy art"]
    },
    "technical": {
      "composition": "wide angle landscape view",
      "lighting": "golden hour sunset lighting",
      "color": "warm earth tones with cool shadows"
    },
    "challenges": [
      "Complex lighting transitions",
      "Multiple depth layers"
    ]
  },
  "generation": {
    "prompt": "Majestic mountain landscape at sunset, digital painting style with impressionistic influences, wide-angle view capturing a serene lake reflecting golden hour light, surrounded by ancient trees, dramatic clouds catching last rays of sun, volumetric lighting through atmosphere, highly detailed",
    "negative_prompt": "blurry, oversaturated, flat lighting, poor composition, distorted perspective, unrealistic colors, noise, artificial looking, low quality, grainy",
    "parameters": {
      "width": 1024,
      "height": 768,
      "cfg_scale": 7.5,
      "steps": 50
    }
  }
}
```
</tool>