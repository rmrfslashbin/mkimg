from typing import Dict, Any, Optional, List
import json
import re
from pathlib import Path
import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError
from sdprompt.utils.retry import with_retry, create_progress
import logging

class PromptError(Exception):
    """Base class for prompt-related errors"""
    pass

class PromptValidationError(PromptError):
    """Error for invalid prompt content"""
    pass

class PromptAnalysisError(PromptError):
    """Error during prompt analysis"""
    pass

class APIError(PromptError):
    """Error from the Anthropic API"""
    def __init__(self, message: str, status_code: int, request_id: Optional[str] = None):
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(f"{message} (Status: {status_code}, Request ID: {request_id})")

class ResponseParsingError(PromptError):
    """Error parsing Claude's response"""
    def __init__(self, message: str, missing_sections: List[str], response: Optional[str] = None):
        self.missing_sections = missing_sections
        self.response = response
        super().__init__(f"{message}: Missing sections: {', '.join(missing_sections)}")

class PromptAnalysis(BaseModel):
    subject: Dict[str, Any]
    style: Dict[str, Any]
    technical: Dict[str, Any]
    challenges: list[str]

class GenerationSpec(BaseModel):
    prompt: str
    negative_prompt: str
    parameters: Dict[str, Any]

class PromptGenerator:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)

    async def analyze_prompt(self, prompt: str) -> dict:
        """Analyze user prompt and generate optimized prompt for image generation"""
        try:
            # Create message with async client
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this image generation prompt and help optimize it for Stable Diffusion. 
                    Return a JSON object with these fields:
                    - generation.prompt: The optimized prompt
                    - generation.negative_prompt: Things to avoid (optional)
                    - generation.parameters: Dictionary of generation parameters
                    - analysis.style: Detected style
                    - analysis.subject: Main subject
                    - analysis.mood: Overall mood/tone
                    
                    User prompt: {prompt}"""
                }]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in response")
                
            # Parse response
            result = json.loads(json_match.group())
            
            # Validate and adjust parameters
            params = result.get("generation", {}).get("parameters", {})
            
            # Ensure valid seed (0 if not specified or invalid)
            if "seed" in params:
                try:
                    seed_val = int(params["seed"])
                    if seed_val < 0:
                        params["seed"] = 0
                except (ValueError, TypeError):
                    params["seed"] = 0
            
            # Clamp cfg_scale
            if "cfg_scale" in params:
                params["cfg_scale"] = min(max(float(params["cfg_scale"]), 1.0), 10.0)
            
            # Remove unsupported parameters
            supported_params = {"cfg_scale", "seed", "output_format", "model", "aspect_ratio"}
            params = {k: v for k, v in params.items() if k in supported_params}
            
            # Update result with adjusted parameters
            if "generation" in result:
                result["generation"]["parameters"] = params
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze prompt: {str(e)}")
    
    async def _get_system_prompt(self) -> str:
        return """Analyze image generation prompts and optimize them for Stability AI's API.

Key parameters and limits:
- prompt: required, 1-10000 characters, should be descriptive and specific
- negative_prompt: optional, max 10000 characters (not supported with turbo models)
- cfg_scale: 1.0-10.0 (default: 7.0)
- aspect_ratio (required):
  - 16:9 (landscape)
  - 1:1 (square)
  - 21:9 (wide)
  - 2:3 (portrait)
  - 3:2 (landscape)
  - 4:5 (portrait)
  - 5:4 (landscape)
  - 9:16 (portrait)
  - 9:21 (wide portrait)
- output_format: png or jpeg
- seed: optional, 0-4294967294
- models:
  - sd3.5-large (6.5 credits, best quality)
  - sd3.5-large-turbo (4 credits, faster, no negative prompts)
  - sd3.5-medium (3.5 credits, balanced)
  - sd3-large (6.5 credits)
  - sd3-large-turbo (4 credits, no negative prompts)
  - sd3-medium (3.5 credits)

Return a JSON object with:
{
    "generation": {
        "prompt": "optimized prompt with style tags",
        "negative_prompt": "things to avoid (omit for turbo models)",
        "parameters": {
            "cfg_scale": float (1.0-10.0),
            "aspect_ratio": string (from aspect ratio list),
            "output_format": string ("png" or "jpeg"),
            "model": string (from model list),
            "seed": int (optional, 0-4294967294)
        }
    },
    "analysis": {
        "style": "detected style",
        "subject": "main subject",
        "mood": "overall mood/tone"
    }
}

Guidelines:
- Keep prompts clear and focused
- Include artistic style and quality terms
- Consider composition and lighting
- Avoid problematic content
- Choose appropriate model based on needs:
  - Use turbo models for speed (but no negative prompts)
  - Use large models for best quality
  - Use medium models for balance
- Choose aspect ratio appropriate for the scene

Keep prompts focused and coherent."""
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's response into structured format"""
        required_sections = {"ANALYSIS", "PROMPT", "TECHNICAL SPECIFICATIONS", "NEGATIVE PROMPT"}
        sections = {
            "ANALYSIS": [],
            "PROMPT": "",
            "TECHNICAL SPECIFICATIONS": {},
            "NEGATIVE PROMPT": "",
        }
        
        # Parse sections
        current_section = None
        found_sections = set()
        
        for line in response.split("\n"):
            section = line.strip().upper()
            if section in required_sections:
                current_section = section
                found_sections.add(section)
                continue
                
            if current_section and line.strip():
                if current_section == "ANALYSIS":
                    if line.startswith("- "):
                        sections[current_section].append(line[2:].strip())
                elif current_section == "PROMPT":
                    sections[current_section] = line.strip()
                elif current_section == "TECHNICAL SPECIFICATIONS":
                    if ":" in line:
                        key, value = line.split(":", 1)
                        sections[current_section][key.strip()] = value.strip()
                elif current_section == "NEGATIVE PROMPT":
                    sections[current_section] = line.strip()
        
        # Validate all required sections are present
        missing_sections = required_sections - found_sections
        if missing_sections:
            raise ResponseParsingError(
                "Incomplete response from Claude",
                list(missing_sections),
                response
            )
        
        # Convert to tool spec format
        result = {
            "status": {
                "success": True,
                "errors": [],
                "warnings": []
            },
            "analysis": {
                "subject": self._extract_subject(sections["ANALYSIS"]),
                "style": self._extract_style(sections["TECHNICAL SPECIFICATIONS"]),
                "technical": self._extract_technical(sections["TECHNICAL SPECIFICATIONS"]),
                "challenges": self._extract_challenges(sections["ANALYSIS"])
            },
            "generation": {
                "prompt": sections["PROMPT"],
                "negative_prompt": sections["NEGATIVE PROMPT"],
                "parameters": self._extract_parameters(sections["TECHNICAL SPECIFICATIONS"])
            }
        }
        
        return result
    
    def _extract_subject(self, analysis: list[str]) -> Dict[str, Any]:
        """Extract subject information from analysis"""
        primary = next((item for item in analysis if "subject" in item.lower()), "")
        elements = [item for item in analysis if "element" in item.lower()]
        return {
            "primary": primary,
            "elements": elements
        }
    
    def _extract_style(self, tech_specs: Dict[str, str]) -> Dict[str, Any]:
        """Extract style information from technical specifications"""
        style = tech_specs.get("Style/Medium", "")
        influences = [s.strip() for s in style.split(",") if s.strip()]
        return {
            "primary": influences[0] if influences else "",
            "influences": influences[1:] if len(influences) > 1 else []
        }
    
    def _extract_technical(self, tech_specs: Dict[str, str]) -> Dict[str, Any]:
        """Extract technical information from specifications"""
        return {
            "composition": tech_specs.get("Composition", ""),
            "lighting": tech_specs.get("Lighting", ""),
            "color": tech_specs.get("Color Scheme", "")
        }
    
    def _extract_challenges(self, analysis: list[str]) -> list[str]:
        """Extract challenges from analysis"""
        return [item for item in analysis if "challenge" in item.lower()]
    
    def _extract_parameters(self, tech_specs: Dict[str, str]) -> Dict[str, Any]:
        """Extract generation parameters from specifications"""
        return {
            "width": 1024,  # Default values
            "height": 1024,
            "cfg_scale": 7.0,
            "steps": 50
        } 