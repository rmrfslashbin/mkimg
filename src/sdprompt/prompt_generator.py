from typing import Dict, Any, Optional, List
import json
import re
from pathlib import Path
import anthropic
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
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.Client(api_key=api_key)
        self.model = model
        self._system_prompt: Optional[str] = None
        
    @with_retry(retries=2, delay=1.0, exceptions=(anthropic.APIError, APIError))
    async def analyze_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """Analyze user prompt using Claude with progress tracking"""
        try:
            # Validate prompt
            if not user_prompt.strip():
                raise PromptValidationError("Prompt cannot be empty")
            if len(user_prompt) > 500:
                raise PromptValidationError("Prompt too long (max 500 characters)")
                
            with create_progress() as progress:
                # Load system prompt
                task = progress.add_task("Loading system prompt...", total=3)
                system_prompt = await self._get_system_prompt()
                progress.update(task, advance=1)
                
                # Prepare message
                progress.update(task, description="Analyzing prompt...")
                
                try:
                    # Create the message synchronously since the client handles async internally
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        system=system_prompt.replace("{{user_prompt}}", user_prompt),
                        messages=[
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        temperature=0.7,
                    )
                    
                    # Validate and extract the response text
                    if not message or not isinstance(message.content, list) or not message.content:
                        raise APIError("Invalid response format from Claude", 500)
                    
                    response_text = message.content[0].text
                    if not response_text:
                        raise APIError("Empty response from Claude", 500)
                    
                except anthropic.APIError as e:
                    request_id = getattr(e, 'request_id', None)
                    raise APIError(
                        f"API Error: {str(e)}",
                        status_code=getattr(e, 'status_code', 500),
                        request_id=request_id
                    )
                
                progress.update(task, advance=1)
                
                # Parse response
                progress.update(task, description="Processing response...")
                try:
                    result = self._parse_response(response_text)
                except ResponseParsingError as e:
                    logging.error(f"Failed to parse response: {str(e)}")
                    raise
                
                progress.update(task, advance=1)
                return result
                
        except PromptError:
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise PromptAnalysisError(f"Failed to analyze prompt: {str(e)}")
    
    async def _get_system_prompt(self) -> str:
        """Load and cache system prompt template"""
        if self._system_prompt is None:
            self._system_prompt = self._load_system_prompt()
        return self._system_prompt
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        prompt_path = Path(__file__).parent / "prompts" / "system_prompt.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found at {prompt_path}")
            
        with open(prompt_path) as f:
            return f.read()
        
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