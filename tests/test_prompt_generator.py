import pytest
from pathlib import Path
from src.prompt_generator import PromptGenerator
import json

@pytest.fixture
def prompt_generator():
    return PromptGenerator(api_key="sk-test", model="claude-3-sonnet")

@pytest.mark.asyncio
async def test_prompt_analysis(prompt_generator):
    result = await prompt_generator.analyze_prompt(
        "A serene mountain landscape at sunset"
    )
    
    assert "status" in result
    assert "analysis" in result
    assert "generation" in result
    
    assert result["status"]["success"]
    assert isinstance(result["generation"]["prompt"], str)
    assert isinstance(result["generation"]["negative_prompt"], str)

@pytest.mark.asyncio
async def test_prompt_validation(prompt_generator):
    with pytest.raises(ValueError):
        await prompt_generator.analyze_prompt("")
        
    with pytest.raises(ValueError):
        await prompt_generator.analyze_prompt("a" * 1000)  # Too long

def test_response_parsing(prompt_generator):
    sample_response = """
ANALYSIS
- Subject: Mountain landscape
- Elements: Sunset, peaks, atmosphere
- Technical: Wide angle composition

PROMPT
A majestic mountain landscape at golden hour, captured in wide angle...

TECHNICAL SPECIFICATIONS
Style/Medium: Digital painting, photorealistic
Composition: Wide angle landscape
Lighting: Golden hour, atmospheric
Color Scheme: Warm earth tones

NEGATIVE PROMPT
blurry, oversaturated, poor composition...
"""
    
    result = prompt_generator._parse_response(sample_response)
    assert result["generation"]["prompt"].startswith("A majestic mountain")
    assert "mountain landscape" in result["analysis"]["subject"]["primary"] 

@pytest.mark.asyncio
async def test_prompt_error_handling(prompt_generator):
    # Test empty prompt
    with pytest.raises(PromptValidationError):
        await prompt_generator.analyze_prompt("")
    
    # Test oversized prompt
    with pytest.raises(PromptValidationError):
        await prompt_generator.analyze_prompt("x" * 1000)
    
    # Test API error handling
    prompt_generator.client = None  # Force API error
    with pytest.raises(PromptAnalysisError):
        await prompt_generator.analyze_prompt("Test prompt")

def test_response_section_parsing(prompt_generator):
    # Test missing sections
    with pytest.raises(PromptAnalysisError):
        prompt_generator._parse_response("Invalid response")
    
    # Test partial response
    result = prompt_generator._parse_response("""
ANALYSIS
- Subject: Test

PROMPT
Test prompt
""")
    assert result["status"]["success"] is False
    assert len(result["status"]["errors"]) > 0

def test_technical_parameter_extraction(prompt_generator):
    response = """
TECHNICAL SPECIFICATIONS
Style/Medium: Oil painting, impressionist
Composition: Rule of thirds
Lighting: Natural, diffused
Color Scheme: Warm, earth tones
Camera: Wide angle
Effects: Depth of field
"""
    
    result = prompt_generator._extract_technical(
        {k.strip(): v.strip() 
         for k, v in [line.split(":", 1) 
         for line in response.strip().split("\n")[1:]]}
    )
    
    assert "composition" in result
    assert "lighting" in result
    assert "color" in result
    assert result["composition"] == "Rule of thirds" 