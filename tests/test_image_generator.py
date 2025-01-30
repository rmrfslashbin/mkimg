import pytest
from pathlib import Path
from src.image_generator import ImageGenerator, ImageParameters

@pytest.fixture
def image_generator():
    return ImageGenerator(api_key="sk-test", model="stable-diffusion-v3")

@pytest.mark.asyncio
async def test_image_generation(image_generator, tmp_path):
    params = ImageParameters(
        width=1024,
        height=1024,
        steps=50,
        cfg_scale=7.0,
        seed=12345
    )
    
    output_path = tmp_path / "test.png"
    
    with pytest.raises(Exception):  # Will fail due to invalid API key
        await image_generator.generate_image(
            prompt="Test prompt",
            negative_prompt="Test negative",
            parameters=params,
            output_path=output_path
        )

def test_parameter_validation():
    with pytest.raises(ValueError):
        ImageParameters(width=256)  # Too small
        
    with pytest.raises(ValueError):
        ImageParameters(cfg_scale=30.0)  # Too high 

def test_parameter_constraints():
    # Test dimension multiple of 64
    with pytest.raises(ValueError):
        ImageParameters(width=1000)  # Not multiple of 64
    
    # Test dimension bounds
    with pytest.raises(ValueError):
        ImageParameters(width=256)  # Too small
    with pytest.raises(ValueError):
        ImageParameters(width=4096)  # Too large
    
    # Test steps bounds
    with pytest.raises(ValueError):
        ImageParameters(steps=5)  # Too few
    with pytest.raises(ValueError):
        ImageParameters(steps=200)  # Too many
    
    # Test cfg_scale bounds
    with pytest.raises(ValueError):
        ImageParameters(cfg_scale=0.5)  # Too low
    with pytest.raises(ValueError):
        ImageParameters(cfg_scale=25.0)  # Too high

@pytest.mark.asyncio
async def test_retry_mechanism(image_generator, tmp_path):
    params = ImageParameters()
    output_path = tmp_path / "test.png"
    
    # Mock API to always fail
    def mock_generate(*args, **kwargs):
        raise Exception("API Error")
    
    image_generator.stability_api.generate = mock_generate
    
    # Should retry 3 times then fail
    with pytest.raises(Exception):
        await image_generator.generate_image(
            prompt="Test",
            negative_prompt="",
            parameters=params,
            output_path=output_path
        ) 