import pytest
from pathlib import Path
from src.config import ConfigBuilder, load_config

def test_config_builder_yaml():
    builder = ConfigBuilder()
    config_data = {
        "anthropic": {
            "model": "claude-3-sonnet",
            "api_key": "sk-test-key"
        },
        "stability": {
            "model": "stable-diffusion-v3",
            "api_key": "sk-test-key"
        },
        "output": {
            "format": "png",
            "directory": "./output"
        },
        "logging": {
            "level": "INFO"
        }
    }
    
    builder.config = config_data
    
    # Test YAML export
    tmp_path = Path("test_config.yaml")
    builder.export_yaml(tmp_path)
    
    # Load and verify
    loaded_config = load_config(yaml_path=tmp_path)
    assert loaded_config.anthropic.model == "claude-3-sonnet"
    assert loaded_config.stability.model == "stable-diffusion-v3"
    assert loaded_config.output.format == "png"
    
    # Cleanup
    tmp_path.unlink()

def test_config_validation():
    with pytest.raises(ValueError):
        load_config(cli_args={
            "anthropic_api_key": "invalid-key",
            "stability_api_key": "invalid-key"
        }) 