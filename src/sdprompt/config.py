from pathlib import Path
from typing import Optional, Dict, Any, Union, Literal
import os
import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import re
from enum import Enum

class StabilityModel(str, Enum):
    """Valid models for Stability AI API"""
    SD3_LARGE = "sd3-large"
    SD3_LARGE_TURBO = "sd3-large-turbo"
    SD3_MEDIUM = "sd3-medium"
    SD35_LARGE = "sd3.5-large"
    SD35_LARGE_TURBO = "sd3.5-large-turbo"
    SD35_MEDIUM = "sd3.5-medium"

class AnthropicModel(str, Enum):
    """Valid models for Anthropic API"""
    OPUS = "claude-3-opus-20240229"
    SONNET = "claude-3-sonnet-20240229"
    HAIKU = "claude-3-haiku-20240229"
    OPUS_LATEST = "claude-3-opus-latest"
    SONNET_LATEST = "claude-3-5-sonnet-latest"
    HAIKU_LATEST = "claude-3-5-haiku-latest"

class BFLModel(str, Enum):
    """Valid models for Black Forest Labs API"""
    FLUX_PRO_11_ULTRA = "flux-pro-1.1-ultra"
    FLUX_PRO_11 = "flux-pro-1.1"
    FLUX_PRO = "flux-pro"
    FLUX_DEV = "flux-dev"

class AnthropicConfig(BaseModel):
    api_key: str = Field(..., pattern=r"^sk-ant-")
    model: AnthropicModel = Field(default=AnthropicModel.SONNET_LATEST)

    @field_validator("api_key")
    def validate_api_key(cls, v: str) -> str:
        if not v.startswith("sk-"):
            raise ValueError("Anthropic API key must start with 'sk-'")
        return v

class StabilityConfig(BaseModel):
    api_key: str = Field(..., pattern=r"^sk-")
    model: StabilityModel = Field(default=StabilityModel.SD35_LARGE)

    @field_validator("api_key")
    def validate_api_key(cls, v: str) -> str:
        if not re.match(r"^sk-[a-zA-Z0-9]{48}$", v):
            raise ValueError("Invalid Stability API key format")
        return v

class BFLConfig(BaseModel):
    api_key: str
    model: BFLModel = Field(default=BFLModel.FLUX_PRO_11)
    base_url: str = Field(default="https://api.us1.bfl.ai/v1")

    @field_validator("api_key")
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("BFL API key cannot be empty")
        return v

class OutputConfig(BaseModel):
    format: str = Field(default="png")
    directory: str = Field(default="./output")

    @field_validator("directory")
    def validate_directory(cls, v: str) -> str:
        # Convert string to Path for directory operations
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise ValueError(f"Directory {path} is not writable")
        return v  # Return original string value

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    file: Optional[str] = None

    @field_validator("file")
    def validate_log_file(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            # Convert string to Path for directory operations
            path = Path(v)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not os.access(path.parent, os.W_OK):
                raise ValueError(f"Log file directory {path.parent} is not writable")
        return v  # Return original string value

class AppConfig(BaseModel):
    anthropic: AnthropicConfig
    stability: StabilityConfig
    bfl: BFLConfig
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()

class ConfigBuilder:
    """Builds and manages configuration from multiple sources"""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.env_prefix = "SDPROMPT_"

    def load_yaml(self, path: Union[str, Path]) -> None:
        """Load configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path) as f:
            yaml_config = yaml.safe_load(f)
            self.config.update(yaml_config)

    def load_env(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from environment variables"""
        if path:
            load_dotenv(path)

        # Map environment variables to config structure
        env_mapping = {
            "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
            "ANTHROPIC_MODEL": ("anthropic", "model"),
            "STABILITY_API_KEY": ("stability", "api_key"),
            "STABILITY_MODEL": ("stability", "model"),
            "BFL_API_KEY": ("bfl", "api_key"),        # Add BFL mappings
            "BFL_MODEL": ("bfl", "model"),
            "BFL_BASE_URL": ("bfl", "base_url"),
            "OUTPUT_FORMAT": ("output", "format"),
            "OUTPUT_DIR": ("output", "directory"),
            "LOG_LEVEL": ("logging", "level"),
            "LOG_FILE": ("logging", "file"),
        }

        for env_var, config_path in env_mapping.items():
            env_key = f"{self.env_prefix}{env_var}"
            if env_key in os.environ:
                self._set_nested_dict(self.config, config_path, os.environ[env_key])

    def update_from_cli(self, cli_args: Dict[str, Any]) -> None:
        """Update configuration with CLI arguments"""
        # Map CLI arguments to config structure
        cli_mapping = {
            "anthropic_api_key": ("anthropic", "api_key"),
            "anthropic_model": ("anthropic", "model"),
            "stability_api_key": ("stability", "api_key"),
            "stability_model": ("stability", "model"),
            "bfl_api_key": ("bfl", "api_key"),        # Add BFL mappings
            "bfl_model": ("bfl", "model"),
            "bfl_base_url": ("bfl", "base_url"),
            "output_format": ("output", "format"),
            "output_dir": ("output", "directory"),
            "log_level": ("logging", "level"),
            "log_file": ("logging", "file"),
        }

        for cli_arg, config_path in cli_mapping.items():
            if cli_arg in cli_args and cli_args[cli_arg] is not None:
                self._set_nested_dict(self.config, config_path, cli_args[cli_arg])

    def _set_nested_dict(self, d: dict, keys: tuple, value: Any) -> None:
        """Set a value in a nested dictionary using a tuple of keys"""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def export_env(self, path: Union[str, Path]) -> None:
        """Export current configuration to .env file"""
        path = Path(path)
        
        env_vars = []
        for section, values in self.config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    env_name = f"{self.env_prefix}{section.upper()}_{key.upper()}"
                    env_vars.append(f"{env_name}={value}")

        with open(path, "w") as f:
            f.write("\n".join(env_vars))

    def export_yaml(self, path: Union[str, Path]) -> None:
        """Export current configuration to YAML file"""
        path = Path(path)
        
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def build(self) -> AppConfig:
        """Build and validate the final configuration"""
        return AppConfig(**self.config)

def load_config(
    yaml_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
    cli_args: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """
    Load configuration from all sources in order of precedence:
    CLI args > Environment variables > YAML config
    """
    builder = ConfigBuilder()
    
    if yaml_path:
        builder.load_yaml(yaml_path)
    
    builder.load_env(env_path)
    
    if cli_args:
        builder.update_from_cli(cli_args)
    
    return builder.build() 