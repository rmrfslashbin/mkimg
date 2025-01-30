import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.traceback import install
import asyncio
import time
import sys
from functools import wraps
import yaml

from sdprompt.config import load_config, ConfigBuilder
from sdprompt.utils.logging import setup_logging
from sdprompt.prompt_generator import PromptGenerator
from sdprompt.image_generator import ImageGenerator, ImageParameters
from sdprompt.metadata import MetadataHandler
from sdprompt.utils.hash import compute_file_hash

# Install rich traceback handler
install()
console = Console()

@click.group()
@click.version_option()
def cli():
    """Stable Diffusion Prompt Generator CLI"""
    pass

def coro(f):
    """Decorator to handle coroutines in click commands"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@cli.command(name="generate")
@click.option(
    "-c", "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML config file"
)
@click.option(
    "-e", "--env",
    type=click.Path(path_type=Path),
    help="Path to .env file"
)
@click.option(
    "-i", "--input",
    type=click.Path(exists=True, path_type=Path),
    help="Path to input prompt file"
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    help="Directory for generated images"
)
@click.option(
    "-f", "--format",
    type=click.Choice(["png", "jpeg", "webp"]),
    default="png",
    help="Output image format"
)
@click.option(
    "-n", "--count",
    type=int,
    default=1,
    help="Number of images to generate"
)
@click.option(
    "--anthropic-model",
    help="Anthropic model to use"
)
@click.option(
    "--stability-model",
    help="Stability model to use"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Analyze prompt without generating image"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Increase output verbosity"
)
@coro
async def generate(
    config: Optional[Path],
    env: Optional[Path],
    input: Optional[Path],
    output_dir: Optional[Path],
    format: str,
    count: int,
    anthropic_model: Optional[str],
    stability_model: Optional[str],
    dry_run: bool,
    verbose: bool,
):
    """Generate images from prompts"""
    try:
        # Load configuration
        cli_args = {
            "output_format": format,
            "output_dir": output_dir,
            "anthropic_model": anthropic_model,
            "stability_model": stability_model,
        }
        
        app_config = load_config(
            yaml_path=config,
            env_path=env,
            cli_args={k: v for k, v in cli_args.items() if v is not None}
        )
        
        # Set up logging
        setup_logging(
            level=app_config.logging.level,
            log_file=app_config.logging.file,
            verbose=verbose
        )
        
        # Initialize components
        prompt_generator = PromptGenerator(
            api_key=app_config.anthropic.api_key,
            model=app_config.anthropic.model
        )
        
        image_generator = ImageGenerator(
            api_key=app_config.stability.api_key,
            model=app_config.stability.model
        )
        
        metadata_handler = MetadataHandler(
            output_dir=Path(app_config.output.directory)
        )
        
        # Get user prompt
        if input:
            with open(input) as f:
                user_prompt = f.read().strip()
        else:
            user_prompt = sys.stdin.read().strip()
            
        if not user_prompt:
            raise click.UsageError("No prompt provided")
            
        console.print("[yellow]Analyzing prompt...[/yellow]")
        
        # Analyze prompt
        prompt_data = await prompt_generator.analyze_prompt(user_prompt)
        
        console.print("[green]Prompt analysis complete[/green]")
        console.print("\nGenerated prompt:")
        console.print(f"[blue]{prompt_data['generation']['prompt']}[/blue]")
        
        if prompt_data['generation']['negative_prompt']:
            console.print("\nNegative prompt:")
            console.print(f"[red]{prompt_data['generation']['negative_prompt']}[/red]")
            
        console.print("\nParameters:")
        for key, value in prompt_data['generation']['parameters'].items():
            console.print(f"  {key}: {value}")
            
        if dry_run:
            console.print("\n[yellow]Dry run - skipping image generation[/yellow]")
            return
            
        # Generate images
        for i in range(count):
            image_number = str(i + 1).zfill(3)
            output_path = Path(app_config.output.directory) / f"image_{image_number}.{format}"
            
            console.print(f"\n[yellow]Generating image {i+1} of {count}...[/yellow]")
            
            start_time = time.time()
            
            # Generate image
            generation_result = await image_generator.generate_image(
                prompt=prompt_data["generation"]["prompt"],
                negative_prompt=prompt_data["generation"]["negative_prompt"],
                parameters=ImageParameters(**prompt_data["generation"]["parameters"]),
                output_path=output_path
            )
            
            generation_time = time.time() - start_time
            generation_result["generation_time"] = generation_time
            
            # Save metadata
            metadata_path = metadata_handler.save_metadata(
                image_path=output_path,
                prompt_data=prompt_data,
                generation_result=generation_result,
                original_prompt=user_prompt
            )
            
            console.print(f"[green]Image generated: {output_path}[/green]")
            console.print(f"[green]Metadata saved: {metadata_path}[/green]")
            
        console.print("\n[green]All images generated successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.Abort()

@cli.command()
@click.option(
    "--format",
    type=click.Choice(["env", "yaml"]),
    required=True,
    help="Output format for configuration"
)
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for configuration file"
)
@click.option(
    "--anthropic-api-key",
    help="Anthropic API key (recommended to set via env var SDPROMPT_ANTHROPIC_API_KEY)"
)
@click.option(
    "--stability-api-key",
    help="Stability API key (recommended to set via env var SDPROMPT_STABILITY_API_KEY)"
)
@click.option(
    "--anthropic-model",
    type=click.Choice([
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240229"
    ]),
    default="claude-3-opus-20240229",
    help="Anthropic model name"
)
@click.option(
    "--stability-model",
    type=click.Choice(["stable-diffusion-v3", "stable-diffusion-xl-1024-v1-0"]),
    default="stable-diffusion-v3",
    help="Stability model name"
)
@click.option(
    "--output-format",
    type=click.Choice(["png", "jpeg", "webp"]),
    default="png",
    help="Default output format"
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./output",
    help="Default output directory"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Logging level"
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    help="Log file path"
)
def config(
    format: str,
    output: Path,
    anthropic_api_key: Optional[str],
    stability_api_key: Optional[str],
    anthropic_model: str,
    stability_model: str,
    output_format: str,
    output_dir: Path,
    log_level: str,
    log_file: Optional[Path],
):
    """Generate configuration file from CLI arguments"""
    try:
        # Set up logging
        setup_logging(level=log_level, log_file=log_file)
        
        # Create configuration structure
        config_data = {
            "anthropic": {
                "model": anthropic_model,
            },
            "stability": {
                "model": stability_model,
            },
            "output": {
                "format": output_format,
                "directory": str(output_dir),
            },
            "logging": {
                "level": log_level,
            }
        }
        
        # Add optional values
        if anthropic_api_key:
            config_data["anthropic"]["api_key"] = anthropic_api_key
        if stability_api_key:
            config_data["stability"]["api_key"] = stability_api_key
        if log_file:
            config_data["logging"]["file"] = str(log_file)
            
        # Create builder and export
        builder = ConfigBuilder()
        builder.config = config_data
        
        # Create output directory if needed
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "env":
            builder.export_env(output)
            console.print(f"[green]Environment configuration exported to {output}[/green]")
        else:
            builder.export_yaml(output)
            console.print(f"[green]YAML configuration exported to {output}[/green]")
            
        # Print next steps if API keys weren't provided
        if not (anthropic_api_key and stability_api_key):
            console.print("\n[yellow]Note: API keys were not provided. You should set them:")
            if format == "env":
                console.print("\nAdd these lines to your .env file:")
                if not anthropic_api_key:
                    console.print("SDPROMPT_ANTHROPIC_API_KEY=your_api_key_here")
                if not stability_api_key:
                    console.print("SDPROMPT_STABILITY_API_KEY=your_api_key_here")
            else:
                console.print("\nUpdate these fields in your YAML file:")
                if not anthropic_api_key:
                    console.print("anthropic:\n  api_key: your_api_key_here")
                if not stability_api_key:
                    console.print("stability:\n  api_key: your_api_key_here")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.Abort()

@cli.command()
@click.argument('metadata_file', type=click.Path(exists=True))
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def verify(metadata_file: str, verbose: bool):
    """Verify an image against its metadata"""
    try:
        metadata_path = Path(metadata_file)
        if not metadata_path.suffix == '.yaml':
            raise click.BadParameter("Metadata file must be a YAML file")
            
        # Load metadata
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
            
        if verbose:
            click.echo("Metadata loaded successfully")
            
        # Get image path - construct from metadata filename
        image_name = metadata_path.stem + '.png'  # Assuming PNG for now
        image_path = metadata_path.parent / image_name
        
        if not image_path.exists():
            raise click.ClickException(f"Image file not found: {image_path}")
            
        if verbose:
            click.echo(f"Checking image: {image_path}")
            
        # Verify file size
        actual_size = image_path.stat().st_size
        expected_size = metadata.get('image_info', {}).get('file_size_bytes')
        if expected_size and actual_size != expected_size:
            if verbose:
                click.echo(f"Expected size: {expected_size:,} bytes")
                click.echo(f"Actual size: {actual_size:,} bytes")
            raise click.ClickException(
                f"File size mismatch: expected {expected_size:,}, got {actual_size:,}"
            )
            
        if verbose:
            click.echo("File size verified successfully")
            
        # Verify checksum
        actual_hash = compute_file_hash(image_path)
        expected_hash = metadata.get('image_info', {}).get('checksum_sha256')
        if expected_hash and actual_hash != expected_hash:
            if verbose:
                click.echo(f"Expected hash: {expected_hash}")
                click.echo(f"Actual hash: {actual_hash}")
            raise click.ClickException(
                f"Checksum mismatch:\nExpected: {expected_hash}\nActual:   {actual_hash}"
            )
            
        if verbose:
            click.echo("\nVerification Results:")
            click.echo(f"Image path: {image_path}")
            click.echo(f"File size: {actual_size:,} bytes")
            click.echo(f"SHA256: {actual_hash}")
            
        click.echo("✅ Image verification successful!")
        
    except Exception as e:
        raise click.ClickException(str(e))

def main():
    cli()  # Remove the asyncio.run here since we handle it in the decorator

if __name__ == "__main__":
    main() 