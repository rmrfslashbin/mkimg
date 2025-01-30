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
from datetime import datetime
import re
from PIL import Image
from rich.table import Table
from rich import box

from sdprompt.config import load_config, ConfigBuilder
from sdprompt.utils.logging import setup_logging
from sdprompt.prompt_generator import PromptGenerator
from sdprompt.image_generator import ImageGenerator, ImageParameters
from sdprompt.metadata import MetadataHandler
from sdprompt.utils.hash import compute_file_hash
from sdprompt.utils.image import ImageVerifier

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
@click.option('--max-age', type=int, help='Maximum age in seconds')
def verify(metadata_file: str, verbose: bool, max_age: int = None):
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
            
        # Get image path
        image_name = metadata_path.stem + '.png'
        image_path = metadata_path.parent / image_name
        
        if not image_path.exists():
            raise click.ClickException(f"Image file not found: {image_path}")
            
        if verbose:
            click.echo(f"Checking image: {image_path}")
            
        # Get expected values
        expected = metadata.get('image_info', {})
        generation_info = metadata.get('generation_info', {})
        
        # Verify dimensions
        if 'dimensions' in expected:
            width, height = map(int, expected['dimensions'].split('x'))
            if not ImageVerifier.verify_dimensions(image_path, width, height):
                raise click.ClickException("Image dimensions do not match metadata")
                
        # Verify format
        if 'format' in expected:
            if not ImageVerifier.verify_format(image_path, expected['format']):
                raise click.ClickException("Image format does not match metadata")
                
        # Verify timestamp
        if 'timestamp' in metadata:
            if not ImageVerifier.verify_timestamp(metadata['timestamp'], max_age):
                raise click.ClickException("Image timestamp verification failed")
                
        # Verify file size
        actual_size = image_path.stat().st_size
        if 'file_size_bytes' in expected and actual_size != expected['file_size_bytes']:
            if verbose:
                click.echo(f"Expected size: {expected['file_size_bytes']:,} bytes")
                click.echo(f"Actual size: {actual_size:,} bytes")
            raise click.ClickException(
                f"File size mismatch: expected {expected['file_size_bytes']:,}, got {actual_size:,}"
            )
            
        # Verify checksum
        actual_hash = compute_file_hash(image_path)
        if 'checksum_sha256' in expected and actual_hash != expected['checksum_sha256']:
            if verbose:
                click.echo(f"Expected hash: {expected['checksum_sha256']}")
                click.echo(f"Actual hash: {actual_hash}")
            raise click.ClickException(
                f"Checksum mismatch:\nExpected: {expected['checksum_sha256']}\nActual:   {actual_hash}"
            )
            
        if verbose:
            click.echo("\nVerification Results:")
            click.echo(f"Image path: {image_path}")
            click.echo(f"Format: {expected.get('format', 'unknown')}")
            click.echo(f"Dimensions: {expected.get('dimensions', 'unknown')}")
            click.echo(f"File size: {actual_size:,} bytes")
            click.echo(f"SHA256: {actual_hash}")
            click.echo(f"Generated by: {generation_info.get('engine', 'unknown')}")
            
        click.echo("✅ Image verification successful!")
        
    except Exception as e:
        raise click.ClickException(str(e))

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.option('-j', '--jobs', type=int, default=1, help='Number of parallel jobs')
@click.option('--max-age', type=int, help='Maximum age in seconds')
def verify_all(directory: str, verbose: bool, jobs: int, max_age: int = None):
    """Verify all images in a directory"""
    try:
        from rich.table import Table
        from concurrent.futures import ThreadPoolExecutor
        import concurrent.futures
        
        dir_path = Path(directory)
        yaml_files = list(dir_path.glob('*.yaml'))
        
        if not yaml_files:
            raise click.ClickException(f"No YAML files found in {directory}")
            
        results = []
        console.print(f"Verifying {len(yaml_files)} images...")
        
        def verify_file(yaml_file):
            try:
                verify(str(yaml_file), verbose=False, max_age=max_age)
                return (str(yaml_file.stem), True, None)
            except Exception as e:
                return (str(yaml_file.stem), False, str(e))
                
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_file = {executor.submit(verify_file, f): f for f in yaml_files}
            for future in concurrent.futures.as_completed(future_to_file):
                results.append(future.result())
                
        # Show summary
        success = sum(1 for _, success, _ in results if success)
        
        if verbose:
            table = Table(title="Verification Results")
            table.add_column("File")
            table.add_column("Status")
            table.add_column("Error", style="red")
            
            for file_name, success, error in results:
                status = "✅" if success else "❌"
                table.add_row(
                    file_name,
                    status,
                    error or ""
                )
                
            console.print(table)
            
        click.echo(f"\nVerification complete: {success}/{len(yaml_files)} successful")
        
        if success != len(yaml_files):
            raise click.ClickException("Some verifications failed")
            
    except Exception as e:
        raise click.ClickException(str(e))

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('-f', '--filter', 
    help='Filter expression examples:\n'
         '"size>5MB" - Images larger than 5MB\n'
         '"prompt contains mountain" - Search prompts\n'
         '"dimensions=1024x1024" - Specific size\n'
         '"model contains opus" - Filter by model'
)
@click.option('--sort', type=click.Choice(['date', 'size', 'model']), help='Sort results')
@click.option('--reverse', is_flag=True, help='Reverse sort order')
@click.option('-v', '--verbose', is_flag=True, help='Show detailed information')
def list(directory: str, filter: str, sort: str, reverse: bool, verbose: bool):
    """List and query generated images"""
    try:
        from rich.table import Table
        from rich.style import Style
        import re
        
        dir_path = Path(directory)
        yaml_files = [f for f in dir_path.glob('*.yaml')]
        
        if not yaml_files:
            raise click.ClickException(f"No YAML files found in {directory}")
            
        # Load all metadata
        entries = []
        console.print(f"Loading metadata for {len(yaml_files)} files...")
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    metadata = yaml.safe_load(f)
                    entries.append((str(yaml_file.stem), metadata))
            except Exception as e:
                console.print(f"[red]Error loading {yaml_file.name}: {str(e)}[/red]")
                
        if not entries:
            raise click.ClickException("No valid metadata files found")
            
        # Apply filters if specified
        if filter:
            filtered = []
            for file_name, metadata in entries:
                try:
                    if eval_filter(filter, metadata):
                        filtered.append((file_name, metadata))
                except Exception as e:
                    console.print(f"[yellow]Warning: Filter error for {file_name}: {e}[/yellow]")
            entries = filtered
            
        # Sort results
        if sort:
            entries.sort(
                key=lambda x: get_sort_key(x[1], sort),
                reverse=reverse
            )
            
        # Display results
        table = Table(
            title=f"Generated Images ({len(entries)} results)",
            show_lines=True,
            title_style="bold magenta",
            header_style="bold cyan",
            padding=(0, 1),
            collapse_padding=False,
            row_styles=["", "dim"],
            show_edge=True,
            expand=True,
            box=box.ROUNDED
        )
        
        # Define columns with adjusted widths
        cols = [
            ("Status", {"justify": "center", "width": 10, "no_wrap": True}),
            ("File", {"style": "bright_blue", "width": 12, "no_wrap": True}),
            ("Date", {"style": "green", "width": 20, "no_wrap": True}),
            ("Size", {"style": "yellow", "width": 10, "justify": "right", "no_wrap": True}),
            ("Dimensions", {"style": "cyan", "width": 12, "justify": "center", "no_wrap": True}),
            ("Models", {"style": "cyan", "width": 25, "no_wrap": True}),
        ]
        
        if verbose:
            cols.extend([
                ("Prompt", {"style": "bright_white", "width": 40, "overflow": "fold", "no_wrap": False}),
                ("Settings", {"style": "bright_black", "width": 25, "overflow": "fold", "no_wrap": False})
            ])
            
        # Add columns with consistent settings
        for name, settings in cols:
            table.add_column(name, **settings)
            
        # Format model display more compactly
        def format_model_info(anthropic: str, stability: str) -> str:
            parts = []
            if anthropic != 'Unknown':
                model_parts = anthropic.split('-')
                parts.append(f"[cyan]A:[/cyan] {model_parts[0]}")
                if len(model_parts) > 1:
                    parts.append(f"   {'-'.join(model_parts[1:])}")
            else:
                parts.append(f"[cyan]A:[/cyan] Unknown")
                
            if stability != 'Unknown':
                parts.append(f"[cyan]S:[/cyan] {stability}")
            else:
                parts.append(f"[cyan]S:[/cyan] Unknown")
                
            return "\n".join(parts)
        
        for file_name, metadata in entries:
            image_info = metadata.get('image_info', {})
            gen_info = metadata.get('generation_info', {})
            
            # Format status with fixed width
            status = get_image_status(file_name, dir_path, metadata)
            status = f"{status:10}"  # Ensure consistent width
            
            # Format filename with fixed width
            file_display = f"{file_name:12}"
            
            # Format timestamp with consistent width
            timestamp = metadata.get('timestamp')  # Get timestamp from metadata first
            if isinstance(timestamp, datetime):
                timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            elif not isinstance(timestamp, str):
                timestamp = 'Unknown'
            timestamp = f"{timestamp:20}"
            
            # Get dimensions from metadata or actual image
            dimensions = image_info.get('dimensions', '')
            if not dimensions or dimensions == 'Unknown':
                try:
                    image_path = dir_path / f"{file_name}.png"
                    if image_path.exists():
                        with Image.open(image_path) as img:
                            dimensions = f"{img.width}x{img.height}"
                except Exception:
                    dimensions = 'Unknown'
            
            # Get file size from both metadata and actual file
            size = image_info.get('file_size_bytes', 0)
            try:
                image_path = dir_path / f"{file_name}.png"
                if image_path.exists():
                    size = image_path.stat().st_size
            except Exception:
                pass
            
            # Get model information with versions
            anthropic_model = gen_info.get('model', 'Unknown')
            stability_model = gen_info.get('engine', 'Unknown')
            
            # Format model display
            model_display = format_model_info(
                gen_info.get('model', 'Unknown'),
                gen_info.get('engine', 'Unknown')
            )
            
            # Format prompt with better wrapping
            if verbose:
                prompt = str(metadata.get('original_prompt', 'N/A'))
                if len(prompt) > 37:  # Adjusted for new width
                    words = prompt.split()
                    lines = []
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= 37:
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            lines.append(" ".join(current_line))
                            current_line = [word]
                            current_length = len(word)
                            
                    if current_line:
                        lines.append(" ".join(current_line))
                    prompt = "\n".join(lines)
                
            row = [
                status,
                file_display,
                timestamp,
                format_size(size),
                format_dimensions(dimensions),
                model_display
            ]
            
            if verbose:
                settings_text = format_settings(gen_info.get('generation_settings', {}))
                row.extend([
                    prompt,
                    settings_text if settings_text else "No settings available"
                ])
                
            table.add_row(*row)
            
        console.print(table)
        
        # Show summary if filtering was applied
        if filter:
            console.print("\n[yellow]Filter applied: " + filter + "[/yellow]")
            console.print(f"[yellow]Showing {len(entries)} of {len(yaml_files)} images[/yellow]")
        
    except Exception as e:
        raise click.ClickException(str(e))

def parse_size(size_str: str) -> int:
    """Convert size string with units to bytes"""
    size_str = size_str.strip().lower()
    multipliers = {
        'b': 1,
        'kb': 1024,
        'mb': 1024 * 1024,
        'gb': 1024 * 1024 * 1024,
        'tb': 1024 * 1024 * 1024 * 1024
    }
    
    # Try to match number and unit
    import re
    match = re.match(r'^([\d.]+)\s*([kmgt]?b)?$', size_str)
    if not match:
        return int(size_str)  # Try plain number
        
    number, unit = match.groups()
    unit = unit or 'b'  # Default to bytes if no unit
    
    return int(float(number) * multipliers[unit])

def eval_filter(expr: str, metadata: dict) -> bool:
    """Evaluate a filter expression against metadata"""
    import re
    
    # Define comparison operators
    ops = {
        '>': lambda x, y: float(x) > float(y),
        '<': lambda x, y: float(x) < float(y),
        '>=': lambda x, y: float(x) >= float(y),
        '<=': lambda x, y: float(x) <= float(y),
        '=': lambda x, y: str(x).lower() == str(y).lower(),
        '!=': lambda x, y: str(x).lower() != str(y).lower(),
        'contains': lambda x, y: str(y).lower() in str(x).lower(),
    }
    
    # Parse expression
    match = re.match(r'(\w+)(?:\.(\w+))?\s*([><=!]+|contains)\s*(.+)', expr)
    if not match:
        raise ValueError(f"Invalid filter expression: {expr}")
        
    field, subfield, op, value = match.groups()
    value = value.strip('"\'')  # Remove quotes
    
    # Get field value from metadata
    try:
        if field == 'size':
            data = metadata.get('image_info', {}).get('file_size_bytes', 0)
            try:
                value = parse_size(value)  # Convert size string to bytes
            except ValueError:
                return False
        elif field == 'dimensions':
            data = metadata.get('image_info', {}).get('dimensions', '')
            if not data or data == 'Unknown':
                width = metadata.get('image_info', {}).get('width', 0)
                height = metadata.get('image_info', {}).get('height', 0)
                if width and height:
                    data = f"{width}x{height}"
                else:
                    return False
        elif field == 'prompt':
            data = metadata.get('original_prompt', '')
            if not data:
                return False
        else:
            data = metadata[field]
            if subfield:
                data = data.get(subfield)
            if data is None:
                return False
            
    except (KeyError, TypeError):
        return False
        
    try:
        return ops[op](str(data), str(value))
    except (ValueError, TypeError):
        return False

def get_sort_key(metadata: dict, sort_field: str):
    """Get sort key from metadata"""
    if sort_field == 'date':
        return metadata.get('timestamp', '')
    elif sort_field == 'size':
        return metadata.get('image_info', {}).get('file_size_bytes', 0)
    elif sort_field == 'model':
        return metadata.get('generation_info', {}).get('model', '')
    return ''

def format_size(size_bytes: int) -> str:
    """Format file size in human readable format with color"""
    try:
        size_bytes = int(size_bytes)  # Ensure integer
        if size_bytes == 0:
            return "[red]Unknown[/red]"
            
        # Color code based on size
        color = "green"
        if size_bytes > 10 * 1024 * 1024:  # > 10MB
            color = "red"
        elif size_bytes > 5 * 1024 * 1024:  # > 5MB
            color = "yellow"
            
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"[{color}]{size_bytes:,.1f} {unit}[/{color}]"
            size_bytes /= 1024
        return f"[red]{size_bytes:,.1f} TB[/red]"
    except (TypeError, ValueError):
        return "[red]Unknown[/red]"

def format_dimensions(dimensions: str) -> str:
    """Format dimensions with color based on common sizes"""
    if not dimensions or dimensions == 'Unknown':
        return "[red]Unknown[/red]"
        
    try:
        width, height = map(int, dimensions.split('x'))
        common_sizes = {
            (512, 512): "green",
            (768, 768): "green",
            (1024, 1024): "yellow",
            (1536, 1536): "red",
            (2048, 2048): "red"
        }
        color = common_sizes.get((width, height), "bright_black")
        return f"[{color}]{width}x{height}[/{color}]"
    except (ValueError, TypeError):
        return "[red]Unknown[/red]"

def format_settings(settings: dict) -> str:
    """Format generation settings in a readable way"""
    return "\n".join(f"{k}: {v}" for k, v in settings.items())

def get_image_status(file_name: str, dir_path: Path, metadata: dict) -> str:
    """Get image status with color coding"""
    image_path = dir_path / f"{file_name}.png"
    
    if not image_path.exists():
        return "[red]●[/red] Missing"
        
    try:
        # Check if image matches metadata
        actual_size = image_path.stat().st_size
        expected_size = metadata.get('image_info', {}).get('file_size_bytes', 0)
        
        if actual_size != expected_size:
            return "[yellow]●[/yellow] Modified"
            
        return "[green]●[/green] OK"
    except Exception:
        return "[red]●[/red] Error"

def main():
    cli()  # Remove the asyncio.run here since we handle it in the decorator

if __name__ == "__main__":
    main() 