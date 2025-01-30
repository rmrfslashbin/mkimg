from typing import TypeVar, Callable, Any, Optional
import asyncio
from functools import wraps
import logging
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

T = TypeVar("T")

def with_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Retry decorator for async functions"""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries:
                        break
                        
                    logging.warning(
                        f"Attempt {attempt + 1}/{retries} failed: {str(e)}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                    
            raise last_exception or Exception("All retry attempts failed")
            
        return wrapper
    return decorator

def create_progress() -> Progress:
    """Create a rich progress bar with custom formatting"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True
    ) 