import sys
from pathlib import Path

# Add the src directory to the system path
sys.path.append(str(Path(__file__).parent / "src"))

from sdprompt.main import cli

if __name__ == "__main__":
    cli() 