import sys
from pathlib import Path
# make src/ importable as top-level modules (matches how scripts run)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
