#!/usr/bin/env python3
"""Source-checkout entry point; does not require reinstalling the package."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'src'))
from hexfellow_deploy.calibration import main
if __name__ == '__main__':
    raise SystemExit(main())
