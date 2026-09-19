"""Launch the shared ImageNet plain ViT³-derived training workflow."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .imagenet import main
else:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.imagenet import main


if __name__ == "__main__":
    main()
