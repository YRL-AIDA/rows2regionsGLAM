"""p10 — Cluster Analysis: entry point.

Launches clustering analysis for all 5 tokenizer configurations
using raw (pre-GNN) feature vectors from the test split.
"""

import logging
import os
import sys

# Ensure rows2regionsGLAM is on path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiments.core.bootstrap import init

# Initialize environment (load .env, set up paths, suppress warnings)
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from cluster_analysis import run_all_tokenizers

if __name__ == "__main__":
    run_all_tokenizers()
