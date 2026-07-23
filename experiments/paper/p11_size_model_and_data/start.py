"""size_model_and_data — Model and Data Scaling Experiment.
Entry point for run.sh / scripts/run.py.

Usage:
    ./run.sh paper/size_model_and_data --env .env.serverML

Or directly:
    python experiments/paper/size_model_and_data/start.py --block 2

Blocks:
    1 — Features and Graph (TODO)
    2 — Model and Data Scaling
    3 — Architecture Random Search (TODO)
    4 — Learning Rate × Batch Size (TODO)
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="size_model_and_data — Model and Data Scaling Experiment")
    parser.add_argument("--block", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Block to run (1-4, default: 2)")
    args = parser.parse_args()

    if args.block == 2:
        from block2_data_model_scale import run_block2
        run_block2()
    elif args.block == 1:
        print("Block 1 (Features & Graph) — TODO", file=sys.stderr)
        sys.exit(1)
    elif args.block == 3:
        print("Block 3 (Random Architecture Search) — TODO", file=sys.stderr)
        sys.exit(1)
    elif args.block == 4:
        print("Block 4 (LR × Batch Size) — TODO", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
