"""
Lightweight UnSAMv2 training entrypoint that wraps the SAM 2 training pipeline.
It delegates to `sam2/training/train.py` while setting a sensible default config.
"""

import argparse
import sys
from pathlib import Path


def _add_repo_to_sys_path() -> None:
    """Ensure the inner `sam2` repo is importable when run from project root."""
    repo_root = Path(__file__).resolve().parent
    sam2_dir = repo_root / "sam2"
    if str(sam2_dir) not in sys.path:
        sys.path.insert(0, str(sam2_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UnSAMv2 training launcher")
    parser.add_argument(
        "-c",
        "--config",
        default="configs/unsamv2_small_training.yaml",
        help="Hydra config name under sam2/sam2/configs",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUs per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="number of nodes")
    return parser.parse_args()


def main() -> None:
    _add_repo_to_sys_path()
    from hydra import initialize_config_module

    from training.train import main as sam2_train_main
    from training.utils.train_utils import register_omegaconf_resolvers

    initialize_config_module("sam2", version_base="1.2")
    args = _parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()
    sam2_train_main(args)


if __name__ == "__main__":
    main()
