"""Run from-scratch DPO (same code for HH-RLHF and UltraFeedback).

  python scripts/train_dpo.py --config configs/dpo_hh.yaml [--set sft_checkpoint=/path/to/sft/best ...]
  python scripts/train_dpo.py --config configs/dpo_uf.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.dpo import train_dpo  # noqa: E402
from pa2.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    print(train_dpo(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
