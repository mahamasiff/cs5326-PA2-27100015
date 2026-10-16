"""Run from-scratch SFT.

  python scripts/train_sft.py --config configs/sft.yaml [--set key=value ...]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pa2.sft import train_sft  # noqa: E402
from pa2.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    print(train_sft(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
