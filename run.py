#!/usr/bin/env python3
"""
Rabies Defender — AI-driven viral sequence understanding and countermeasure design.

Usage:
  python run.py              Train the model
  python run.py --eval       Evaluate trained model
  python run.py --screen     Screen drugs against a variant
"""
import os
import sys
import argparse

import torch

from config import DEVICE
from train import main as train_main
from evaluate import main as eval_main


def parse_args():
    parser = argparse.ArgumentParser(description="Rabies Defender — AI antiviral sequence model")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained model")
    parser.add_argument("--screen", action="store_true", help="Screen drugs against variants")
    parser.add_argument("--model-size", choices=["lite", "pro"], default="pro",
                        help="Model size (default: pro)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval:
        eval_main()
    elif args.screen:
        print("Drug screening mode — requires trained model + drug fingerprints")
        print("Coming in Phase 3")
    else:
        train_main(model_size=args.model_size)


if __name__ == "__main__":
    main()
