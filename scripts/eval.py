# scripts/eval.py
# Author: MERO  telegram @QP4RM  mero@merobest.xyz
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stryk.cli import main

if __name__ == "__main__":
    args = list(sys.argv[1:])
    raise SystemExit(main(["eval", *args] if args and args[0] != "eval" else (args or ["eval"])))
