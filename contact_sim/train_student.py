"""Train the DAgger student on the aggregated shards (GPU 1 by default; Isaac uses GPU 0).

    .venv/bin/python scripts/ur_contact/train_student.py --shards outputs/ur_nut/dagger/shard_it*.npz \
        --output outputs/ur_nut/dagger/student_it1.pt [--no-wrench] [--no-vision]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import student as st  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-wrench", action="store_true")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cpu")
    a = ap.parse_args()
    obs, act, valid = st.load_shards(a.shards)
    model, info = st.train(obs, act, valid, epochs=a.epochs, device=a.device, mask_wrench=a.no_wrench, mask_vision=a.no_vision)
    info.update(shards=[str(s) for s in a.shards], mask_wrench=a.no_wrench, mask_vision=a.no_vision)
    st.save(model.cpu(), a.output, info)
    h = info["history"]
    print(f"{a.output}: {info['samples']} samples, val mse {h[-1]['val']:.4f} (first epoch {h[0]['val']:.4f}), "
          f"masks wrench={a.no_wrench} vision={a.no_vision}")
    _ = np


if __name__ == "__main__":
    main()
