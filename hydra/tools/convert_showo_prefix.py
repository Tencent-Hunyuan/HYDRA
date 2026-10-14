# coding=utf-8
# Copyright 2025 NUS Show Lab, HuggingFace.
# Modified from Show-o2: https://github.com/showlab/Show-o (Apache-2.0)
# SPDX-License-Identifier: Apache-2.0
"""Convert legacy HYDRA checkpoints to the current parameter naming.

Renames the `showo.*` parameter prefix (the LLM backbone attribute in the
original code) to `llm.*`, matching `HydraQwen2_5`'s current module layout.
Handles both single-file and sharded checkpoints (in-place rewrite).

Usage:
    python tools/convert_showo_prefix.py <checkpoint_dir_or_file>
    python tools/convert_showo_prefix.py ckpt/hydra_1.5b
    python tools/convert_showo_prefix.py ckpt/hydra_7b
"""

import json
import os
import sys

import torch


def convert_state_dict(state_dict):
    out = {}
    renamed = 0
    for k, v in state_dict.items():
        if k.startswith("showo."):
            out["llm." + k[len("showo."):]] = v
            renamed += 1
        else:
            out[k] = v
    return out, renamed


def convert_file(path):
    print(f"Loading {path} ...")
    sd = torch.load(path, map_location="cpu")
    sd, renamed = convert_state_dict(sd)
    print(f"  renamed {renamed}/{len(sd)} tensors: showo.* -> llm.*")
    torch.save(sd, path)
    print(f"  saved in-place: {path}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isfile(target):
        convert_file(target)
        return

    files = sorted(f for f in os.listdir(target)
                   if f.startswith("pytorch_model") and f.endswith(".bin"))
    if not files:
        print(f"No pytorch_model*.bin found in {target}")
        sys.exit(1)

    for f in files:
        convert_file(os.path.join(target, f))

    # sharded checkpoints keep a diffusers-style index; rewrite its keys too
    index_path = os.path.join(target, "diffusion_pytorch_model.bin.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        if "weight_map" in index:
            index["weight_map"] = {
                (("llm." + k[len("showo."):]) if k.startswith("showo.") else k): v
                for k, v in index["weight_map"].items()
            }
            with open(index_path, "w") as fh:
                json.dump(index, fh, indent=2)
            print(f"  rewrote index: {index_path}")


if __name__ == "__main__":
    main()
