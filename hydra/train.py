# coding=utf-8
# Copyright 2026 HYDRA Authors.
# Modified from Show-o2: https://github.com/showlab/Show-o (Apache-2.0)
# SPDX-License-Identifier: Apache-2.0
"""Unified training entry for HYDRA.

All training stages of HYDRA are launched through this single script. The
trainer is selected automatically from the stage number in the config file
name (e.g. ``configs/hydra_1.5b_stage2.yaml`` -> stage 2), or explicitly with
``stage=N`` / ``trainer=<name>`` on the command line.

Stage layout (matches the paper):

=====  ============================  =======================================
stage  what happens
=====  ============================  =======================================
1      Stage I: image-text alignment, flow-head training
       (LLM frozen, 150K steps, data ratio 0:1:3)
2      Stage II: joint training, all parameters
       (100K steps, data ratio 0:2:2)
3      Stage III: quality refinement, instruction tuning
       (20K steps, data ratio 1:2:2)
=====  ============================  =======================================

Examples:

    # launch stage 1 for the 1.5B model
    accelerate launch --config_file configs/zero2.yaml train.py config=configs/hydra_1.5b_stage1.yaml

    # launch stage 3, overriding a training parameter
    accelerate launch --config_file configs/zero2.yaml train.py config=configs/hydra_1.5b_stage3.yaml training.lr=1e-5
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# The paper's three-stage progressive pipeline runs through the single
# unified trainer; stage differences are carried by the YAML configs.
STAGE_TRAINERS = {
    1: "train_stage",
    2: "train_stage",
    3: "train_stage",
}

USAGE = __doc__


def parse_cli(argv):
    """Extract config path / stage / trainer from key=value CLI args."""
    config_path = None
    stage = None
    trainer = None
    kept = []
    for arg in argv:
        if arg.startswith("config="):
            config_path = arg.split("=", 1)[1]
        elif arg.startswith("stage="):
            stage = arg.split("=", 1)[1]
        elif arg.startswith("trainer="):
            trainer = arg.split("=", 1)[1]
        else:
            kept.append(arg)
    return config_path, stage, trainer, kept


def infer_stage(config_path):
    """Infer the training stage from the config file name (e.g. *_stage2*.yaml)."""
    m = re.search(r"stage[_-]?(\d)", Path(config_path).name)
    return int(m.group(1)) if m else None


def main():
    config_path, stage, trainer, kept_args = parse_cli(sys.argv[1:])

    if config_path is None:
        print(USAGE)
        sys.exit(1)

    if trainer is None:
        if stage is None:
            stage = infer_stage(config_path)
        if stage is not None:
            trainer = STAGE_TRAINERS.get(int(stage))
        if trainer is None:
            trainer = "train_stage"

    if trainer in STAGE_TRAINERS.values():
        module_name = trainer
    else:
        print(f"Unknown trainer: {trainer}\n{USAGE}")
        sys.exit(1)

    print(f"[train.py] config: {config_path} | trainer: {module_name}")

    import importlib
    module = importlib.import_module(module_name)

    # Delegate: the trainer's main() re-parses the CLI (config=... plus any
    # OmegaConf-style overrides). stage=/trainer= are consumed here.
    sys.argv = [sys.argv[0], f"config={config_path}"] + kept_args
    module.main()


if __name__ == "__main__":
    main()
