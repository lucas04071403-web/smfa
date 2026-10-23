"""Run the released trainer with checkpointing on the available single GPU."""

import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1] / "S-MLLMUn"
sys.path.insert(0, str(REPO))
from unlearning_methods import SMFA_train as official

prepare_model = official.prepare_model_for_kbit_training


def prepare_with_checkpointing(model, *args, **kwargs):
    model = prepare_model(model, *args, **kwargs)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    official.prepare_model_for_kbit_training = prepare_with_checkpointing
    args = official.parse_arguments()
    print("Full official training; seed=42, gradient checkpointing enabled", flush=True)
    print(vars(args), flush=True)
    official.main(args)
