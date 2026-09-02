"""Canonical artifact sources for the standalone sweep."""
from __future__ import annotations

from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent

BASE_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
DPARALLEL_MODEL_ID = "Zigeng/dParallel-LLaDA-8B-instruct"

# Public, unofficial reproduction of the Apple confidence-only GRPO policy
# trained with group size 8 on the GSM8K+MATH mixture.  Use the uploader's
# reward-selected checkpoint rather than the collapsed final training step.
PAPER_POLICY_REPO_ID = "orkunkinay/ml-rl-dllm-gs8"
PAPER_POLICY_FILENAME = "checkpoint-best/model.safetensors"

# The exact standard GSM8K LoRA used by the existing mddm full256 baseline.
# It is committed with this standalone experiment so a clean clone needs no
# shared-filesystem checkpoint path or environment variable.
SFT_ADAPTER_PATH = EXPERIMENT_ROOT / "artifacts" / "gsm8k_lora_sft"
