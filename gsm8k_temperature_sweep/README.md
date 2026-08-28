# Full GSM8K temperature and pass@k sweep

This folder is a self-contained, resumable train-and-evaluate chain for six LLaDA-family methods on GSM8K. It first completes the original five-method table, then trains an offline-DPO version of Apple's unmasking policy on all 7,473 training examples, evaluates it on the complete official 1,319-example test split, and creates a final matched table over token temperatures `T = 0.1, 0.2, ..., 1.2`.

## Methods

| Key | Evaluated method | Checkpoint/model |
|---|---|---|
| `base` | Frozen Base confidence decoder | `GSAI-ML/LLaDA-8B-Instruct` |
| `jsd_mean_field` | Training-free JSD pair-interaction fixed-point decoder | Frozen Base |
| `dparallel` | [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488) | `Zigeng/dParallel-LLaDA-8B-instruct` |
| `paper_policy` | [Learning Unmasking Policies for Diffusion Language Models](https://arxiv.org/abs/2512.09106) | Frozen Base plus supplied policy checkpoint |
| `lora_sft` | Standard full-GSM8K LoRA SFT | Frozen Base plus supplied adapter |
| `dpo_policy` | Offline trajectory-DPO unmasking policy | Frozen Base plus a newly trained Apple-architecture policy head |

The JSD row implements the pairwise-distribution variational update described by [Mean-Field Parallel Decoding for Discrete Diffusion Language Models](https://arxiv.org/abs/2606.15805), using the exact selector already developed in this repository.

## Fixed matched contract

- Dataset: `openai/gsm8k`, `main`, official `test`, exactly 1,319 examples.
- Prompt: one identical LLaDA chat-template prompt for every method.
- Canvas: 256 completion tokens, 32-token semi-autoregressive blocks.
- Paths: exactly ten independently sampled recurrent trajectories per example and temperature.
- Token temperature: conventional categorical sampling from `softmax(logits / T)`.
- Base/JSD/LoRA decoder: global confidence threshold 0.90 and highest-confidence fallback.
- dParallel: entropy threshold 0.50 and minimum-entropy fallback.
- Paper policy: official confidence-only one-block DiT architecture, full context, Bernoulli-argmax evaluation, fixed policy temperature 0.5 (the paper's block-32 setting).
- NFE: one per full model forward per trajectory. Decoder/head work is not an NFE, but it is included in synchronized latency.
- `sample_accuracy`: marginal exact-match accuracy over all ten paths.
- `pass@5`: whether any of paths 0–4 is exact-match correct.
- `pass@10`: whether any of paths 0–9 is exact-match correct.
- `Tok/NFE`: `sum(generated tokens) / sum(full model forwards)` over all ten paths and all examples.

The original table is 5 methods × 12 temperatures × 1,319 examples × 10 paths = **791,400 complete trajectories**. The DPO evaluation adds 12 × 1,319 × 10 = **158,280 trajectories**, producing a 72-row final table. Every cell is a full run, not a screen or promotion gate.

## Offline-DPO policy

The DPO baseline uses the paper's exact confidence-only, one-block DiT Bernoulli policy and keeps LLaDA fully frozen. After the original 60 rows finish, it collects four deterministic frozen-Base trajectories per training prompt using confidence thresholds `0.30, 0.50, 0.70, 0.90`. It ranks paths with the paper's multiplicative terminal reward:

```text
correct * ((L - min(NFE, L) + 1) / L) ** alpha
```

Every strict within-prompt reward ordering becomes an offline preference pair. Tied paths—especially pairs of incorrect paths—produce no preference, preventing a fast-but-wrong training signal. The policy is initialized from and optimized relative to a frozen smart-initialized reference head using the standard pairwise DPO log-ratio loss. There is no critic, GRPO/PPO update, trainable Base parameter, RL checkpoint initialization, or official-test training signal.

## Checkpoint prerequisite

The Apple source release provides training/evaluation code and expects evaluation to receive a locally trained `checkpoint-*/model.safetensors`; it does not include a pretrained policy weight file in the repository. Therefore the sweep requires a real policy checkpoint through `UNMASKING_POLICY_CHECKPOINT`. It never substitutes random policy weights or silently omits the row.

Likewise, the standard LoRA adapter is an artifact, not part of Git. Copy both artifacts to storage visible from every Unity compute node before submission:

```bash
export UNMASKING_POLICY_CHECKPOINT=/project/your_group/checkpoints/unmasking/checkpoint-XXXX/model.safetensors
export SFT_ADAPTER_PATH=/project/your_group/checkpoints/gsm8k_lora/adapter-final
export HF_TOKEN=hf_...
```

If the unmasking checkpoint still needs to be trained, the pinned upstream recipe is:

```bash
cd "$ML_RL_DLLM_REPO"
accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
```

## Unity setup and launch

Minimal single-entrypoint launch (after setting the three artifact/authentication variables):

```bash
mkdir -p final_results/manifests
JOB_ID=$(sbatch --parsable slurm/submit_all.sbatch)
echo "Submitted job: $JOB_ID"
tail -n 100 -F "final_results/manifests/slurm-${JOB_ID}.out" "final_results/manifests/slurm-${JOB_ID}.err"
```

The controller bootstraps and validates the environment, then submits one autonomous dependency chain:

```text
60 original full-test cells (sequential)
  -> saved 60-row table
  -> full 7,473-example DPO collection and training
  -> 12 DPO full-test cells (sequential)
  -> saved 72-row final table
```

After pulling `mddm` on Unity:

```bash
cd gsm8k_temperature_sweep
export MDDM_SWEEP_STATE_ROOT="${SCRATCH}/mddm-gsm8k-passk"
export MDDM_SWEEP_OUTPUT_ROOT=/project/your_group/mddm-gsm8k-passk-results
export SFT_ADAPTER_PATH=/project/your_group/checkpoints/gsm8k_lora/adapter-final
export UNMASKING_POLICY_CHECKPOINT=/project/your_group/checkpoints/unmasking/checkpoint-XXXX/model.safetensors
export HF_TOKEN=hf_...

bash scripts/bootstrap_env.sh
export MDDM_SWEEP_VENV="$MDDM_SWEEP_STATE_ROOT/venv"
export ML_RL_DLLM_REPO="$MDDM_SWEEP_STATE_ROOT/ml-rl-dllm"
bash scripts/submit.sh
```

`submit.sh` runs a filesystem/revision/task-matrix preflight and submits the complete dependency chain to `gpu-preempt`, capped at one active A100. All GPU work is sequential. Dependencies check process success only; there are no accuracy or throughput gates.

The equivalent direct array command is:

```bash
sbatch --partition=gpu-preempt --array=0-59%1 --gpus=a100:1 \
  --output="$MDDM_SWEEP_OUTPUT_ROOT/logs/%A_%a.out" \
  --error="$MDDM_SWEEP_OUTPUT_ROOT/logs/%A_%a.err" \
  --export=ALL slurm/sweep.sbatch
```

The direct array command above runs only the original 60 cells; use `slurm/submit_all.sbatch` for the autonomous DPO extension. Set `MDDM_SWEEP_ARRAY_LIMIT` or `MDDM_SWEEP_GPUS` only when intentionally overriding the one-A100 sequential default. Unity preempt jobs may be killed after their grace period, so evaluation records, DPO preference records, and DPO trainer state are resumable.

## Outputs

```text
$MDDM_SWEEP_OUTPUT_ROOT/
  base/T0.1/{contract.json,runtime_manifest.json,records/,summary.json}
  jsd_mean_field/T0.1/...
  dparallel/T0.1/...
  paper_policy/T0.1/...
  lora_sft/T0.1/...
  dpo_policy/T0.1/...
  checkpoints/dpo_policy/{training_contract.json,training_manifest.json,model.safetensors,preferences/}
  tables/{final_table.csv,final_table.md,all_summaries.json}
  logs/
```

Before submitting GPU work, the controller resolves both Hugging Face model references to immutable commit SHAs and exports those same SHAs through every baseline, DPO-training, and DPO-evaluation job. Contracts also seal adapter/policy hashes, evaluator source, thresholds, geometry, prompt, and seed. The intermediate table contains the original 60 cells. Final aggregation fails if any of the 72 summaries is absent, malformed, or not a full 1,319-example result.

For a manual table rebuild:

```bash
"$MDDM_SWEEP_VENV/bin/python" aggregate.py --output-root "$MDDM_SWEEP_OUTPUT_ROOT"
```

For local code checks that do not load an 8B model:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile *.py
```
