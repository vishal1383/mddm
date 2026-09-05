# Full GSM8K temperature and pass@k sweep

This folder is a self-contained, resumable train-and-evaluate chain for seven LLaDA-family methods on GSM8K. The priority order is deliberate: it evaluates the released JustGRPO checkpoint first, then trains Apple's unmasking-policy method from scratch, saves and evaluates that policy, and finally completes/reuses the remaining baselines and local DPO run. Every evaluation covers the complete official 1,319-example test split at token temperatures `T = {0.1, 0.5, 0.8, 1.2}`.

## Methods

| Key | Evaluated method | Checkpoint/model |
|---|---|---|
| `base` | Frozen Base confidence decoder | `GSAI-ML/LLaDA-8B-Instruct` |
| `jsd_mean_field` | Training-free JSD pair-interaction fixed-point decoder | Frozen Base |
| `dparallel` | [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488) | `Zigeng/dParallel-LLaDA-8B-instruct` |
| `justgrpo` | [JustGRPO](https://arxiv.org/abs/2601.15165) | Released `nzl-thu/LLaDA-Instruct-JustGRPO-GSM8K` checkpoint, evaluated with the matched confidence decoder |
| `lora_sft` | Standard full-GSM8K LoRA SFT | Frozen Base plus the exact adapter bundled under `artifacts/gsm8k_lora_sft` |
| `apple_policy_rl` | [Learning Unmasking Policies for Diffusion Language Models](https://arxiv.org/abs/2512.09106) | Frozen Base plus a policy newly trained by the pinned official Apple code |
| `dpo_policy_v3` | Hidden-state select-then-sample DPO policy | Frozen Base plus a new two-block projected-hidden-state selector |

The JSD row implements the pairwise-distribution variational update described by [Mean-Field Parallel Decoding for Discrete Diffusion Language Models](https://arxiv.org/abs/2606.15805), using the exact selector already developed in this repository.

## Fixed matched contract

- Dataset: `openai/gsm8k`, `main`, official `test`, exactly 1,319 examples.
- Prompt: one identical LLaDA chat-template prompt for every method.
- Canvas: 256 completion tokens, 32-token semi-autoregressive blocks.
- Paths: exactly ten independently sampled recurrent trajectories per example and temperature.
- Token temperature: conventional categorical sampling from `softmax(logits / T)`.
- Base/JSD/LoRA decoder: global confidence threshold 0.90 and highest-confidence fallback.
- dParallel: entropy threshold 0.50 and minimum-entropy fallback.
- JustGRPO: the released GSM8K model checkpoint under the same threshold-0.90 matched confidence transition used for Base and LoRA; this is intentionally distinct from the paper's fixed-step reporting sampler.
- Apple policy: official confidence-only one-block DiT architecture, full context, Bernoulli-argmax evaluation, fixed policy temperature 0.5 (the paper's block-32 setting).
- Hidden-state DPO: two-block projected-hidden selector, Bernoulli evaluation at its matched training policy temperature 1.0; token temperature remains the table's `T`.
- NFE: one per full model forward per trajectory. Decoder/head work is not an NFE, but it is included in synchronized latency.
- `sample_accuracy`: marginal exact-match accuracy over all ten paths.
- `pass@5`: whether any of paths 0–4 is exact-match correct.
- `pass@10`: whether any of paths 0–9 is exact-match correct.
- `Tok/NFE`: `sum(generated tokens) / sum(full model forwards)` over all ten paths and all examples.

The six non-DPO methods contribute 24 full cells, and the DPO evaluation adds four, producing a 28-row final table and **369,320 complete trajectories**. Every cell is a full run, not a screen or promotion gate.

## Newly trained Apple policy

The chain pins Apple's official `ml-rl-dllm` source at commit `35e4830485f1821d57f9ac3f1a303f3d4531fb82`. It trains the paper's BL32 confidence-only Bernoulli policy for one epoch on the full proportional GSM8K+MATH mixture (roughly 15K examples), using eight trajectories per group, greedy Base token sampling, and the multiplicative correctness/compute reward with `alpha=0.3`. The 8-GPU paper configuration is adapted to one A100 by using a global batch of 16; policy architecture, rollout, reward, and GRPO loss are unchanged.

Periodic optimizer checkpoints make training resumable under preemption. The official reward-selected `checkpoint-best` is sealed by SHA-256 in `training_manifest.json` before any evaluation starts. The Base is explicitly frozen and contributes zero trainable parameters.

## Logit-free online-DPO policy

The DPO method keeps LLaDA fully frozen. Its position selector receives only a fixed 128-dimensional projection of the frozen Base final hidden states, the canvas mask, and normalized decoding time. It never receives token logits, confidence, entropy, margins, JSD, or dParallel features. Selected token values are subsequently sampled from the frozen Base conditional, preserving the select-then-sample separation.

For each training prompt, ten trajectories are sampled from the current hidden-state policy using different action-rate offsets. The fastest correct trajectory is preferred to the fastest incorrect trajectory for safety and to the slowest correct trajectory for efficiency. Incorrect-only prompts and tied outcomes fabricate no preference. The head is updated immediately with the standard reference-relative DPO loss, so subsequent trajectories are on-policy with respect to the latest head. There is no scalar reward, critic, PPO/GRPO update, trainable Base parameter, logit-derived behavior selector, or official-test training signal.

## Artifact resolution

No checkpoint-path exports are required. Base, dParallel, and JustGRPO are downloaded from their Hugging Face repositories and sealed to immutable revisions. The exact standard-LoRA adapter used by the existing mddm full256 baseline is committed inside this standalone folder. The Apple and hidden-state DPO stages write resumable state under `final_results/checkpoints/apple_policy_rl` and `final_results/checkpoints/dpo_policy_v3` respectively.

## Unity setup and launch

Minimal single-entrypoint launch from the `mddm` repository root:

```bash
mkdir -p gsm8k_temperature_sweep/final_results/manifests
JOB_ID=$(sbatch --parsable gsm8k_temperature_sweep/slurm/submit_all.sbatch)
echo "Submitted job: $JOB_ID"
tail -n 100 -F \
  "gsm8k_temperature_sweep/final_results/manifests/slurm-${JOB_ID}.out" \
  "gsm8k_temperature_sweep/final_results/manifests/slurm-${JOB_ID}.err"
```

By default, all large state stays inside the project clone: the environment,
Hugging Face cache, and upstream policy checkout use
`gsm8k_temperature_sweep/.runtime`, while checkpoints, records, predictions,
and tables use `gsm8k_temperature_sweep/final_results`.

The submitted job itself owns one A100 on `gpu-preempt`. It bootstraps and validates the environment, then runs every stage sequentially inside that single allocation—there are no nested `sbatch` calls:

No checkpoint-path exports are required. Hugging Face authentication uses the existing login/environment cache when needed.

```text
4 JustGRPO full-test cells
  -> full Apple GSM8K+MATH GRPO training and saved checkpoint
  -> 4 saved-Apple-policy full-test cells
  -> finish/reuse Base, JSD, dParallel, and LoRA cells
  -> saved 24-row non-DPO table
  -> full 7,473-example DPO collection/training and 4 full-test cells
  -> saved 28-row final table
```

`submit_all.sbatch` explicitly requests `--partition=gpu-preempt` and `--gres=gpu:a100:1`, with no separate QoS. It has a 45-hour wall-time, receives `USR1` before preemption, saves atomic progress, and requeues the same job. On restart, valid completed cells are skipped without loading an 8B model; evaluation records, DPO preference records, DPO trainer state, and sealed model revisions are reused. There are no accuracy or throughput gates.

## Outputs

```text
$MDDM_SWEEP_OUTPUT_ROOT/
  base/T0.1/{contract.json,runtime_manifest.json,records/,summary.json}
  jsd_mean_field/T0.1/...
  dparallel/T0.1/...
  justgrpo/T0.1/...
  lora_sft/T0.1/...
  apple_policy_rl/T0.1/...
  dpo_policy_v3/T0.1/...
  checkpoints/apple_policy_rl/{training_contract.json,training_manifest.json,checkpoint-best/,checkpoint-*/}
  checkpoints/dpo_policy_v3/{training_contract.json,training_manifest.json,trainer_resume.pt,model.safetensors,preferences/}
  tables/{baseline_table.csv,baseline_table.md,baseline_all_summaries.json}
  tables/{final_table.csv,final_table.md,all_summaries.json}
  logs/
```

Before running GPU work, the controller resolves Base, dParallel, and JustGRPO to immutable commit SHAs and verifies the pinned Apple source. Contracts also seal adapter/policy hashes, evaluator source, thresholds, geometry, prompt, and seed. The intermediate table contains 24 cells. Final aggregation fails if any of the 28 summaries is absent, malformed, or not a full 1,319-example result.

For a manual table rebuild:

```bash
"$MDDM_SWEEP_VENV/bin/python" aggregate.py --output-root "$MDDM_SWEEP_OUTPUT_ROOT"
```

For local code checks that do not load an 8B model:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile *.py
```
