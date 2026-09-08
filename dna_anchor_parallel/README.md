# Four-A100 D3LM anchor census

Continues the original full-corpus experiment without changing its model,
sampling, mask seeds, discovery/evaluation budgets, candidate coverage, teacher
tie-breaks or cache schema. This is cache collection, not an MDM decoding-speed
experiment or a trained policy result.

## Utilization design

- Four resident FP32 models on four distinct A100-80GB GPUs.
- Four independent CPU context workers per GPU (two CPU threads each), plus
  two CPU threads per inference server. Requested: 48 CPU cores and 128GB RAM.
- A bounded shared work queue balances contexts across all workers. GPU request
  queues overlap CPU entropy/statistics, feature export and archive
  verification with the next client's GPU inference.
- No mixed precision, TF32, request merging, extra sampling, reduced evaluation
  or selection based on fresh outcomes. Native GPU batch size remains 32.
- Actual utilization, memory and power for **each GPU UUID** are sampled every
  five seconds. Server logs separately count empty-queue time and time serving
  inference/transfers. A claimed 100% utilization is not guaranteed: startup,
  tails, CPU/PCIe stalls, and scheduler interruptions are measured limitations.

## Safety and parity

The old `dna_anchor_census` source is unchanged and its source hashes, source
manifest, checkpoint, package versions and experiment contract must match.
Existing completion markers/archives are resumed, never overwritten. One global
lock excludes the old worker. An execution manifest binds the new runner.

Before production, each GPU reproduces one already archived context (indices
1–4). Native sampled token IDs, masks, both selected teacher positions, all
discrete unlock outcomes and artifact tensors are compared. Discrete fields
must match exactly; floating fields use `atol=2e-5, rtol=1e-6`. A mismatch stops
the entire pipeline. Parity does not establish corpus-wide scientific benefit.

Only fully verified tar archives receive completion markers. A crash between
archive and marker is recovered on restart. Partial scratch is not reused or
deleted. Completed archives retain sampled anchors, every candidate's child
statistics, both teacher labels, hidden states and probability-change arrays.

## Unity

From the actual remote mddm checkout:

```bash
sbatch dna_anchor_parallel/run_unity.sbatch
# Optional bounded end-to-end performance test (same full-corpus contract):
sbatch --time=00:30:00 dna_anchor_parallel/run_unity.sbatch --max-new 256
```

Uses `gpu-preempt`, exactly four typed A100 GPUs, one node, and 45 hours.
USR1/TERM stops assigning new contexts and drains current ones; it does not
automatically submit another job. Preemption and queue time can exceed any
calendar deadline. `--requeue` is a Slurm eligibility flag, not a guarantee.

Read `.runtime/all_species_v1/status.json` for total progress, rolling measured
ETA and each GPU's last-180-second utilization. Per-invocation evidence lives
in `parallel_runs/<job>-<timestamp>/`: execution contract, parity checks, forward
journals, context completion events and `gpu_utilization.csv`.

**No plots or aggregate reports run in the GPU job.** Only per-context statistics
needed to select the anchors, and reusable cache artifacts, are computed there.
Submit the separate CPU-only plotting job later:

```bash
sbatch dna_anchor_parallel/plots_unity.sbatch
```

That job reads caches, regenerates the same aggregate plots as the original
analysis and renders individual examples. It does not load a model or use a GPU.

Tests (in the task's pinned Python environment):

```bash
PYTHONPATH=dna_anchor_parallel:dna_anchor_census python -m unittest discover -s dna_anchor_parallel -p 'test_*.py' -v
PYTHONPATH=dna_anchor_census python -m unittest discover -s dna_anchor_census -p 'test_*.py' -v
```
