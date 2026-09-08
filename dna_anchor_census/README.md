# D3LM DNA anchor analysis on one A100

Repeat the GSM8K-style before/after anchor analysis on frozen
[D3LM-from-nt](https://huggingface.co/Hengchang-Liu/D3LM-from-nt) and
[EPD-GenDNA](https://huggingface.co/datasets/Zehui127127/latent-dna-diffusion).
No model or policy training and no throughput benchmark.

The full pinned CSV has 159,123 rows and 156,403 unique eligible 2,048-base ACGT
sequences. Duplicates and exclusions are inventoried. The original four-species
population (61,236 sequences) is reported separately from the other species.
The 505 previously inspected sequences are development, never fresh confirmation.
The whole GPU run is a new protocol; it does not mix old CPU outcomes into its
averages. Exact pretraining overlap and homology separation are not established.

## Both teachers, same root

Mask 170 six-mer positions. Test **every masked position**, using two ordinary
native T=1 samples per position. Never inject a gold anchor. Rank positions two
ways, before eight independent fresh draws for each teacher and each control:

| Teacher | Primary target | Deterministic ties |
| --- | --- | --- |
| IG | Mean summed entropy reduction on other masked tokens | More newly confident tokens, then lowest position |
| Max-new | Mean count of other tokens newly crossing 95% confidence | Larger net confidence count, larger IG, then lowest position |

IG uses the definition in `mdm_probe.metrics.information_gain` (entropy reduction,
reported here in bits), not KL divergence or joint mutual information. Max-new
is a model-confidence target, not merely IG under another name. Newly correct,
newly wrong, and the historical reference-correct GSM8K `candidate_key` label
are also saved separately; reference agreement is not biological validity.
The historical key orders gain, gain, correct-after, negative-new-wrong,
anchor log-probability, negative-position (default oracle/gain variant).
Source hashes at binding: metrics.py
`f59bc00e75af991100cebbc856d3da16a1956e8c55aa8ddddec3c33b4498b436`;
precompute_threshold_unlock_targets.py
`4707bbaffd0fa26c2f93ff4714a8ca59e8c9024fe31d78a044a22b4be870b5e3`.

Confidence-first, random and earliest-position controls reveal one token each.
All five arms score the same remaining positions, excluding all their anchors.
Primary threshold is 95%; the earlier DNA report's 90% counts are also retained.
All fresh samples, zero unlocks, negative IG and non-DNA proposals are retained.
This is an offline lookahead teacher, not a deployed efficient policy or proof
that the whole distribution has useful anchors. Selection covers every masked
position but not every possible token value or masking state.

## Reusable cache and plots

Each verified `cache/partNNNN/eNNNNNN.tar` retains:

- root input/reference IDs, logits, last-layer hidden states and mask positions;
- all native proposal draws and sampled anchor token IDs;
- all tested singleton child inputs, predictions, confidences, entropy and
  reference probabilities, including losing candidates;
- both teacher labels, tie rules, full discovery scores and fresh outcomes;
- dense before/after probabilities for every common target, not just successes;
- artifact checksums, source identity, deterministic seeds and split flags.

Full child distributions are kept for the first two distinct positive children
per sequence and all children of every 500th sequence. Child sufficient statistics
are kept for every intervention; the storage cap never filters measured outcomes.
Per-context tar files reduce shared-filesystem small-file load. Generated local
scratch files are removed only after their archive contents are hash-verified;
they remain recoverable from the tar. Results and models are not committed to Git.

Plots include before/after probability and entropy by token position, candidate
IG versus candidate unlock count, same-token probability-shootup scatter plots,
newly unlocked counts versus summed probability gains across sequences,
and both teachers versus controls. Illustrations use the first fresh draw on
predetermined sequences, not the best observed draw. Corpus summaries include
all outcomes. Train/validation assignment is exact-sequence-hash based only;
perform appropriate homology-grouped splitting before a confirmation claim.

This singleton, one-step cache can train auxiliary anchor heads. It contains
neither same-state joint-action returns nor completed-generation safety/NFE
labels; do not manufacture those from singleton labels or confidence gains.

## Unity

From the mddm repository root, run `sbatch dna_anchor_census/run_unity.slurm`.
The job requests `--gres=gpu:a100:1` **and** `--constraint=a100`. Python refuses
zero GPUs, more than one visible GPU, or any GPU whose name lacks A100.
It matches `gsm8k_temperature_sweep/slurm/submit_all.sbatch` on the
`gpu-preempt` partition, 45-hour limit and explicit `--requeue` flag.
The script reuses the existing project's PyTorch interpreter read-only and
installs separate task-local dependencies. Checkpoints and plots go under
`dna_anchor_census/.runtime/all_species_v1/`. Resume with the same command and
state directory after a verified interruption; never launch a second overlapping
copy. A lock prevents concurrent writers. Only one A100 is allocated.
