# Four-A100 census continuation

Implemented and submitted as Unity job64078116 (pending priority at03:58UTC).
Latest user direction: parallel across examples; GPU job only collects reusable
caches, with no plots/aggregate reporting. Plotting has a separate CPU-only job.
The notes below describe the original implementation plan, now implemented.

User requests full eligible corpus within a47-hour deadline, parallelizable with
good utilization across four A100s on Unity. User explicitly cancelled the
single-GPU run; do not restart job64076757. It stopped cleanly with440 completed
contexts and preserved reusable caches in all_species_v1.

Measured serial execution: last100 contexts475.082seconds; about206A100-hours
remaining at that pace. Four copies alone project51–52hours, not within47hours.
Slurm measured26percent averageGPUutilization, suggesting CPU/postprocessing gaps.
This is an optimization opportunity, not proof of improved performance.

Implementation direction:

- Keep dna_anchor_census source unchanged and hash-bound; add a sibling runner.
- Four independent GPU servers, each strictly one visible A100, with CPU context
  workers pipelining the unchanged native_protocol.run_context calls.
- Preserve native FP32/noTF32 forward path, batch32, CPU proposal sampling,
  all170positions/two discovery draws/eight fresh draws/five arms, both teachers,
  allnegative outcomes, full cache and plots. No shorter or easier experiment.
- Preserve existing per-sequence seeds, indices and cache schema. Reuse440 saved
  contexts only after compatibility tests. Disjoint work ownership/locks,
  durable archives/completion markers, and preemption-resumable status required.
- Separate GPU inference from CPU statistics/cache writing to improve duty cycle;
  no automatic assumption that more workers helps. Measure short end-to-end run.
- Compare sampled values, discrete selections/unlock counts, probabilities and
  rootfeatures against original outputs; never fabricate bitwise parity.
- Check four-rank identity/unique UUIDs, cache merge and no duplicates/omissions.
- Use gpu-preempt and the existing45-hour allocation limit, leaving margin within
  requested47hours. Queue delays/preemption make a strict calendar guarantee
  impossible without reservation; state that rather than promising one.
- Push only scoped files from this isolated worktree; pull in the actual Unity
  mddm checkout. No four-GPU job has yet been submitted.

Next: implement/test pipeline and restart-safe rank ownership; run a bounded
four-GPU parity/performance preflight before committing to a completion ETA.

User update: no plots or aggregate reporting in the GPU allocation. Save caches
only; provide a separate CPU-only plotting sbatch for later submission.
