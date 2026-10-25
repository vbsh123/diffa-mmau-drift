# Official-FPD cache benchmark

Same frozen **50 MMAU-Pro questions** and reasoning prompt as the drift study,
128 maximum answer tokens, official confidence-adaptive FPD and natural EOS.
This is a separate package/job, with no changes to `drift_study/` or its source
fingerprint. It does not modify the running drift experiment or its outputs.

| Name | Acoustic | Fixed text | Semantic |
|---|---:|---:|---:|
| 161616 | 16 | 16 | 16 |
| 888 | 8 | 8 | 8 |
| 16816 | 16 | 8 | 16 |
| 16168 | 16 | 16 | 8 |
| 323216 | 32 | 32 | 16 |
| 321632 | 32 | 16 | 32 |

These are prefix refresh intervals in **actual FPD forwards**, not token counts
or block counts. All selected layers initially use the same triplet. Every answer
row is recomputed on every forward, including committed answers and future MASK
slots. All K/V columns remain present. Original RoPE positions, full embedding,
final norm and full vocabulary head are retained.

On reuse steps, the selected layers physically skip prefix Q/K/V projections,
attention queries/output, and MLP rows. They reuse post-RoPE K/V and post-block
hidden states for those prefix positions. This is approximate bidirectional
caching; no tokens are pruned and there is no sparse attention mask.

## Start from the measured mean

Default `CACHE_START=mean`:

1. Read all **50 completed uncached drift results** and their original manifest.
2. Verify sample IDs/order, actual WAV hashes, prompts, targets, dataset revision,
   decoder settings, model revisions, seed, source fingerprint, and forward traces.
3. Compute the arithmetic mean of actual forward counts and set `M=ceil(mean)`.
4. Recompute densely for the first M forwards of **every** policy/question.
   Initialize caches on the last of those dense forwards. On the next forward,
   begin reuse, refreshing each group every configured interval from that anchor.

Example: mean 25.5 gives M=26. Zero-based forwards 0–25 are dense, forward26 can
reuse, and interval8 refreshes again at forward33 (eight forwards after anchor25).

The threshold is frozen before cache decoding. No question's own future EOS or
cache-policy answer length is used. The same50 questions provide calibration and
evaluation, so this is **development data**, not held-out validation.

Short answers ending during the dense phase remain in the results with **zero
actual reuse**. Their cached path should reproduce the baseline. The report shows
how many examples actually reused states; do not assume all50 reach M.

Other explicit start modes:

- `CACHE_START=immediate`: first forward initializes caches, subsequent forwards
  reuse according to the intervals.
- `CACHE_START=half-mean`: M=ceil(mean/2), otherwise identical semantics.

The default run contains just the selected start mode, not all three modes.
Different modes and layer selections get separate default output directories.

## Layer selection is ready for the drift findings

Default `CACHE_LAYERS=all` applies caching at all32 transformer blocks. To restrict
caching to deeper layers later, use e.g. `CACHE_LAYERS=16-31`. Explicit lists such
as `CACHE_LAYERS=16,20,24` and mixed ranges such as `8-15,24-31` also work.

Indices are **zero-based transformer blocks**, matching the drift measurements;
block0 is not the embedding. Layers outside the selection recompute all rows
normally on every forward. There is no hard-coded deep/shallow cutoff or inferred
layer choice before the drift results are inspected. Each selected layer has its
own K/V/output cache. Accounting records computed rows separately for every layer.

## Vast command

**Finish the drift job first. Do not run both jobs on the same GPU.** Clone a
separate checkout for caching; do not pull changes into an active drift checkout.
The original run must retain its `config.json`, `samples/*/result.json`, forward
traces and frozen manifest, not just the analysis report.

```bash
cd /workspace
git clone https://github.com/vbsh123/diffa-mmau-drift.git diffa-mmau-cache
cd diffa-mmau-cache
nohup env \
  SKIP_INSTALL=1 \
  AUDIO_VENV=/workspace/diffa-mmau-drift/.venv \
  DIFFA_ROOT=/workspace/diffa-mmau-drift/external/DIFFA \
  BASELINE_RUN=/workspace/diffa-mmau-drift/runs/mmau50_fpd128 \
  bash scripts/cache.sh > cache50_mean_all.log 2>&1 &
tail -f cache50_mean_all.log
```

This reuses the installed environment and official source. HF model downloads
are reused through the normal shared cache. The new checkout prepares the same
small deterministic audio cohort independently. If the paths differ, adjust them.
`BASELINE_MANIFEST` overrides the assumed original `<drift-repo>/data/mmau50.json`.
On a new machine omit `SKIP_INSTALL=1` and `AUDIO_VENV` to install dependencies.

All50 baseline results are required before any model loading. To inspect the
resolved mean/start point without loading a model:

```bash
source /workspace/diffa-mmau-drift/.venv/bin/activate
python -m cache_study.run \
  --baseline-run /workspace/diffa-mmau-drift/runs/mmau50_fpd128 \
  --manifest data/mmau50.json --preflight-only
```

For a three-question smoke run add `SAMPLES=3 RUN_NAME=runs/cache_smoke3` to the
`env` command. Calibration still requires all50 completed original questions.
For a later deep-layer experiment add `CACHE_LAYERS=16-31` after choosing the
range from the drift results. The default output becomes `runs/cache50_mean_16-31`.

## Timing, verification and outputs

- **350 timed generations**: one fresh uncached baseline and six cache policies
  on each of50 questions. Conditions are shuffled deterministically per question.
- One process warmup per condition, not per question. Cache warmups use immediate
  reuse to exercise the cache path; their outputs/times never enter quality/timing
  aggregates. One clean timed generation per condition; no timed repeats.
- First two questions additionally check refresh-every-forward identity caching
  against current baseline, requiring exact full canvas IDs and forward count.
- Every generation starts with an empty cache. Actual generation uses the same
  untouched official FPD function, with confidence-adaptive commits and natural EOS.
- No drift/attention probes, activation histories, fixed-input replay, scoring,
  or file I/O inside decode timers. Cache setup/management and synchronization
  are included. A 1024-forward watchdog aborts a stalled decode without forcing
  commits; failures remain incorrect in accuracy and are counted explicitly.
- The historical instrumented drift timing is **never** used for speedup. Its
  forward counts calibrate M. Fresh baseline outputs are compared with saved
  drift outputs and the match flag is recorded.
- Final texts/IDs, grades, EOS/capacity flags, lengths, forward counts, seconds,
  refresh counts, actual reuse forwards, per-layer row work and policy order are
  saved immediately per completed condition. An unchanged rerun resumes those
  conditions without rerunning their timed measurements. New source/settings need
  a fresh output root.

Read `runs/cache50_mean_all/analysis/report.md`, `summary.json`, and `pairs.csv`.
Reports update after each condition. Paired losses/recoveries use common examples;
paired latency speedup uses successful baseline/cache pairs. Failures remain in
accuracy but are excluded from speed/TPS, with sample counts exposed. Autonomous
generation lengths and NFE can change, so latency is not an equal-work replay.

CPU tests compare physical row skipping to an independent dense stale-K/V and
stale-output reference for every requested schedule, both immediate/delayed starts,
and all/deep-only layer selections. They check that uncached layers and all answer
rows stay fresh, late-start/no-reuse outputs are exact, BF16 matches the reference,
and native FPD identity caching preserves output IDs/NFE. Full-model GPU quality
and speed remain to be measured.
