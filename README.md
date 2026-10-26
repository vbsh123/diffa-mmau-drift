# DIFFA-2 long-audio drift with official FPD

The default study now selects **50 complete MMAU-Pro clips longer than 60 seconds**.
There is **no fixed maximum clip duration**, no `short` metadata filter, and no
24MiB payload cutoff. Entire clips must fit the pinned backbone's actual context
budget: 4096 positions including audio, text and 128 answer slots. A clip that does
not fit is excluded as a whole with a recorded `context_budget` reason; it is
never cropped, summarized, or silently truncated.

The previous <=30-second experiment is preserved on branch `short-audio-v1`.
Its manifests and outputs are separate. The new default uses
`data/mmau50_long.json` and `runs/mmau50_long_fpd128` and refuses to reuse an old
short-cohort manifest.

## Complete audio encoding

Whisper still has its native 30-second encoder window. This is an **encoder window
size**, not a limit on the duration of a dataset clip. The input path partitions
all resampled samples into balanced, adjacent windows of at most 480000 samples.
Every sample belongs to exactly one window. Each window is encoded using the
original Whisper, acoustic connector and semantic connector. Projected tokens
are concatenated chronologically as `[A1,S1,A2,S2,...]`.

All acoustic and semantic tokens are retained. Acoustic tokens are **64 per
window**, not 64 per complete long clip. Semantic tokens grow with total duration.
For example, 90 seconds produces 1125 semantic plus 192 acoustic tokens: 1317 audio
tokens, compared with roughly 304 in the old short-audio cohort. Actual chunk
sample boundaries and token counts are saved for every result. The three token
groups use their true, potentially noncontiguous positions.

This is a documented long-input extension around the original encoder/connectors.
The **official FPD decoder remains unchanged**: 128 maximum answer tokens,
32-token blocks, factor 1.0, temperature 0, confidence-adaptive commits, natural EOS.
The prompt explicitly requests brief step-by-step audio-grounded reasoning and
`Final answer: <letter>`. There is no forced answer length or fixed commit quota.
Long-input answer quality must be evaluated on the GPU; CPU validation of the
input path is not a claim that the checkpoint was trained for this extension.

## Vast: new checkout and run

Use an A40 48GB or another suitable BF16 CUDA GPU, Python 3.10–3.12, a CUDA 12.4
compatible driver and approximately 150GB free disk. Long-prefix drift and matrix
collection cost more than the old short run; runtime is not yet measured.

Use a fresh checkout, leaving any earlier process/output untouched. If the old
installation is at the previously documented paths, reuse it:

```bash
cd /workspace
git clone https://github.com/vbsh123/diffa-mmau-drift.git diffa-mmau-long
cd diffa-mmau-long
nohup env SKIP_INSTALL=1 \
  AUDIO_VENV=/workspace/diffa-mmau-drift/.venv \
  DIFFA_ROOT=/workspace/diffa-mmau-drift/external/DIFFA \
  bash scripts/run.sh > mmau50_long_fpd128.log 2>&1 &
tail -f mmau50_long_fpd128.log
```

On a machine without that installation, run `nohup bash scripts/run.sh` instead;
setup creates `.venv`, installs pinned dependencies and fetches official source.
The repository is public; GitHub authentication is unnecessary. Do not run this
alongside another inference job on the same GPU.

Setup/preparation without model inference:

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m drift_study.data --out data/mmau50_long.json --samples 50
```

`MIN_SECONDS` defaults to 60; `MAX_SECONDS` is unset by default. These explicit
user overrides also work through `scripts/run.sh`. The model token budget is
always checked. The sampler reads original WAV headers selectively from the
47GB archive before fetching eligible complete files. The small tokenizer/config
are loaded for token-budget preflight; model weights are not loaded at this stage.

Optional GPU smoke run of the first three of the same 50 examples:

```bash
nohup env SKIP_INSTALL=1 \
  AUDIO_VENV=/workspace/diffa-mmau-drift/.venv \
  DIFFA_ROOT=/workspace/diffa-mmau-drift/external/DIFFA \
  bash scripts/smoke.sh > smoke3_long_fpd128.log 2>&1 &
```

The first two examples compare untouched official FPD with observed FPD using
identical windowed embeddings: exact complete output IDs and forward count must
match. Every actual window's projected token count is checked against preflight.
Completed examples resume only under unchanged source/config/environment.
Interrupted examples restart. No old output is overwritten.

## Cohort and observations

Seed 20261025; 17 speech, 17 sound, 16 music. Single-audio MCQs, 2–26 distinct options,
exactly one matching reference, complete finite mono WAVs. No correctness-based
selection. All published duration labels are eligible; actual full-file headers
and model token fit determine admission. The pinned dataset revision is
`b8516c217464d53be36d6dc334fda3cd578ed103`.

Hidden-state drift is measured at **all 32 transformer block outputs**, every
actual forward, for semantic audio, acoustic audio and fixed input text.
Generated answer slots are excluded from these groups. The primary statistic
is mean per-position FP32 cosine distance against the previous forward; initial
reference and flattened-group cosine are also saved. Examples have equal weight.
Single-forward outputs have no adjacent comparison and are counted separately.
All answers, including incorrect/capacity-limited answers, remain in the analysis.

Step-pair matrices use the first example of each domain, selected beforehand,
at blocks 0, 7, 15, 23, 31. BF16 states are saved losslessly to temporary scratch;
small token chunks are expanded for pairwise computation and scratch is deleted.
Attention is also measured by default at every layer and forward. FP32 softmax
weights are reconstructed from actual post-RoPE queries/keys, over all heads and
positions, in bounded query chunks. Native SDPA outputs remain unchanged. Saved
summaries separate acoustic, semantic, fixed text, masked answer and decoded
answer queries/keys. Raw mass and mass relative to a uniform distribution over
positions are both reported; larger key groups naturally receive more raw mass.
Full attention matrices are not saved. `ATTENTION_EVERY=4` observes every fourth
forward; `ATTENTION_EVERY=0` disables attention only. Timings include observation overhead
and must not be reported as clean inference throughput.

VidLaDA Figures 4–5 motivate the cosine matrices and layer/step heatmaps. Its
released code does not specify every diagnostic reduction/hook detail, so our
block-output, token-mean reduction and references are documented explicitly.

## Results and cache benchmark

`runs/mmau50_long_fpd128/analysis/report.md` and `summary.json` update after each
example. Final figures include raw/normalized-progress drift heatmaps, contributor
counts and exemplar similarity matrices. Each sample retains output IDs/text,
strict grade, actual NFE, EOS/capacity flags, full chunk coverage, group counts,
trace, all-layer drift and artifact hashes. Config pins source, inputs and models.

Rebuild figures without loading a model:

```bash
python -m drift_study.report runs/mmau50_long_fpd128 --plots
```

Package outputs:

```bash
tar -czf mmau50_long_fpd128.tar.gz -C runs mmau50_long_fpd128
```

The [cache benchmark](CACHING.md) defaults to forward10 and twice as frequent
semantic refresh in blocks24–31 for all six requested policies. It times a fresh
baseline without drift/attention probes. Fixed start needs no historical baseline.

## One command for all three runs

Clone into a new folder so active/older runs remain resumable. Existing Vast
installation/source paths are reused below; adjust them if your paths differ.
On a fresh machine omit `SKIP_INSTALL=1` and `AUDIO_VENV`.

```bash
cd /workspace
git clone https://github.com/vbsh123/diffa-mmau-drift.git diffa-mmau-next
cd diffa-mmau-next
nohup env SKIP_INSTALL=1 \
  AUDIO_VENV=/workspace/diffa-mmau-drift/.venv \
  DIFFA_ROOT=/workspace/diffa-mmau-drift/external/DIFFA \
  bash scripts/next.sh > next.log 2>&1 &
tail -f next.log
```

Job1 runs the six forward10 layerwise caches plus a fresh baseline on the original
MMAU50. Job2 repeats on the new long-audio50, with its own fresh baseline. Job3
separately measures all32 layers' drift and attention on that long cohort.
The original frozen manifest and WAVs must still exist at
`/workspace/diffa-mmau-drift/data/mmau50.json`; set `ORIGINAL_MANIFEST` if elsewhere.
Every hash/prompt is checked before inference. The launcher fixes50 questions,
all32 cache layers, forward10, the deep semantic factor2, and every-forward
attention regardless of stale shell overrides. It clears any inherited duration
maximum. Use the individual scripts for deliberately different experiments. The new long cohort is prepared
automatically and is also MMAU-Pro, with different longer clips.
Output folders are `runs/cache50_original_forward10_all_deepsem2_24-31`,
`runs/cache50_long_forward10_all_deepsem2_24-31`, and
`runs/mmau50_long_fpd128_attention`. Finished results are bundled in
`runs/next_results.tar.gz`. Each output folder retains an exact `cohort.json` manifest. The final drift run
also requires matching output IDs, NFE, token groups and window coverage against
the clean long cache baseline on every question; a mismatch stops the run.
Every completed question/condition resumes under
unchanged source/settings. The command stops on a failed job and resumes when
rerun after resolving it. Measurements can take substantially longer than clean
inference because attention reconstruction includes all queries/keys/heads.
`analysis/attention_report.md`, `attention_summary.json` and three per-layer
attention figures supplement the drift figures.

## Verification and references

```bash
bash scripts/test.sh
```

Tests cover complete long-wave loading, header-only inspection without size
cutoffs, lossless sample coverage across windows, exact token estimates, native
DIFFA encoder/connector processing with small random weights, interleaved group
membership, native FPD, cache reference equivalence and reports. Full-checkpoint
GPU validation runs separately on Vast.

- [DIFFA-2 paper](https://arxiv.org/html/2601.23161) and [official code](https://github.com/NKU-HLT/DIFFA/tree/0a87594e110984fbaf00ebf5bfe5840c212e4438/DIFFA-2).
- [VidLaDA paper](https://arxiv.org/html/2601.17868v2), Figures 4–5 and §4.1.
- [MMAU-Pro dataset](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro).

Audio, checkpoints, local caches and outputs remain excluded from Git.


## Answer hidden-state drift

New drift runs also record post-block hidden-state drift for the generated answer.
This is separate from attention and from the unchanged fixed-prefix drift file.
`answer_drift.jsonl` compares the final visible answer span (including reasoning,
excluding first EOS and later canvas positions) against previous/initial forwards.
Stable MASK and stable decoded controls require identical input IDs in the two
compared forwards. Changed-ID positions and all content positions are reported
separately; their change includes token replacement. Empty groups are omitted.
The final span is retrospective and cannot serve as an online cache rule.

The recorder observes the native decoder without changing its decisions. Clean
cache timing still has no observation hooks. `--no-answer-drift` disables these
additional diagnostics. Final reports include the offline interactive
`analysis/answer_drift_by_forward.html` with per-forward/five-forward views,
layer controls, token counts and CSV export. No Plotly installation is required.
Rebuild that HTML for a completed or partial run without loading a model with
`python -m drift_study.answer_report runs/mmau50_long_answer_drift`.

Earlier MMAU downloads did not save answer hidden states, so their answer drift
cannot be recovered offline. To measure it, use a **new checkout/output folder**
with the same existing manifest, e.g. `RUN_NAME=runs/mmau50_long_answer_drift`
with `scripts/run.sh`. Do not update source inside an active/resumable run.
The older GSM drift files already include stable masked/decoded/changed answer
positions, but use the full512-slot canvas and a different decoder; comparisons
with new MMAU visible-span curves are descriptive.
