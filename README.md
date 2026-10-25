# DIFFA-2: MMAU-Pro drift with official FPD

Fresh 50-example observation study of **semantic audio, acoustic audio, and fixed
input text**. Uses the pinned official DIFFA-2 factor-based parallel decoder
directly. **128 maximum answer tokens**, 32-token blocks, factor=1.0, temperature=0.
The prompt requests brief step-by-step audio-grounded reasoning and ends with
`Final answer: <letter>`.

FPD determines the commit count from confidence at each forward. There is no
fixed commit quota, fixed confidence threshold, forced answer length, temporal
KV/hidden reuse, pruning, sparse attention, or injected reasoning tags. The 128
slots are a maximum canvas, not a requirement to generate 128 tokens. Official
EOS handling stops once there are no unresolved masks before EOS. Capacity
exhaustion and incomplete final-answer formats remain visible in results.

The official generator is imported from an unchanged checkout, not reimplemented.
A read-only wrapper logs its original transfer decisions and returns its exact
tensors. Block hooks observe all 32 block outputs. A watchdog aborts pathological
runs after 1024 forwards instead of forcing tokens or silently modifying decoding.

## Vast: one setup-and-run command

Use a **single A40 48GB** (or another BF16 CUDA GPU with enough VRAM), Python
3.10–3.12, a CUDA 12.4-compatible driver, and approximately **100GB free disk**
for the Python environment, model cache, and temporary activations. Full model
inference has not been run locally; GPU runtime is not yet measured.

```bash
cd /workspace
git clone https://github.com/vbsh123/diffa-mmau-drift.git
cd diffa-mmau-drift
nohup bash scripts/run.sh > mmau50_fpd128.log 2>&1 &
tail -f mmau50_fpd128.log
```

This installs pinned dependencies, fetches and verifies official DIFFA source,
prepares the frozen cohort, loads models once, verifies the first two instrumented
decodes against untouched official FPD (exact complete IDs and forward count),
measures all 50, and writes reports and PNG figures. Closing `tail` with Ctrl-C
does not stop the job. Run only one experiment per GPU.

Setup without inference:

```bash
bash scripts/setup.sh
```

Optional three-example GPU check (one per domain, drawn from the same 50):

```bash
nohup bash scripts/smoke.sh > smoke3_fpd128.log 2>&1 &
tail -f smoke3_fpd128.log
```

Then run the full study, reusing the installed environment and downloaded inputs:

```bash
nohup env SKIP_INSTALL=1 bash scripts/run.sh > mmau50_fpd128.log 2>&1 &
```

An unchanged full run resumes completed examples after validating artifact hashes.
Source/environment/config changes require a new `RUN_NAME`. Incomplete examples
restart; historical experiments are never imported. Do not update the checkout
while a process is running. `AUDIO_VENV` and `DIFFA_ROOT` can reuse compatible
installations. The smoke run and full run have separate output roots.

## Cohort

- Pinned `gamma-lab-umd/MMAU-Pro` revision
  `b8516c217464d53be36d6dc334fda3cd578ed103`.
- Seed 20261025; 17 speech, 17 sound, 16 music; interleaved domains.
- Published `short` category, single audio, MCQs with 2–26 distinct choices,
  exactly one matching reference. Full audio must be **mono and <=30 seconds**.
- No correctness-based selection, cropping, stereo downmixing, multi-audio
  concatenation, or silent replacement of an insufficient domain pool.
- Candidates are sorted by ID, shuffled deterministically per domain, and
  admitted by full-file audio headers. Exclusions, actual duration, option order,
  hashes, prompts, source revision, and bytes fetched are saved in the manifest.
- Metadata is ~1MB. HTTP Range reads fetch central-directory metadata and individual
  WAV members from the 47GB archive; the archive is **not downloaded in full**.
  Servers that ignore range requests cause failure before reading their payload.
  Non-audio metadata, files over 24MiB, and non-mono/duration failures are excluded.

This is a **short, mono, single-audio MCQ subset**, not a representative evaluation
of all MMAU-Pro or a long-audio scaling experiment. DIFFA's existing single-clip
encoder is retained. All admitted examples, including wrong answers, enter drift
statistics. Generated reasoning is requested but is not separately graded.

## Drift measurement

For group g at block l and forwards t,s, the primary metric is the average over
fixed token positions of `1 - cosine(h[t,l,i], h[s,l,i])`, calculated in FP32.
We additionally save cosine distance of the flattened group, token median/p90/max,
and zero-norm counts. Two zero vectors have similarity 1; one zero vector has 0.

- Main hidden-state measurements: **all 32 transformer block outputs**, every
  actual forward, against the previous forward and initial forward. Block 0 is
  the first transformer output, not the embedding layer. No layer is omitted.
- Semantic and acoustic memberships come from actual inserted speech positions.
  All fixed text, including system/user instructions, choices and chat delimiters,
  enters the text group. Generated/MASK answer positions and padding are excluded.
- Acoustic tokens are the connector's actual count (normally 64). Semantic and
  text counts are measured independently for every example.
- Full cross-step cosine matrices: first admitted example of each domain,
  predetermined before inference, at blocks **0,7,15,23,31**. These selected
  matrices supplement the all-layer measurements; they are not all-layer matrices.
  BF16 states are stored losslessly in temporary disk scratch. Only small token
  chunks are expanded to FP32 during pairwise analysis, then scratch is deleted.
- No K/V drift or attention probes in this first experiment: hidden states match
  the quantity used in VidLaDA's diagnostic figures.

Examples have equal weight: average valid consecutive-forward comparisons within
each example/layer, then layers and examples. Single-forward decodes have no
adjacent comparison and are counted explicitly. Domain summaries, per-layer
means, and paired example-bootstrap intervals for audio-minus-text are saved.
Raw forward-index heatmaps include contributing-example counts. Separately,
normalized-progress heatmaps interpolate within each example's observed range;
they do not pad ended trajectories with zeros. Correct-only filtering is not used.

VidLaDA Figures 4–5 show hidden-state cosine matrices and `1-cosine` drift across
layers/steps. The released repository does not include a diagnostic script
specifying its activation hook, token reduction, or reference step. Our precise
choices are documented above; this is a transparent adaptation of the published
analysis, not a claim of identical undocumented implementation. VidLaDA's 128-token
controlled ablations also used fixed schedules; **we retain DIFFA FPD instead**.

## Outputs

`runs/mmau50_fpd128/` contains:

- `config.json`: exact decoder, model/source/input revisions, GPU/environment.
- `model_load.json`: checkpoint/LoRA loading checks.
- `samples/00000/result.json`: output IDs/text, final-letter grade, natural EOS,
  capacity flag, group sizes, verification outcome, artifact hashes.
- `samples/00000/trace.jsonl`: actual commits, positions, IDs, eligible masks and
  block for every official FPD forward.
- `samples/00000/drift.jsonl` and `measurement.json`: all-layer/group drift and
  layer/reference conventions.
- `samples/00000/pairwise.npz`: exemplar step-pair matrices where selected.
- `analysis/report.md`, `summary.json`: refreshed after every completed example.
- `analysis/*drift.png`, `*contributors.png`, `pairwise_*.png`, and heatmap NPZ
  arrays with contributor counts: generated after the full collection.

All instrumented wall times include observation overhead and **must not be used
as clean inference throughput**. There are no speedup claims in this pilot.

Rebuild figures later without loading models:

```bash
source .venv/bin/activate
python -m drift_study.report runs/mmau50_fpd128 --plots
```

Package completed outputs for download:

```bash
tar -czf mmau50_fpd128.tar.gz -C runs mmau50_fpd128
```

## Verification

```bash
bash scripts/test.sh
```

CPU tests exercise the real pinned LLaDA implementation and official FPD function
with a small random model, exact observation parity, group positions, metric and
pairwise reductions, selective archive reads, deterministic sampling, and report
coverage. The first two full-model GPU examples perform the separate exact-ID/NFE
check at runtime. Tests do not establish full-checkpoint GPU performance.

## References

- [DIFFA-2 paper](https://arxiv.org/html/2601.23161), especially Appendix B.4.
- [Pinned official DIFFA code](https://github.com/NKU-HLT/DIFFA/tree/0a87594e110984fbaf00ebf5bfe5840c212e4438/DIFFA-2).
- [VidLaDA paper](https://arxiv.org/html/2601.17868v2), Figures 4–5, §4.1, Appendix G.5.
- [MMAU-Pro dataset](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro).

Dataset audio, model weights, local caches and run outputs are excluded from Git.
