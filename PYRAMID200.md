# Exploratory short-audio depth-pyramid study

This run tests a full four-group depth schedule on the frozen short200 cohort.
It follows inspection of that cohort's previous results, so it is exploratory,
not a fresh confirmation set. Drift is a motivation, not an established predictor
of cache accuracy. Acoustic and semantic drift differ; both directions of a shared
audio/text ratio are tested rather than assuming either is correct.

Refresh intervals below are layer groups0–7,8–15,16–23,24–31. Larger intervals
mean less frequent refresh. A and S are the two audio groups; T is fixed text
context. Answer positions are updated every forward.

| Policy | A intervals | T intervals | S intervals |
|---|---|---|---|
| pyramid_equal | 64/32/16/8 | 64/32/16/8 | 64/32/16/8 |
| pyramid_audio_slow | 128/64/32/16 | 64/32/16/8 | 128/64/32/16 |
| pyramid_audio_fast | 32/16/8/4 | 64/32/16/8 | 32/16/8/4 |

Controls are dense, flat16/16/16, flat16/8/16, and8/8/8 with semantic intervals
halved only in24–31 (the latter two tied the previous dense accuracy). Every
condition is rerun on the same GPU; no old timings or predictions are substituted.
Seven conditions ×200 questions =1400 timed generations, plus untimed warmups and
two identity checks. Conditions are shuffled within each question.

All conditions retain official autonomous FPD,128 answer slots, block32,
temperature0, factor1, natural EOS, the existing reasoning prompt and strict scorer.
Forwards0–9 are dense; forward9 anchors the cache and first reuse is forward10.
An interval8 next refreshes at17,25,…; an interval64 refreshes at73,137,….
Within each modality, shallower intervals are integer multiples of deeper ones,
so a shallow refresh also refreshes that modality in every deeper layer group.
No sparse attention, anchor-token pruning or long-audio concatenation is added.

## Run on Vast

Run after the previous GPU job finishes. Existing200-question manifests are reused
and verified; otherwise deterministic short preparation runs before model loading.
No previous-machine50-question files are required. Complete clips are at most30s,
one encoder window each. Stereo, if present, uses the recorded mean-to-mono policy.

```bash
cd /workspace/diffa-mmau-200
git pull --ff-only
nohup env SKIP_INSTALL=1 bash scripts/pyramid200_short.sh > pyramid200_short.log 2>&1 &
tail -f pyramid200_short.log
```

`SKIP_INSTALL=1` reuses a working environment. Omit it on a fresh installation.
The default manifest is `data/mmau200_short_stereo_v2.json`; use
`SHORT200_MANIFEST` only to point to the same frozen cohort elsewhere.
Old experiment outputs are not changed. New source/config changes require a
separate output directory and are rejected on incompatible resumes.

Download **`runs/mmau200_short_pyramid_results.tar.gz`**. It includes the full
run, frozen cohort/config and schedules, individual predictions and timings,
identity checks, experimental plan, and offline paired HTML/CSV/JSON statistics.

## Fixed comparisons

Three exploratory primary contrasts, corrected together with Holm3:

1. Equal pyramid minus flat16/16/16.
2. Audio-slow pyramid minus equal pyramid.
3. Audio-fast pyramid minus equal pyramid.

The first compares complete policies: average refresh work is not exactly matched
(the equal pyramid averages a1/17.07 refresh rate per layer versus1/16 for flat16).
It does not isolate depth allocation independently of total refresh computation.
The next two hold text schedules fixed and change both audio groups together;
they do not isolate acoustic from semantic contributions.

Secondary: each of six cached conditions versus fresh dense, Holm6.
Reports include exact two-sided McNemar p-values, gains/losses, conservative95%
paired difference bounds, supplementary paired bootstrap intervals, TPS ratios
and paired latency speedups. Confidence intervals are unadjusted and do not prove
equivalence. Per-condition parsing failures and mean forward counts are in
`analysis/summary.json`; full outputs allow checking why answers changed.

Regenerate statistics without loading models:

```bash
python -m cache_study.paired runs/cache200_short_pyramid_forward10 \
  --suite pyramid --expected-samples 200 --out runs/mmau200_short_pyramid_statistics
```
