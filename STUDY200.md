# 200-question matched caching study

## Short-only run (recommended current experiment)

Long-audio concatenation across independent encoder windows is an unvalidated
input extension. Use the short-only launcher to avoid that confound. It selects
200 complete clips at most30 seconds, so every example uses one encoder window.
All13 conditions remain matched: dense baseline, six flat policies and six
deep-semantic halvings, starting reuse at zero-based forward10. No old files are
required. Prior50 overlap is unchecked. Total:2600 timed generations.

```bash
cd /workspace/diffa-mmau-200
git pull --ff-only
nohup env SKIP_INSTALL=1 bash scripts/study200_short.sh > study200_short.log 2>&1 &
tail -f study200_short.log
```

Use `SKIP_INSTALL=1` only with a working installation; otherwise omit it.
Stop an active two-cohort job before launching this one on the same GPU.
Outputs are separate from the two-cohort experiment. Download
`runs/mmau200_short_only_results.tar.gz`; it includes the frozen plan, all
per-question results, and HTML/CSV/JSON paired statistics. Holm families contain
six halved-vs-flat and twelve cache-vs-dense comparisons for this single cohort.

## Original two-cohort protocol (long input extension unvalidated)

Two frozen cohorts,200 questions each. All13 conditions run on exactly the same
questions within each cohort: one fresh dense baseline, six flat refresh schedules,
and the same six with semantic intervals halved only in blocks24–31.

| Base A / text / semantic | Flat blocks24–31 | Halved blocks24–31 |
|---|---|---|
| 16 /16 /16 | 16 /16 /16 | 16 /16 /8 |
| 8 /8 /8 | 8 /8 /8 | 8 /8 /4 |
| 16 /8 /16 | 16 /8 /16 | 16 /8 /8 |
| 16 /16 /8 | 16 /16 /8 | 16 /16 /4 |
| 32 /32 /16 | 32 /32 /16 | 32 /32 /8 |
| 32 /16 /32 | 32 /16 /32 | 32 /16 /16 |

Blocks0–23 use the base schedule in both variants. All32 blocks are cached.
Zero-based forwards0–9 are dense; first reuse is forward10. Official FPD, reasoning
prompt,128 maximum answer slots, natural EOS, models and strict scoring are unchanged.
Every generation starts with empty caches. Per-question condition order is shuffled.
Clean timers include cache management and GPU synchronization; no drift/attention
probes or scoring/file I/O enter those timers. Warmups and identity checks are extra.

## Cohort rules fixed before outcomes

- Short arm: complete clips with0<duration<=30s, matching the original short arm.
- Long arm: complete clips>60s, **no fixed maximum duration**. Reject whole clips
  only for documented eligibility or the pinned model's4096-position context budget;
  never crop to fit. Encoder windows cover every input sample.
- Seed20261027, quotas67 speech/67 sound/66 music in each arm. No selection by
  model correctness or generated answer quality.
- Old development manifests are optional. If supplied, exclude their question IDs,
  source audio members and WAV hashes. Without them, previous-cohort overlap is
  **unchecked**; do not describe the new cohorts as confirmed fresh holdouts.
- Mono and stereo WAVs are eligible. Stereo is averaged across its two channels
  before resampling, preserving every time sample. Mono samples are unchanged.
  This channel policy is recorded in each manifest and the frozen plan. The old
  mono-only rule excluded421 of446 long sound candidates.
- One selected question per unique audio hash. Both200-question manifests are
  frozen and checked before GPU inference. If enough unique eligible audio is not
  available for the quotas, preparation stops with diagnostics; it never silently
  shrinks the cohort, repeats audio or substitutes short clips into the long arm.
- Short/long questions differ, so comparing arms remains descriptive rather than
  an isolated causal length intervention. The flat/halved contrast within an arm
  is matched by question, prompt, model and decoding settings.

## Vast command

On a new Vast machine, no old manifests or WAV files are required. The default
command installs dependencies and downloads the pinned source and datasets:

```bash
cd /workspace
# If this checkout already exists, enter it and run git pull --ff-only instead.
git clone https://github.com/vbsh123/diffa-mmau-drift.git diffa-mmau-200
cd diffa-mmau-200
nohup bash scripts/study200.sh > study200.log 2>&1 &
tail -f study200.log
```

If the existing installation in this checkout already works, use
`nohup env SKIP_INSTALL=1 bash scripts/study200.sh > study200.log 2>&1 &`.
`SKIP_INSTALL=1` skips package installation; it does not set up missing dependencies.
`AUDIO_VENV` can point to an existing working environment elsewhere.
Unset stale `ORIGINAL_MANIFEST` and `LONG_MANIFEST` variables on a fresh machine.
Only supply those variables if the corresponding old manifests **and their WAVs**
exist and you want verified old-cohort exclusions.

The revised launcher writes new `mmau200_short_stereo_v2.json` and
`mmau200_long_stereo_v2.json` manifests, avoiding the previous mono-only manifests.
Both cohorts finish preparation before model loading. Insufficient sampling writes
`*.sampling_failure.json` with all rejection counts. Completed GPU outputs remain
immutable; source/config changes require separate output directories.
The intended run has **5200 timed generations**, not200 total generations.

Outputs:

- `runs/cache200_short_forward10_paired`
- `runs/cache200_long_forward10_paired`
- `runs/mmau200_paired_statistics/paired_statistics.html` (offline)
- `runs/mmau200_experimental_plan.json` (frozen before inference)
- Download `runs/mmau200_matched_results.tar.gz` after completion.

## Paired statistics

Primary: halved minus its matching flat policy,200 common questions per comparison.
There are12 primary tests across both arms, corrected together with Holm.
Secondary: each cached policy minus its own clean dense baseline,24 comparisons,
with a separate Holm family. Use exact two-sided McNemar tests; count failures as
incorrect. No post-hoc one-sided test or equivalence/noninferiority claim is made.

Report gains, losses, the complete2×2 contingency table, accuracy difference,
exact p-values and adjusted p-values. For unique clips, displayed95% difference
intervals conservatively subtract97.5% exact binomial gain/loss intervals; this
avoids a misleading zero-width interval when no discordant pairs are observed.
Also retain a20000-draw paired audio-cluster bootstrap as a sensitivity check.
If clips repeat in manually supplied data, show the approximate cluster interval
and an exact clip-level paired label-swap test. Intervals are not adjusted across
policies and are not inversions of the exact test. Nonsignificance does not prove
safety.200 improves precision, but power still depends on the number of discordant
pairs and the size of the actual policy effect.

Quality statistics include all200 questions, including parsing failures. Saved
reports expose parse failures and generation lengths. TPS and paired latency
speedups remain distinct: autonomous output lengths/forward counts can change.
The paired report includes both, relative to the actual matched reference.

Final analysis refuses incomplete or mismatched cohorts/profiles, checks all
per-layer schedules, and checks common model/decoder/source/seed settings. Rebuild
without model loading after all conditions are complete:

```bash
python -m cache_study.paired \
  runs/cache200_short_forward10_paired \
  runs/cache200_long_forward10_paired \
  --out runs/mmau200_paired_statistics --expected-samples 200
```

The200-question timing experiment deliberately retains the existing layer policy
outside the requested semantic halving. The old shallow-layer ranking is a
repeatable but very small cosine-distance difference; it is not evidence that
shallow caching is harmless or important. A separate intervention would be needed
to identify the effect of caching only layers0–7.
