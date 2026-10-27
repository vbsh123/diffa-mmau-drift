# Fresh200-question matched confirmation study

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
- Exclude all old question IDs, source audio members and WAV hashes from both
  prior50-question development manifests.
- One selected question per unique audio hash. Both200-question manifests are
  frozen and checked before GPU inference. If enough unique eligible audio is not
  available for the quotas, preparation stops with diagnostics; it never silently
  shrinks the cohort, repeats audio or substitutes short clips into the long arm.
- Short/long questions differ, so comparing arms remains descriptive rather than
  an isolated causal length intervention. The flat/halved contrast within an arm
  is matched by question, prompt, model and decoding settings.

## Vast command

Use a new checkout and keep the old manifests plus their WAVs available for
exclusion checks. Start after the current GPU measurement job finishes.

```bash
cd /workspace
git clone https://github.com/vbsh123/diffa-mmau-drift.git diffa-mmau-200
cd diffa-mmau-200
nohup env SKIP_INSTALL=1 \
  AUDIO_VENV=/workspace/diffa-mmau-drift/.venv \
  DIFFA_ROOT=/workspace/diffa-mmau-drift/external/DIFFA \
  ORIGINAL_MANIFEST=/workspace/diffa-mmau-drift/data/mmau50.json \
  LONG_MANIFEST=/workspace/diffa-mmau-next/data/mmau50_long.json \
  bash scripts/study200.sh > study200.log 2>&1 &
tail -f study200.log
```

On a fresh machine omit `SKIP_INSTALL` and `AUDIO_VENV` to install dependencies.
Adjust original manifest paths if the earlier checkout lives elsewhere. No old
inference source or results are modified. Source/config changes require a fresh
output root; completed conditions otherwise resume without repeating their timers.
Cohort preparation may take time and completes for both arms before model loading.
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
