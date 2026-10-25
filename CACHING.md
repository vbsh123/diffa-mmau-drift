# Official-FPD layerwise cache benchmark

Default: 50 questions, official confidence-adaptive FPD, up to128 answer tokens,
zero-based forwards0–9 dense, first reuse at forward10. Each generation starts
with an empty cache. Fixed start needs no previous drift run or mean estimate.

Intervals below are actual decoder forwards, ordered **acoustic / fixed text /
semantic**. All32 layers are cached by default. Only semantic refresh changes in
blocks24–31; it is twice as frequent there.

| Policy | Blocks0–23 | Blocks24–31 |
|---|---|---|
| 161616_deepsem2 | 16 /16 /16 | 16 /16 /8 |
| 888_deepsem2 | 8 /8 /8 | 8 /8 /4 |
| 16816_deepsem2 | 16 /8 /16 | 16 /8 /8 |
| 16168_deepsem2 | 16 /16 /8 | 16 /16 /4 |
| 323216_deepsem2 | 32 /32 /16 | 32 /32 /8 |
| 321632_deepsem2 | 32 /16 /32 | 32 /16 /16 |

The dense anchor is forward9. Interval8 refreshes at17,25,…; interval4 at13,17,….
All answer rows recompute every forward. All key/value columns remain present,
with original RoPE positions. Prefix Q/K/V projections, attention queries/output
and MLP rows are physically skipped on reuse steps. Post-RoPE K/V and block outputs
are reused. This is approximate bidirectional caching without sparse attention.

`CACHE_LAYERS=16-31` restricts the cached blocks. Mixed lists/ranges work too.
`DEEP_SEMANTIC_LAYERS=24-31` and `DEEP_SEMANTIC_FACTOR=2` select the faster semantic
refresh profile. `DEEP_SEMANTIC_FACTOR=1` restores the flat schedules.
`MATCHED_FLAT=1` adds all six flat schedules at the same forward10 start, giving
650 timed generations instead of350. This is the controlled way to isolate the
layer-specific change from the earlier forward57 start.

## One command for all three runs

Use the command in [README.md](README.md#one-command-for-all-three-runs).
`scripts/next.sh` runs forward10 layerwise caching on the original frozen MMAU50,
then on a new50 long-audio cohort, then a separate long drift/attention run.
Both cache runs have clean baselines. All three jobs run sequentially on the GPU,
and their result folders are packaged together.

The original manifest defaults to `/workspace/diffa-mmau-drift/data/mmau50.json`.
Set `ORIGINAL_MANIFEST` if it is elsewhere; its WAVs must still exist. Original
cohort reuse is explicitly enabled only for the first job. No short sampler or
new duration cap is introduced. The second/third jobs use `LONG_MANIFEST`, default
`data/mmau50_long.json`, with complete clips longer than60s and no fixed maximum.
This new cohort is also MMAU-Pro. All50 IDs, WAV hashes and prompts are frozen.
Acoustic tokens are64 per encoder window; all projected tokens are preserved.

For standalone caching on an existing short manifest, use `MANIFEST=...` and
`ALLOW_LEGACY_SHORT_COHORT=1` with `scripts/cache.sh`. The standard cache command
prepares/uses the new long cohort.

To run just caching: `bash scripts/cache.sh` after setup. Defaults match the table.
`SAMPLES=3 RUN_NAME=runs/cache_smoke3` gives a smoke run from the frozen50 cohort.
Preflight without a GPU/model load:

```bash
python -m cache_study.run --manifest data/mmau50_long.json --preflight-only
```

The older `CACHE_START=mean`, `half-mean` and `immediate` modes remain available
through `scripts/cache.sh` with `BASELINE_RUN` pointing to a complete matching
50-question drift run. Historical-source validation is strict: changed source,
cohort, prompts, models or decoder settings are rejected. Fixed start avoids this
dependency. Old short-input source is preserved on branch `short-audio-v1`.

## Timing and verification

Each question has one fresh baseline and six policy generations, in deterministically
shuffled order. Process warmups are excluded. First two questions additionally
require exact canvas IDs/NFE from refresh-every-forward identity caching.
No observation probes or scoring/file I/O enter decode timers. Cache setup and GPU
synchronization are included. A1024-forward watchdog records failures without
forcing commits. Failures remain incorrect in quality results.

Outputs retain final texts/IDs, strict grades, EOS/capacity flags, forward counts,
seconds, actual reuse forwards, per-layer refresh counts and computed rows.
Each completed condition saves immediately and resumes only under an identical
source/config/environment. New settings require a fresh output folder.
`analysis/report.md`, `summary.json` and `pairs.csv` update after each condition.
Speedup uses clean paired baseline/cache timings. Answer lengths/NFE may change,
so these are autonomous generation timings rather than equal-work replay.

CPU tests compare physical skipping against an independent dense stale-K/V/output
reference for all six layerwise schedules, delayed starts and BF16. Native FPD
identity output/NFE is checked. Full-checkpoint GPU quality/speed need the Vast run.
