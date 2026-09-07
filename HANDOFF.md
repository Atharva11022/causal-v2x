# Project Handoff

## Non-negotiable project title

**Reliable Near-Miss Driving Scenario Generation Under Degraded V2X Communication Using Stability-Aware Causal Discovery**

The approved abstract is binding. Do not change the project direction, title, or scientific scope without explicit user approval.

## User constraints

- One-month publication-oriented execution.
- Target: Scopus-indexed conference paper and dissertation deliverables.
- Hardware: MacBook M2, 8 GB RAM, 512 GB storage.
- Preserve compute efficiency and avoid unnecessary downloads.
- GitHub Copilot credits are limited. Prefer local terminal execution, focused edits, and no repeated broad exploration.
- Documentation work is currently deferred. Continue code and workflow first, but documentation must eventually be completed for Phase I/II requirements.

## Approved scientific concept

The system must:

1. Use real cooperative vehicle-infrastructure trajectory data.
2. Construct ego and cooperative interaction context.
3. Derive near-miss events using trajectory safety measures.
4. Simulate degraded V2X communication through context dropout and delay.
5. Learn causal graphs under clean and degraded conditions.
6. Estimate causal-edge stability using resampling.
7. Use stable causal edges to build a stability-aware generator mask.
8. Generate near-miss reaction scenarios with a conditional variational autoencoder.
9. Compare the masked generator against a dense baseline.
10. Evaluate reliability using causal consistency, realism/distribution error, diversity, and degradation comparisons.

## Dataset state

Primary dataset: official AIR-THU **V2X-Seq-TFD** trajectory forecasting example.

Downloaded locally:

- `data/v2x_seq_tfd_example` — approximately 286 MB compressed.
- `data/v2x_seq_tfd_example_extracted/` — approximately 1.1 GB extracted.

The TFD example contains cooperative trajectory CSVs with timestamps, persistent IDs, positions, headings, velocities, dimensions, and intersection identifiers.

The full official TFD trajectory release could not be located in the current public Google Drive structure. The visible full folder exposed SPD archives rather than full TFD trajectory CSV archives. Do not claim full-dataset results. Current results are from the official TFD example subset.

## Current source files

- `src/01_preprocess.py`: loads TFD cooperative vehicle trajectories, computes speed/acceleration, finds plausible forward nearby vehicles, builds interaction features, labels clean near misses, and generates context-only dropout/delay variants.
- `src/02_causal_discovery.py`: scenario-level bootstrap PC causal discovery across seven degradation conditions; excludes deterministic TTC from graph nodes; produces threshold, alpha, and seed sensitivity outputs; produces ego-only and cooperative comparisons; saves robust edges stable across all conditions.
- `src/03_generative_replay.py`: trains condition-specific stability-masked CVAEs and dense baselines; trains an additional robust-all-condition mask model for `dropout_20_delay_1`; saves generated outputs and model checkpoints.
- `src/04_evaluate.py`: evaluates all conditions and the robust primary ablation using distribution mean MAE, diversity, linked correlation, unlinked correlation, and consistency gap.
- `tests/test_pipeline.py`: five lightweight regression tests for TFD loading, near-miss labeling, degradation semantics, causal stability, and primary evaluation artifacts.

## Verified data results

Latest preprocessing run:

- 49,826 trajectory rows.
- 57 scenarios.
- 148 vehicles.
- 20,696 plausible interaction rows.
- 1,628 clean near-miss events.
- Near-miss rate: 7.87% of interaction rows.

The preprocessing filters acceleration estimates using valid 0.05–0.2 second intervals and an absolute acceleration limit of 15 m/s^2. It restricts the forward interaction corridor using lateral distance and vehicle widths.

## Verified causal results

Seven degradation conditions:

- `clean`
- `dropout_10`
- `dropout_25`
- `dropout_50`
- `delay_1`
- `delay_2`
- `dropout_20_delay_1`

Latest standard causal output:

- 84 edge-stability records.
- Clean cooperative stable edges: 7.
- Robust edges stable across all seven conditions: 5 graph edges.
- The robust graph maps to 4 generator context-to-target links.

Threshold sensitivity is stored in `outputs/causal_threshold_sensitivity.csv` for thresholds 0.60, 0.70, and 0.80.

Ego-only alpha sensitivity is stored in `outputs/causal_ego_alpha_sensitivity.csv` for alpha values 0.01, 0.05, 0.10, and 0.20. The only ego-only candidate edge has stability 0.25, 0.40, 0.45, and 0.55 respectively, never reaching the selected 0.70 threshold. Therefore the latest ego-only stable-edge count is 0, while cooperative stable-edge count is 7. Treat this result carefully and report the sensitivity evidence rather than overclaiming.

Repeated bootstrap seed sensitivity is in `outputs/causal_seed_sensitivity.csv`:

- clean stable edges: 7, 7, 8 for seed bases 42, 142, 242.
- primary combined degradation stable edges: 7, 8, 6 for seed bases 42, 142, 242.

## Verified generator results

Primary degraded condition: `dropout_20_delay_1`.

Latest evaluation values:

- adaptive causal-masked consistency gap: 0.324039.
- robust-all-condition masked consistency gap: 0.214556.
- dense baseline consistency gap: 0.029534.

Both masked models outperform the dense baseline on the primary degraded condition. The adaptive mask is better than the stricter robust mask in the latest run.

Across the seven-condition matrix, the adaptive masked model outperformed the dense baseline on most conditions; delay-2 is an adverse result and must be reported honestly.

The generator currently produces two reaction-state variables: `ego_accel` and `ego_speed`. It does not generate complete multi-agent trajectory windows. The scientifically accurate wording is “near-miss reaction-state generation conditioned on cooperative context,” unless the model is later extended.

## Important methodological decisions

- V2X context is real cooperative trajectory data from V2X-Seq-TFD, not NGSIM.
- Communication degradation is simulated only on shared context; clean targets and near-miss labels are preserved.
- TTC is not included as a causal graph variable because it is deterministically derived from gap and relative speed.
- Bootstrap resampling is scenario-level to avoid treating correlated temporal frames as IID.
- Stability threshold is 0.70, with sensitivity analysis.
- Scenario-level train/validation split is used for generator training.

## Existing outputs

Important files in `outputs/`:

- `causal_edge_stability.csv`
- `causal_edges_v2x.csv`
- `causal_edges_ego_only.csv`
- `causal_edges_robust.csv`
- `causal_stability_summary.csv`
- `causal_threshold_sensitivity.csv`
- `causal_ego_alpha_sensitivity.csv`
- `causal_seed_sensitivity.csv`
- `causal_context_comparison.csv`
- `evaluation_summary.csv`
- `performance_metrics.csv`
- causal graph and consistency PNG files
- CVAE checkpoints and generated `.npy` files

## Validation already completed

- `python3 src/01_preprocess.py` completed successfully on the TFD example.
- `python3 src/02_causal_discovery.py` completed successfully.
- `python3 src/03_generative_replay.py` completed successfully for all seven conditions and the robust primary ablation.
- `python3 src/04_evaluate.py` completed successfully.
- `python3 -m py_compile` passed for the source scripts.
- `python3 -m unittest discover -s tests -v` passed all 5 tests.
- VS Code diagnostics were clean for the current causal, generator, and evaluation scripts after import formatting.

## What remains, in priority order

### Code/workflow priority

1. Perform one clean end-to-end run from the current source and save the final results checkpoint.
2. Add or verify final test cases for output-file contracts and reproducibility.
3. Decide whether reaction-state generation is sufficient for the paper or whether a short trajectory-window generator is feasible. Do not expand scope casually; hardware and deadline favor retaining reaction-state generation with precise wording.
4. Add final robustness metrics if inexpensive, such as per-seed generator variation.
5. Verify the full project from a fresh virtual environment if time permits.

### Documentation priority, currently deferred by user

1. Rewrite `docs/report.md` to match V2X-Seq-TFD and current implementation; remove outdated NGSIM claims.
2. Create requirements specification.
3. Create high-level architecture/design document.
4. Create test plan and test-results artifact.
5. Update `README.md`.
6. Check/update `docs/presentation.pptx`.
7. Prepare final conference paper and dissertation report.
8. Include limitations: official TFD example subset, derived near-miss labels, simulated communication degradation, reaction-state rather than full trajectory generation, and no guarantee of Scopus acceptance.

## How to continue

Activate environment:

```bash
source /Users/atharva/causal-v2x-project/venv/bin/activate
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Run the core workflow:

```bash
python3 src/01_preprocess.py
python3 src/02_causal_discovery.py
python3 src/03_generative_replay.py
python3 src/04_evaluate.py
```

Do not download SPD archives or make claims about full TFD results unless the full TFD trajectory release is actually obtained and processed.
