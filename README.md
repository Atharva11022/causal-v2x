# Reliable Near-Miss Driving Scenario Generation Under Degraded V2X Communication Using Stability-Aware Causal Discovery

This repository implements a resource-conscious proof of concept for reliable safety-critical scenario generation from real cooperative vehicle-infrastructure trajectories. The project studies how causal relationships between interacting vehicles change when shared V2X context is delayed or dropped, and whether stability-aware causal masks improve the causal consistency of generated near-miss reaction states.

## Current status

The current results use the official **V2X-Seq-TFD example release** from AIR-THU, not the full TFD release. The example contains real cooperative trajectory CSVs with timestamps, persistent vehicle IDs, positions, headings, velocities, vehicle dimensions, and intersection identifiers.

Verified example run:

- 57 scenarios
- 49,826 trajectory rows
- 148 vehicles
- 20,696 interaction rows
- 1,628 derived near-miss events
- 777 valid five-step ego reaction windows per communication condition
- 117 training-only directed causal-edge stability records
- 6 clean cooperative stable edges at the selected stability threshold

Near-miss labels are derived from trajectory variables using closing-gap TTC and hard deceleration criteria. They are not pre-existing dataset labels. Communication degradation is simulated on cooperative context through controlled dropout and temporal delay; clean targets and event labels are preserved.

The V2X data itself is real cooperative data, not a proxy: the local files come from AIR-THU's official V2X-Seq-TFD vehicle-infrastructure cooperative-view trajectory release. The released trajectory CSVs contain cooperative tracks, not packet-level network logs. Therefore the project studies real V2X-derived observations under reproducible simulated communication degradation, rather than claiming measured wireless latency or packet-loss behavior.

## Research workflow

1. Load real V2X-Seq-TFD cooperative trajectories.
2. Build ego and surrounding-vehicle interaction features.
3. Derive near-miss reaction-state events.
4. Simulate clean, dropout, delay, and combined dropout-delay conditions.
5. Run scenario-level bootstrap PC causal discovery.
6. Estimate edge stability and threshold sensitivity.
7. Build condition-specific stability-aware causal masks.
8. Train masked CVAEs and a dense baseline.
9. Compare causal consistency, distributional error, diversity, and degradation behavior.

The project now includes a five-step ego reaction-trajectory generator for each near-miss context, predicting ego speed and acceleration at 0.1-second intervals. It is not yet a complete multi-agent trajectory or scene generator.

## Repository layout

| Path | Purpose |
|---|---|
| `src/00_check_setup.py` | Check Python, dependencies, and accelerator availability |
| `src/01_preprocess.py` | Load TFD trajectories, derive interactions and near-miss events, simulate degradation |
| `src/02_causal_discovery.py` | Scenario-bootstrap PC discovery, stability analysis, sensitivity analysis, and graph comparison |
| `src/03_generative_replay.py` | Train adaptive masked, robust masked, and dense CVAE variants |
| `src/04_evaluate.py` | Evaluate all communication conditions and generator variants |
| `src/05_robustness.py` | Run multi-seed and same-sparsity random-mask controls |
| `src/06_trajectory_generation.py` | Generate five-step ego reaction trajectories with physical-validity checks |
| `tests/test_pipeline.py` | Lightweight regression tests for the pipeline contracts |
| `outputs/` | Small plots and CSV evidence artifacts; models and NumPy arrays are ignored |
| `docs/` | Reports, presentation material, and the final abstract |
| `HANDOFF.md` | Full continuation context for another coding agent |

## Data setup

Raw datasets are intentionally excluded from GitHub. Download the official V2X-Seq-TFD example from the AIR-THU repository and extract it locally under:

```text
data/v2x_seq_tfd_example_extracted/V2X-Seq-TFD-Example/
```

The preprocessing script expects cooperative training trajectories at:

```text
cooperative-vehicle-infrastructure/vehicle-trajectories/train/data/*.csv
```

The official project and dataset references are:

- [AIR-THU/DAIR-V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq)
- [V2X-Seq project page](https://thudair.baai.ac.cn/index)
- [V2X-Seq paper](https://arxiv.org/abs/2305.05938)

## Environment and execution

The experiments were developed for a MacBook M2 with 8 GB RAM. The models are intentionally small and use MPS when available, with CPU fallback.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 src/00_check_setup.py
python3 src/01_preprocess.py
python3 src/02_causal_discovery.py
python3 src/03_generative_replay.py
python3 src/04_evaluate.py
python3 src/05_robustness.py
python3 src/06_trajectory_generation.py

python3 -m unittest discover -s tests -v
```

Run the scripts in numerical order. The generated CSV and PNG evidence files are reproducible from the local dataset; model checkpoints, NumPy arrays, raw data, extracted datasets, virtual environments, and caches are not tracked.

## Primary degraded-condition result

For the `dropout_20_delay_1` condition in the current example experiment:

| Generator | Consistency gap |
|---|---:|
| Adaptive stability-aware mask | 0.3070 |
| Robust all-condition mask | 0.3474 |
| Dense baseline | -0.0443 |

The consistency gap is the mean absolute correlation for graph-supported context-to-reaction links minus the corresponding correlation for unsupported links. Delay-2 is an adverse condition in the current example results and should be reported as such.

The primary robustness control uses three training seeds and a same-sparsity random mask. Mean consistency gaps are 0.3261 for the adaptive mask, 0.3261 for the robust mask, 0.2977 for the random mask, and 0.0238 for the dense baseline. The random-mask interval overlaps the causal-mask interval, so these results support structural selectivity without claiming unique causal attribution.

## Limitations

- Results currently use the official TFD example subset rather than the full TFD release.
- Near-miss labels are operational definitions derived from trajectories.
- Dropout and delay are controlled simulations, not measurements from a live V2X network.
- The generator produces reaction states rather than complete multi-agent trajectories.
- Results are preliminary example-data evidence and do not guarantee conference acceptance.

## Reproducibility and handoff

See [HANDOFF.md](HANDOFF.md) for the exact current state, verified commands, output contracts, methodological decisions, and remaining work. The report and presentation require synchronization with this implementation before final submission.

## License

MIT. See [LICENSE](LICENSE).
