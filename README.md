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
- 84 bootstrap causal-edge stability records
- 7 clean cooperative stable edges at the selected stability threshold

Near-miss labels are derived from trajectory variables using closing-gap TTC and hard deceleration criteria. They are not pre-existing dataset labels. Communication degradation is simulated on cooperative context through controlled dropout and temporal delay; clean targets and event labels are preserved.

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

The current generator produces two reaction-state variables, ego acceleration and ego speed. It does not generate complete multi-agent trajectory sequences.

## Repository layout

| Path | Purpose |
|---|---|
| `src/00_check_setup.py` | Check Python, dependencies, and accelerator availability |
| `src/01_preprocess.py` | Load TFD trajectories, derive interactions and near-miss events, simulate degradation |
| `src/02_causal_discovery.py` | Scenario-bootstrap PC discovery, stability analysis, sensitivity analysis, and graph comparison |
| `src/03_generative_replay.py` | Train adaptive masked, robust masked, and dense CVAE variants |
| `src/04_evaluate.py` | Evaluate all communication conditions and generator variants |
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

python3 -m unittest discover -s tests -v
```

Run the scripts in numerical order. The generated CSV and PNG evidence files are reproducible from the local dataset; model checkpoints, NumPy arrays, raw data, extracted datasets, virtual environments, and caches are not tracked.

## Primary degraded-condition result

For the `dropout_20_delay_1` condition in the current example experiment:

| Generator | Consistency gap |
|---|---:|
| Adaptive stability-aware mask | 0.3240 |
| Robust all-condition mask | 0.2146 |
| Dense baseline | 0.0295 |

The consistency gap is the mean absolute correlation for graph-supported context-to-reaction links minus the corresponding correlation for unsupported links. Delay-2 is an adverse condition in the current example results and should be reported as such.

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
