# Causal Discovery from Naturalistic Driving Data with V2X Cooperative Perception and Generative Replay of Near-Miss Scenarios

A causal-discovery + generative-modeling pipeline that:
1. Learns causal graphs of driver behavior from naturalistic trajectory data, comparing an **ego-only** view against an **ego + V2X-shared-context** view.
2. Quantifies how many causal links are only recoverable once cooperative (V2X-style) observations are available.
3. Trains a **causal-graph-constrained generative model** to synthesize near-miss driving scenarios, and evaluates it against an unconstrained baseline.

> Solo project, 3-credit academic project, built end-to-end (data pipeline → causal discovery → generative modeling → evaluation) on a local machine (Apple M2, 8GB RAM) using PyTorch with MPS acceleration.

## Motivation

Published causal-discovery methods on driving data are typically trained on ego-vehicle-only observations, missing interactions that happen outside a single vehicle's sensor range. Meanwhile, near-miss scenario generators are scored on realism and diversity, not on whether generated agent reactions are actually caused by the agents they're reacting to. This project connects the two: use V2X-style multi-agent context to recover a richer causal graph, then use that graph to constrain what a near-miss generator is allowed to produce.

## Dataset

**NGSIM (US-101)** — naturalistic highway vehicle trajectories, captured from synchronized overhead cameras that recorded all vehicles in the scene simultaneously. This "all-agents-visible" recording setup is used as a proxy for V2X-shared observation: ego-only features use just the subject vehicle's own kinematics, while the V2X-augmented feature set adds the preceding/neighboring vehicle's kinematics — information a single vehicle wouldn't have without cooperative sharing.

*(HighD was the originally targeted dataset per the project's literature review, but access required manual approval via LevelXData with no response after two requests; NGSIM was substituted as a freely-accessible dataset with equivalent structure.)*

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Setup check | `src/00_check_setup.py` | Verifies environment + MPS availability |
| Preprocessing & near-miss mining | `src/01_preprocess.py` | `data/ego_features.csv`, `data/v2x_features.csv`, `data/near_miss_events.csv` |
| Causal discovery | `src/02_causal_discovery.py` | `outputs/causal_graph_comparison.png`, `outputs/causal_edges_v2x.csv` |
| Generative replay | `src/03_generative_replay.py` | Trained causal-masked & baseline CVAEs, generated samples |
| Evaluation | `src/04_evaluate.py` | `outputs/evaluation_summary.csv`, `outputs/causal_consistency_comparison.png` |

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/00_check_setup.py

# Download NGSIM via Kaggle API into data/ (see docs/report.md for setup)

python src/01_preprocess.py
python src/02_causal_discovery.py
python src/03_generative_replay.py
python src/04_evaluate.py
```

## Results

<!-- Fill in once Phase 4 (causal discovery) and Phase 6 (evaluation) have run -->

**Causal graph comparison:**
`outputs/causal_graph_comparison.png`
- Ego-only graph: _N_ edges
- Ego + V2X-proxy graph: _N_ edges
- New causal links recovered with V2X context: _N_

**Generative replay evaluation:**
`outputs/causal_consistency_comparison.png`
- Causal-masked model consistency gap: _value_
- Baseline model consistency gap: _value_

## Limitations & future work

- V2X context here is a proxy (derived from a fixed-camera dataset that already sees all agents), not real multi-sensor V2X data. A natural extension is validating against genuine cooperative-perception datasets (e.g. DAIR-V2X, TUMTraf-V2X).
- The generative model operates on aggregated/snapshot feature vectors rather than full trajectory sequences; a sequence model (LSTM/Transformer-based CVAE, or diffusion) would be a stronger next step given more compute/time.

## Repo structure

```
├── src/            pipeline scripts (run in numeric order)
├── data/           raw + processed data (gitignored, regenerate via src/01_preprocess.py)
├── outputs/        plots, edge lists, evaluation results
└── docs/           written report and presentation
```

## License

MIT — see [LICENSE](LICENSE).
