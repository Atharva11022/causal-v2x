# Reliable Near-Miss Driving Scenario Generation Under Degraded V2X Communication Using Stability-Aware Causal Discovery

**Atharva Abhyankar**
MIS: 712552009 - Data Science

## Abstract

This project studies reliable near-miss driving scenario generation when cooperative V2X context is incomplete or delayed. It uses the official AIR-THU V2X-Seq-TFD example release, containing real vehicle-infrastructure cooperative trajectory data from 57 scenarios. Interaction features are derived from tracked vehicle motion, near-miss events are operationally identified using closing-gap TTC and hard deceleration, and shared context is degraded through timestamp-aware dropout and delay simulations. Scenario-level bootstrap PC discovery is performed on training scenarios only, and only identifiable directed arrows are retained for stability-aware masks. The project includes a five-step ego reaction-trajectory generator that predicts speed and acceleration at 0.1-second intervals. In the primary `dropout_20_delay_1` condition, the adaptive directed mask achieves a trajectory mean MAE of 0.2785 compared with 1.2309 for the dense baseline. The adaptive mask has zero negative-speed and acceleration-limit violations in the generated samples. These results support stability-aware directed masking for ego reaction-trajectory generation under simulated degraded cooperative context, while the example-subset scope, operational labels, simulated communication degradation, and lack of full multi-agent scene generation remain important limitations.

## 1. Introduction

V2X communication can provide an ego vehicle with cooperative observations that are unavailable from local sensing. These observations may be delayed, incomplete, or unavailable during safety-critical interactions. A generator trained only under complete cooperative context may learn relationships that do not remain reliable under degraded communication.

This project investigates whether identifiable causal structure that is stable across resampling and communication conditions can constrain near-miss generation. The study combines real cooperative V2X-derived trajectories, operational near-miss extraction, controlled communication ablations, directed causal-edge stability, and conditional generative models.

The contribution is intentionally scoped. The primary sequence model generates five-step ego reaction trajectories, not complete multi-agent scenes. The learned graph is observational evidence and is not treated as proof of physical intervention effects.

## 2. Objectives

1. Build ego and cooperative interaction features from real V2X-Seq-TFD trajectories.
2. Derive reproducible near-miss events from trajectory safety measures.
3. Simulate controlled context dropout and timestamp-aware delay.
4. Estimate identifiable directed causal-edge stability using scenario bootstrap resampling.
5. Transfer stable directed context-to-target links into generative masks.
6. Generate five-step ego speed and acceleration reaction trajectories.
7. Compare adaptive masks, robust masks, dense baselines, and same-sparsity random controls.
8. Evaluate distributional fidelity, physical validity, structural selectivity, diversity, and degradation behavior.

## 3. Dataset and Scope

The primary data source is the official **V2X-Seq-TFD example release** from AIR-THU. The local example contains vehicle-infrastructure cooperative trajectory CSV files with timestamps, persistent vehicle identifiers, positions, headings, velocities, vehicle dimensions, and intersection identifiers.

This is genuine cooperative V2X-derived data, not proxy V2X or NGSIM. The local trajectory CSVs do not contain packet identifiers, measured wireless latency, or packet-loss logs. Therefore the study uses real cooperative observations but evaluates communication degradation through reproducible counterfactual dropout and delay simulations rather than measured wireless-network traces.

The full official TFD trajectory release was not available locally. All results are therefore limited to the official example subset.

| Quantity | Value |
|---|---:|
| Trajectory rows | 49,826 |
| Scenarios | 57 |
| Vehicles | 148 |
| Plausible interaction rows | 20,696 |
| Derived clean near-miss events | 1,628 |
| Valid five-step reaction windows per condition | 777 |
| Near-miss rate among interaction rows | 7.87% |

## 4. Methodology

### 4.1 Interaction features

The preprocessing stage computes speed and acceleration from cooperative tracks. It identifies plausible forward vehicles using longitudinal and lateral geometry and constructs ego speed, ego acceleration, lead speed, lead acceleration, gap distance, relative speed, TTC, and observation availability.

Acceleration estimates use valid intervals between 0.05 and 0.2 seconds and an absolute acceleration limit of 15 m/s^2.

### 4.2 Near-miss events and reaction windows

Near-miss events are operational labels, not supplied ground-truth labels. An event is retained when closing-gap TTC is below 3 seconds or ego acceleration is below -3 m/s^2. TTC is used for labeling and excluded from causal graph variables because it is deterministically derived from gap and relative speed.

For each event anchor, the preprocessing stage searches the full interaction timeline for the same scenario, ego vehicle, and lead vehicle at the next five 0.1-second timestamps. Only complete windows are retained. Each window contains future ego speed and ego acceleration at five steps. Future target values remain clean while the anchor context is degraded.

### 4.3 Communication degradation

The evaluated conditions are:

- `clean`
- `dropout_10`
- `dropout_25`
- `dropout_50`
- `delay_1`
- `delay_2`
- `dropout_20_delay_1`

Delay uses timestamp-aware lookup constrained to the same scenario, ego vehicle, and lead vehicle. Missing or leader-switched history becomes missing context. Dropout and delay affect shared context only; clean targets and event labels are preserved. These are controlled communication-ablation scenarios, not measurements of a deployed V2X stack.

### 4.4 Directed causal stability

The PC algorithm is applied separately to each condition using generator training scenarios only. Scenario bootstrap resampling avoids treating complete scenarios as independent row-level samples. Only identifiable directed arrows in the PC CPDAG are retained. Unoriented adjacencies are excluded from causal masks.

At stability threshold 0.70, the final leakage-free directed analysis produced 117 edge-stability records, 6 clean stable directed edges, and 2 directed edges stable across all seven conditions. The robust directed mask contains two links: `gap_distance -> ego_accel` and `relative_speed -> ego_accel`.

The graph is observational. Directed CPDAG orientation is treated as identifiable graph evidence, not definitive physical causality.

### 4.5 Generative models

The project contains two related generation surfaces:

- The original reaction-state CVAE generates ego speed and acceleration states for the established scalar evaluation.
- `src/06_trajectory_generation.py` generates five-step ego reaction trajectories with speed and acceleration at 0.1-second intervals.

Both surfaces compare dense and directed-mask models. The primary condition also includes a robust all-condition mask. The sequence generator expands each context-to-target mask link across all five future steps.

### 4.6 Evaluation

The evaluation reports:

- Mean absolute error of generated and real means.
- Joint energy distance for the generated and real reaction-state distributions.
- Hard-deceleration rate.
- Diversity standard deviation.
- Supported and unsupported context correlation.
- Consistency gap, defined as supported correlation minus unsupported correlation.
- Negative-speed rate.
- Acceleration-limit violation rate.
- Jerk-limit rate for generated reaction trajectories.

Lower energy distance and lower physical-violation rates are preferred. A larger consistency gap indicates stronger structural selectivity, but it does not prove causal identification.

All models use the same fixed latent bank for paired comparison. The robustness experiment uses seeds 42, 142, and 242, a dense baseline, an adaptive directed mask, a robust directed mask, and a same-sparsity random mask.

## 5. Results

### 5.1 Directed causal stability

The leakage-free directed analysis produced 117 candidate edge records:

| Condition | Stable directed edges |
|---|---:|
| `clean` | 6 |
| `dropout_10` | 6 |
| `dropout_25` | 5 |
| `dropout_50` | 4 |
| `delay_1` | 2 |
| `delay_2` | 3 |
| `dropout_20_delay_1` | 5 |

Seed sensitivity for clean stable edges was 6, 6, and 6. The primary combined condition produced 5, 3, and 5 stable edges across the three seed bases.

### 5.2 Primary scalar reaction-state result

For `dropout_20_delay_1`:

| Generator | Mean MAE | Joint energy distance | Hard-decel rate | Consistency gap |
|---|---:|---:|---:|---:|
| Adaptive directed mask | 0.6781 | 0.5691 | 0.1276 | **0.3070** |
| Robust directed mask | 1.0345 | 1.5542 | 0.1754 | **0.3474** |
| Dense baseline | 1.2677 | 1.1495 | 0.1604 | **-0.0443** |

The robust mask has the strongest structural consistency but worse joint distribution distance. This is a genuine multi-objective tradeoff.

### 5.3 Multi-seed robustness controls

Across seeds 42, 142, and 242, mean consistency gaps were 0.3261 for the adaptive mask, 0.3710 for the robust mask, 0.2977 for the same-sparsity random mask, and 0.0238 for the dense baseline. The random-mask interval overlaps the directed causal-mask interval. The results support sparse structural selectivity but do not prove that the discovered edges uniquely cause the full gain.

### 5.4 Five-step reaction-trajectory generation

The sequence generator uses 777 valid windows per condition. In the primary `dropout_20_delay_1` condition:

| Generator | Mean trajectory MAE | Negative-speed rate | Acceleration-limit violation | Jerk-limit rate |
|---|---:|---:|---:|---:|
| Adaptive directed mask | 0.2785 | 0.0000 | 0.0000 | 0.0710 |
| Robust directed mask | 0.7455 | 0.0000 | 0.0000 | 0.0273 |
| Dense baseline | 1.2309 | 0.0064 | 0.0000 | 0.0517 |

These results establish five-step ego reaction-trajectory generation under degraded cooperative context. They do not establish complete multi-agent trajectory-scene generation.

### 5.5 Cross-condition behavior

The adaptive directed mask improves structural consistency over the dense baseline in most conditions, but distributional and physical metrics vary by degradation type. Delay and dropout can produce different tradeoffs, so no single metric is treated as sufficient evidence of reliability.

## 6. Limitations and Future Work

- Results use the official V2X-Seq-TFD example subset, not the full TFD release.
- Near-miss events are operational labels and partly depend on ego acceleration.
- The generator produces five-step ego reaction trajectories, not complete multi-agent scenes.
- Dropout and delay are controlled simulations, not measured packet traces.
- PC discovery is observational and remains dependent on causal assumptions.
- Scenario bootstrap still contains temporally correlated rows within selected scenarios.
- The evaluation uses three robustness seeds, which is exploratory rather than high-powered statistical inference.
- Joint energy distance is global rather than fully conditional on each context.
- Full physical and rare-event calibration requires larger data and longer trajectory horizons.

Future work should use independent event construction, context-stratified distribution metrics, stronger block-aware causal inference, realistic network traces, additional cooperative datasets, and full multi-agent trajectory generation.

## 7. Reproducibility

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

The pipeline uses MPS on Apple Silicon when available and falls back to CPU. Generated CSV, NumPy, PNG, model, and trajectory-window artifacts are written under `outputs/` or `data/`; raw datasets and virtual environments are not redistributed.

## 8. Availability

Source code: [github.com/Atharva11022/causal-v2x](https://github.com/Atharva11022/causal-v2x). Dataset source: [AIR-THU/DAIR-V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq). Raw trajectory data is not redistributed.

## 9. Conclusion

This project now supports five-step ego reaction-trajectory generation from real cooperative V2X-derived data under controlled degraded-context conditions. Directed stability-aware masks improve structural selectivity in the primary experiment, while the results also expose distributional and calibration tradeoffs. The work is a defensible research prototype for degraded-V2X ego reaction generation, not yet a complete deployed multi-agent scenario generator.
