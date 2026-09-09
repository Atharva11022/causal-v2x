# Claude Master Context: Phase-I Mid-Semester Dissertation Report

## Immediate objective

Prepare the **Mid-Semester Dissertation Phase-I report and PPT** for submission and evaluation. The report must be produced in **LaTeX** and submitted through the Turnitin portal.

Current date: **9 September 2026**.

Reported institutional deadline: **11 September 2026, 5:00 PM**.

The report must be ready for a spiral-bound copy for the examiners and a LaTeX submission to Turnitin. The overall Turnitin similarity must be below 15%. Do not fabricate similarity results; the actual Turnitin percentage can only be verified through Turnitin.

## Non-negotiable approved project

Exact title:

> **Reliable Near-Miss Driving Scenario Generation Under Degraded V2X Communication Using Stability-Aware Causal Discovery**

The approved abstract is binding. Do not replace the project with a different dataset, research question, task, or title. Do not weaken the project into a generic V2X perception project.

The project combines:

1. Real cooperative vehicle-infrastructure trajectory data.
2. Near-miss driving event derivation from trajectory safety measures.
3. Simulated V2X communication degradation through context dropout and temporal delay.
4. Causal discovery under clean and degraded communication.
5. Scenario-level bootstrap edge-stability estimation.
6. Stability-aware causal masking.
7. Conditional variational autoencoder generation of near-miss reaction states.
8. Comparison with a dense unconstrained baseline.

## Student and filename information

Verified from the existing project material:

- Student: **Atharva Abhyankar**
- MIS: **712552009**
- Program/context: **MTech Data Science**

The guide’s exact official name is not present in the repository. Do not invent it. Ask the student for the exact guide name before finalizing the filename and title-page details.

Required filename format from the institutional notice:

```text
Student-Name_MIS-Number_Guide-Name
```

Use the exact guide name supplied by the student. Do not guess spelling, initials, title, or department.

## Exact institutional Phase-I requirements

The notice requires the Mid-Semester Dissertation Phase-I report to cover these chapters/sections:

### Chapter 1

- 1.1 Introduction to the Research Area
- 1.2 Motivation for Study
- 1.3 Research Gaps
- 1.4 Well-Defined Dissertation Problem Statement
- 1.5 Research Objectives

### Chapter 2

- 2.1 Detailed Literature Survey
- 2.2 Review Summary

### Chapter 3

- Software Requirements Specification

### Chapter 4

- Concrete Future Plan

### End matter

- References

The notice says these are guidelines and should be discussed with the guide, but these sections must be covered in the report draft. The report should also include a title page, table of contents, list of figures/tables if appropriate, and a concise proof-of-concept/preliminary-status section only if compatible with the guide’s preferred format.

## What Phase-I report must emphasize

This is Phase-I, not the final dissertation. The report must demonstrate:

- literature survey
- problem definition
- motivation and objectives
- research gaps
- preliminary design
- feasibility
- modular architecture
- requirements specification
- concrete Phase-II plan
- preliminary proof of concept where useful

Do not present preliminary example-subset results as final dissertation results. Clearly label them as proof-of-concept evidence.

## Verified current technical state

Primary dataset:

- Official AIR-THU **V2X-Seq-TFD** trajectory forecasting example.
- This is real cooperative vehicle-infrastructure trajectory data.
- It is not the full TFD release.
- Full TFD trajectory archives were not located in the currently visible public Google Drive structure; the visible full folder exposed SPD archives instead.
- Do not claim full-dataset results.
- Do not say NGSIM is the current primary dataset.
- Do not use the old NGSIM report as current evidence.

Local dataset, excluded from GitHub:

```text
data/v2x_seq_tfd_example
 data/v2x_seq_tfd_example_extracted/V2X-Seq-TFD-Example/
```

The GitHub repository intentionally excludes raw datasets, extracted datasets, virtual environments, model checkpoints, NumPy arrays, and caches.

Verified preprocessing results from the official TFD example:

- 49,826 trajectory rows
- 57 scenarios
- 148 vehicles
- 20,696 plausible interaction rows
- 1,628 derived clean near-miss events
- 7.87% near-miss candidate rate among interaction rows

Near-miss events are operational labels derived from clean trajectories using:

- closing-gap TTC threshold
- hard ego deceleration threshold

They are not pre-existing dataset labels. The report must say this clearly.

Communication degradation conditions:

- clean
- dropout_10
- dropout_25
- dropout_50
- delay_1
- delay_2
- dropout_20_delay_1

Degradation is applied to shared cooperative context only. Clean ego targets and event labels are preserved so communication failure is not confused with a changed ground-truth event.

Causal discovery:

- PC algorithm from causal-learn.
- Scenario-level bootstrap resampling avoids treating correlated temporal frames as independent IID observations.
- TTC is excluded from causal graph nodes because it is deterministically derived from gap and relative speed.
- Selected stability threshold: 0.70.
- Threshold sensitivity was tested at 0.60, 0.70, and 0.80.
- Alpha sensitivity for ego-only discovery was tested at 0.01, 0.05, 0.10, and 0.20.
- Repeated seed bases were tested at 42, 142, and 242.

Latest causal evidence:

- 84 edge-stability records across seven degradation conditions.
- 7 clean cooperative stable edges at threshold 0.70.
- 5 graph edges stable across all seven conditions.
- The robust graph maps to 4 generator context-to-target links.
- Ego-only candidate edge stability ranged from 0.25 to 0.55 across alpha sensitivity and did not reach 0.70.
- Clean stable-edge counts across seed bases 42, 142, 242: 7, 7, 8.
- Primary combined-condition stable-edge counts across seed bases 42, 142, 242: 7, 8, 6.

Generator:

- Small CVAE designed for MacBook M2, 8 GB RAM.
- Generates two ego reaction-state variables: `ego_accel` and `ego_speed`.
- It does not generate complete multi-agent trajectory windows.
- Use the accurate term: **near-miss reaction-state generation conditioned on cooperative context**.
- Models compared:
  - adaptive condition-specific stability-aware masked CVAE
  - robust all-condition masked CVAE
  - dense unconstrained baseline

Latest primary condition: `dropout_20_delay_1`.

Latest consistency-gap evidence:

- adaptive masked model: 0.324039
- robust all-condition mask: 0.214556
- dense baseline: 0.029534

The masked models outperform the dense baseline on the primary combined condition. Delay-2 is an adverse result and must be reported honestly. Do not claim the masked model wins every condition.

## Current repository files Claude should read

Read these first:

```text
CLAUDE_PHASE1_MASTER_CONTEXT.md
HANDOFF.md
README.md
requirements.txt
.gitignore
docs/abstract_final.txt
```

Then read the implementation:

```text
src/00_check_setup.py
src/01_preprocess.py
src/02_causal_discovery.py
src/03_generative_replay.py
src/04_evaluate.py
tests/test_pipeline.py
```

Then use these current evidence files:

```text
outputs/evaluation_summary.csv
outputs/performance_metrics.csv
outputs/causal_edge_stability.csv
outputs/causal_stability_summary.csv
outputs/causal_threshold_sensitivity.csv
outputs/causal_ego_alpha_sensitivity.csv
outputs/causal_seed_sensitivity.csv
outputs/causal_context_comparison.csv
outputs/causal_edges_v2x.csv
outputs/causal_edges_ego_only.csv
outputs/causal_edges_robust.csv
outputs/causal_consistency_comparison.png
outputs/causal_graph_comparison.png
outputs/causal_graph_ego_only.png
```

Do not use these local legacy files as current technical evidence unless they are rewritten and revalidated:

```text
docs/report.md
docs/Internship_Report.docx
docs/Internship_Report.pdf
docs/presentation.pptx
```

They contain older NGSIM-era content and were removed from the public GitHub tree. Their local copies are preserved only as historical material.

## Required Phase-I report deliverable

Create a complete LaTeX report source tree, preferably under a new directory such as:

```text
phase1_report/
  main.tex
  references.bib
  figures/
  tables/
  README.md
```

The final PDF should be generated locally and checked before submission. Keep the report suitable for a spiral-bound academic submission.

The report must contain exactly the following substantive structure unless the guide requests a different arrangement:

```text
Title Page
Abstract
Table of Contents

Chapter 1: Introduction
  1.1 Introduction to the Research Area
  1.2 Motivation for Study
  1.3 Research Gaps
  1.4 Well-Defined Dissertation Problem Statement
  1.5 Research Objectives

Chapter 2: Literature Review
  2.1 Detailed Literature Survey
  2.2 Review Summary

Chapter 3: Software Requirements Specification

Chapter 4: Concrete Future Plan

Preliminary Proof of Concept / Feasibility Evidence
References
```

The guide may adjust the exact formatting, but do not omit the required chapters.

## Chapter content requirements

### Chapter 1

Explain the research area as the intersection of:

- cooperative V2X perception
- degraded communication
- causal discovery from interacting traffic participants
- safety-critical near-miss scenario generation

Motivation must explain why near misses are valuable, why they are rare and expensive to collect, why V2X context matters, and why causal-edge reliability matters under delay/dropout.

Research gaps must be literature-grounded and cautious. The intended gap is the missing integration of:

1. real cooperative trajectory data
2. deliberately degraded shared context
3. stability-aware causal discovery
4. stability-informed near-miss generation

Problem statement must be one precise paragraph and must not claim live-network measurements or full-trajectory generation.

Objectives must match the approved abstract and implementation. Avoid adding objectives that are not in the title or current project.

### Chapter 2

Use a detailed literature survey organized by themes:

1. causal discovery in driving behaviour
2. V2X cooperative perception
3. communication delay, dropout, and interruption
4. near-miss and safety-critical scenario generation
5. causal or graph-constrained generative modelling
6. trajectory datasets, especially V2X-Seq-TFD

Use real, verifiable references only. Do not invent authors, papers, venues, years, metrics, dataset properties, or acceptance claims. Clearly separate what prior work demonstrates from what this project proposes.

The review summary must map literature limitations to the proposed objectives.

### Chapter 3: Software Requirements Specification

Include:

- purpose and scope
- stakeholders
- functional requirements
- non-functional requirements
- hardware requirements
- software/dependency requirements
- dataset/input requirements
- output requirements
- reproducibility requirements
- assumptions
- constraints and limitations
- ethical/data-use considerations

Functional requirements must include loading TFD trajectories, interaction extraction, near-miss labeling, context degradation, causal discovery, stability scoring, mask construction, CVAE training, baseline training, evaluation, and artifact generation.

Hardware constraints must state MacBook M2, 8 GB RAM, and 512 GB storage.

### Chapter 4: Concrete Future Plan

Give a realistic Phase-II schedule with implementation, testing, evaluation, report, presentation, and paper milestones. Explicitly include:

- final clean end-to-end run
- repeated-seed robustness
- threshold sensitivity
- full-data access attempt or documented example-subset limitation
- final ablation tables
- test plan and test results
- paper preparation
- Turnitin review
- guide review and corrections

Do not promise full TFD results if the full data cannot be obtained.

### Preliminary evidence

Present only as proof-of-concept evidence. Use the verified values in this document. Include a table and/or figures for:

- dataset/example scale
- degradation conditions
- edge stability by condition
- threshold/alpha/seed sensitivity
- primary three-way generator comparison

Every result must be labeled as official TFD example-subset evidence.

## LaTeX and originality rules

- Report must be written in LaTeX.
- Use the institution’s template if the student supplies one; otherwise create a clean academic LaTeX structure and state that the guide should approve formatting.
- Use original wording.
- Do not copy the approved abstract verbatim beyond necessary project terminology.
- Do not use fabricated citations.
- Do not claim Turnitin similarity below 15% without an actual Turnitin report.
- Avoid unnecessary quotations.
- Cite every external dataset, method, and literature claim.
- Maintain a BibTeX file with verifiable references.
- Keep similarity reduction ethical: original synthesis, proper citations, no plagiarism tricks or synonym substitution.

## PPT requirement

Prepare a concise Phase-I presentation aligned exactly with the report. It should cover:

1. title and student details
2. research area
3. motivation
4. problem statement
5. research gaps
6. objectives
7. literature themes
8. proposed architecture
9. software requirements and feasibility
10. proof-of-concept evidence
11. concrete Phase-II plan
12. limitations and expected outcomes

Do not use old NGSIM results in the slides.

## Missing information Claude must request before finalizing

Claude must ask the student for:

1. Guide’s exact full name and official designation.
2. Department/institute wording required on the title page.
3. Official LaTeX template, if supplied by the department.
4. Whether the guide wants the report called “Dissertation Phase-I Mid-Semester Report” or another exact title.
5. Any page limit, font, margin, citation style, or PPT slide limit not included in the notice.

Do not invent these details.

## Claude operating instruction

```text
Read CLAUDE_PHASE1_MASTER_CONTEXT.md, HANDOFF.md, README.md, and docs/abstract_final.txt completely before editing or writing anything.

The institutional deadline is 11 September 2026 at 5:00 PM. The deliverable is a Phase-I Mid-Semester Dissertation report in LaTeX plus an aligned PPT. The required chapters are Chapter 1 Introduction, Chapter 2 Literature Review, Chapter 3 Software Requirements Specification, Chapter 4 Concrete Future Plan, and References.

Treat the exact project title and approved abstract as binding. Do not change the research direction. Use only verified project facts from the handoff. Do not revive NGSIM content. Do not invent the guide name, citations, dataset results, full-dataset results, Turnitin score, or institutional formatting rules.

First ask only for missing personal/template details listed in this file. Then create the LaTeX report and aligned presentation plan. Keep the report Phase-I appropriate: literature, problem definition, requirements, preliminary design, feasibility, proof-of-concept evidence, and a concrete Phase-II plan. Label all current experimental results as official V2X-Seq-TFD example-subset proof-of-concept results.

Before finalizing, check:
- exactly the required chapters are present;
- the title is exact;
- all quantitative claims match the verified values;
- all citations are real and present in references.bib;
- no NGSIM-era claims remain;
- no unsupported Scopus or Turnitin guarantees are made;
- LaTeX compiles;
- the filename follows Student-Name_MIS-Number_Guide-Name after the guide name is supplied;
- the report and PPT tell the same story.
```
