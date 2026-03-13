# Geometric Content Moderation — Project Plan

## Thesis

Content moderation is fundamentally a geometric problem. Policies form
hierarchies (hyperbolic), content lives on manifolds (not in flat
spaces), fairness requires distribution-aware metrics (Mahalanobis/SPD),
and robustness requires understanding the manifold structure of decision
boundaries. Current approaches treat moderation as flat multi-label
classification, discarding the rich structural relationships between
policy categories, content contexts, and demographic groups.

**This project builds a geometric content moderation framework** that
uses hyperbolic embeddings, manifold decision boundaries, and
topological features to produce classifiers that are simultaneously
more accurate, more fair, and more robust than flat baselines.

---

## Architecture Overview

```
                    ┌──────────────────────────────────┐
                    │     Content Input (text/image)    │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │  Transformer Encoder (e.g. BERT) │
                    │  + Hyperbolic Attention Bias      │
                    │    (Poincaré policy distances)    │
                    └──────────────┬───────────────────┘
                                   │
                 ┌─────────────────┼─────────────────────┐
                 │                 │                       │
     ┌───────────▼──┐   ┌────────▼────────┐   ┌─────────▼─────────┐
     │  Hyperbolic   │   │  Manifold        │   │  Topological      │
     │  Policy Head  │   │  Decision Head   │   │  Feature Head     │
     │  (Poincaré    │   │  (SPD boundary   │   │  (Persistent      │
     │   ball → tax) │   │   + curvature)   │   │   homology)       │
     └───────┬───────┘   └────────┬────────┘   └─────────┬─────────┘
             │                    │                       │
             └────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────▼────────────────────┐
                    │  Geometric Fusion + Calibration   │
                    │  (manifold-aware ensemble)        │
                    └─────────────┬────────────────────┘
                                  │
              ┌───────────────────┼──────────────────────┐
              │                   │                      │
    ┌─────────▼──────┐  ┌────────▼────────┐  ┌──────────▼────────┐
    │  Classification │  │  Severity Score  │  │  Fairness Audit   │
    │  (policy node)  │  │  (geodesic dist) │  │  (Mahalanobis Δ)  │
    └────────────────┘  └─────────────────┘  └───────────────────┘
```

---

## Module Breakdown

### Module 1: `geomod.manifold` — Core Geometric Primitives

The mathematical foundation. Reusable across all other modules.

| File | Purpose |
|------|---------|
| `poincare.py` | Poincaré ball: Möbius add, exp/log maps, distance, projection, geodesics |
| `spd.py` | SPD manifold: LogEuclidean/affine-invariant metrics, Cholesky param, Fréchet mean |
| `lorentz.py` | Lorentz model (alternative hyperbolic): numerically stabler for high-dim |
| `utils.py` | Shared: parallel transport, curvature estimation, manifold interpolation |

**Key design decision**: Support both Poincaré ball and Lorentz model.
Poincaré is more intuitive for visualization; Lorentz is numerically
stabler for training. Provide seamless conversion between them.

### Module 2: `geomod.policy` — Policy Taxonomy Embeddings

Content policies have natural hierarchy:
```
Violence
├── Threats
│   ├── Direct threats
│   └── Indirect threats
├── Graphic violence
│   ├── Real-world
│   └── Fictional
└── Self-harm
    ├── Promotion
    └── Discussion
Hate Speech
├── Racial
│   ├── Slurs
│   └── Stereotyping
├── Gender
├── Religious
└── Disability
Sexual Content
├── Explicit
├── Suggestive
└── Educational
...
```

| File | Purpose |
|------|---------|
| `taxonomy.py` | Define policy trees, embed into Poincaré ball, learn from labeled data |
| `boundary.py` | Geodesic decision boundaries between adjacent policy categories |
| `severity.py` | Severity as geodesic distance from "benign" origin on the manifold |

**Core idea**: Embed the full policy taxonomy into hyperbolic space.
Root = origin. Depth ∝ specificity. Distance between nodes encodes
semantic relatedness of policy categories. A "borderline" content item
sits near the geodesic midpoint between categories — the model can
express genuine uncertainty geometrically rather than via softmax
probabilities.

### Module 3: `geomod.models` — Neural Architectures

| File | Purpose |
|------|---------|
| `encoder.py` | Geometric transformer encoder (BERT/RoBERTa + hyperbolic bias) |
| `classifier.py` | Multi-head classifier: hyperbolic policy head + manifold severity head |
| `attention.py` | Hyperbolic attention bias module (forward hooks, like deep-past) |
| `ensemble.py` | Fréchet mean ensemble on the manifold (not naive logit averaging) |

**Geometric attention bias**: Inject policy-hierarchy distances into
the transformer's self-attention. Tokens semantically related to nearby
policy categories attend more to each other. This is structurally
identical to what we built for Akkadian (deep-past), but the bias comes
from policy embeddings rather than cuneiform sign hierarchy.

**Hyperbolic classification head**: Instead of a linear layer →
softmax, project the [CLS] embedding into the Poincaré ball and
classify by nearest policy node (geodesic distance). This respects
the hierarchical label structure and produces calibrated severity
scores for free.

### Module 4: `geomod.fairness` — Geometric Fairness Auditing

| File | Purpose |
|------|---------|
| `mahalanobis.py` | Mahalanobis distance between group-conditioned score distributions |
| `spd_audit.py` | SPD manifold comparison of covariance matrices across demographic groups |
| `geodesic_gap.py` | Geodesic fairness gap: measure policy-space distance between group outcomes |
| `report.py` | Generate fairness reports with geometric visualizations |

**Core idea**: Fairness isn't a scalar. Two classifiers can have
identical accuracy parity but very different *geometric* fairness
profiles. A classifier that confuses "discussion of racism" with
"racial slur" is geometrically far less fair than one that confuses
"suggestive" with "explicit" — even if the error rates are identical.

**Mahalanobis fairness**: For each demographic group, compute the
covariance of the model's output scores on the policy manifold. Compare
these covariance matrices using the SPD manifold distance. Large SPD
distance → the model behaves qualitatively differently across groups,
even if aggregate metrics look similar.

### Module 5: `geomod.robustness` — Adversarial Geometric Robustness

| File | Purpose |
|------|---------|
| `mri.py` | Manifold Robustness Index: measure robustness in manifold space |
| `attacks.py` | Geometric adversarial attacks: follow geodesics toward boundary |
| `certify.py` | Certified robustness via manifold curvature bounds |
| `augment.py` | Robustness-aware augmentation: sample from manifold neighborhoods |

**Manifold Robustness Index (MRI)**: Adapted from the structural
fuzzing book. For a content item at point p on the manifold, compute:
- Forward: perturb the input (typos, paraphrase, character subs)
- Map perturbations to the manifold
- MRI = (volume of perturbation cloud) / (distance to decision boundary)

High MRI → robust. Low MRI → fragile (small perturbation crosses the
boundary).

**Geodesic attacks**: Instead of L_p norm adversarial attacks, find the
shortest geodesic path from a content item to the nearest decision
boundary. This produces more semantically meaningful adversarial
examples that expose real policy gaps.

### Module 6: `geomod.data` — Datasets and Benchmarks

| File | Purpose |
|------|---------|
| `datasets.py` | Loaders for Jigsaw, HateXplain, Civil Comments, OpenAI Mod |
| `taxonomy_datasets.py` | Map dataset labels to our policy taxonomy |
| `synthetic.py` | Generate synthetic test cases at specific manifold locations |
| `splits.py` | Stratified splits respecting both label and demographic distribution |

**Supported datasets** (with standard geometric preprocessing):
- Jigsaw Toxic Comment Classification (Kaggle)
- HateXplain (rationale-annotated hate speech)
- Civil Comments (8 identity attributes)
- OpenAI Moderation (API-compatible)

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal**: Core geometric primitives + policy taxonomy embedding.

1. Implement `geomod.manifold.poincare` — full Poincaré ball ops
   (mobius_add, exp_map, log_map, dist, project, geodesic)
2. Implement `geomod.manifold.spd` — SPD manifold basics
   (log_euclidean distance, Cholesky parameterization, Fréchet mean)
3. Implement `geomod.policy.taxonomy` — define policy tree structure,
   embed into Poincaré ball, visualize with matplotlib
4. Write tests for all geometric operations (numerical stability,
   projection correctness, distance triangle inequality)
5. **Deliverable**: Interactive notebook showing policy taxonomy
   embedded in the Poincaré disk, with geodesic distances between
   policy categories

### Phase 2: Models (Week 3-4)

**Goal**: Geometric transformer classifier.

1. Implement `geomod.models.attention` — hyperbolic attention bias
   (forward hooks on BERT/RoBERTa encoder layers)
2. Implement `geomod.models.classifier` — hyperbolic classification
   head (project to ball, classify by nearest policy node)
3. Implement `geomod.data.datasets` — load Jigsaw + Civil Comments
   with policy taxonomy label mapping
4. Training pipeline: fine-tune geometric classifier on Jigsaw
5. **Deliverable**: Ablation comparing flat classifier vs. geometric
   on Jigsaw (accuracy, F1, severity calibration)

### Phase 3: Fairness (Week 5-6)

**Goal**: Geometric fairness auditing toolkit.

1. Implement `geomod.fairness.mahalanobis` — group-conditioned
   Mahalanobis distance between score distributions
2. Implement `geomod.fairness.spd_audit` — SPD manifold comparison
   of group covariance matrices
3. Implement `geomod.fairness.geodesic_gap` — geodesic fairness gap
   metric
4. Apply to Civil Comments (8 identity attributes): compute geometric
   fairness profile for flat vs. geometric classifier
5. **Deliverable**: Fairness report showing that geometric classifier
   has better fairness profile (smaller SPD distances between groups)

### Phase 4: Robustness (Week 7-8)

**Goal**: Adversarial robustness analysis.

1. Implement `geomod.robustness.mri` — Manifold Robustness Index
2. Implement `geomod.robustness.attacks` — geodesic adversarial
   attacks (find shortest path to decision boundary)
3. Implement `geomod.robustness.augment` — manifold-neighborhood
   augmentation for robustness training
4. Evaluate: MRI comparison of flat vs. geometric classifier
5. **Deliverable**: Robustness analysis showing geometric classifier
   is more robust to adversarial perturbations

### Phase 5: Paper + Release (Week 9-10)

**Goal**: Publishable paper + open-source release.

1. Implement `geomod.robustness.certify` — certified robustness bounds
2. Comprehensive ablation study: flat vs. hyperbolic vs. full geometric
   on multiple datasets
3. Write paper: "Geometric Content Moderation: Hyperbolic Policy
   Embeddings, Manifold Decision Boundaries, and Distributional
   Fairness"
4. Target venue: AAAI 2027, ACL 2027, or FAccT 2027
5. **Deliverable**: Paper draft + reproducibility notebook + pip-installable package

---

## Key Technical Decisions

### 1. Why Poincaré ball (not just Euclidean)?
Content policies are hierarchical. Euclidean space can't embed trees
without O(n) distortion. The Poincaré ball embeds trees with O(log n)
distortion. This means:
- Related policy categories are naturally close
- Severity increases with depth (distance from origin)
- "Borderline" content sits geometrically between categories

### 2. Why SPD manifolds for fairness?
Fairness isn't just about mean scores — it's about the *distribution*
of scores. SPD manifolds are the natural space for comparing covariance
matrices. The geodesic distance on the SPD manifold captures differences
in both scale and shape of score distributions, not just their centers.

### 3. Why manifold robustness (not L_p)?
L_p adversarial attacks (character substitution, word swap) don't
respect semantic structure. A geodesic attack finds the *semantically*
nearest decision boundary, producing adversarial examples that
correspond to real policy ambiguities rather than superficial typos.

### 4. Base model: BERT or RoBERTa?
Start with `microsoft/deberta-v3-base` — best performance/size ratio
for classification tasks. The geometric modules are model-agnostic
(forward hooks), so swapping is trivial.

### 5. Policy taxonomy: fixed or learnable?
Start with a fixed taxonomy (manually defined from industry standard
policies). The Poincaré embedding of the taxonomy is learned, but the
tree structure is fixed. Phase 2 experiment: let the model learn
additional edges or refine the hierarchy.

---

## Evaluation Plan

### Metrics

| Metric | What it measures | Module |
|--------|-----------------|--------|
| Macro-F1 | Classification accuracy | models |
| Severity Spearman ρ | Severity ranking quality | policy |
| SPD Fairness Gap | Distributional fairness | fairness |
| Geodesic Fairness Gap | Policy-aware fairness | fairness |
| MRI | Adversarial robustness | robustness |
| Certified Radius | Guaranteed robustness | robustness |

### Ablation Configurations

| Config | Hyperbolic | Manifold Boundary | Augmentation | TDA |
|--------|:---:|:---:|:---:|:---:|
| A: Flat baseline | | | | |
| B: +Hyperbolic | X | | | |
| C: +Manifold | X | X | | |
| D: +Augmentation | X | X | X | |
| E: Full geometric | X | X | X | X |

### Datasets

| Dataset | Size | Labels | Identity | Use |
|---------|------|--------|----------|-----|
| Jigsaw Toxic | 160K | 6 binary | No | Primary accuracy benchmark |
| Civil Comments | 1.8M | 7 binary | 8 groups | Fairness evaluation |
| HateXplain | 20K | 3-class | Rationales | Explainability |
| Synthetic | ~5K | Full taxonomy | Controlled | Robustness testing |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hyperbolic training instability | Training diverges | Lorentz model fallback; gradient clipping; burnin with Euclidean |
| Policy taxonomy too rigid | Misses emergent content types | Learnable edge weights; hierarchical softmax |
| Fairness metrics don't improve | Paper story weakens | Focus on severity calibration as primary contribution |
| GPU memory for geometric ops | OOM on large batches | Efficient pairwise distance (chunked); mixed precision |
| TDA too slow for large datasets | Phase 4 blocked | Subsample for topological features; approximate persistence |

---

## File Tree (Target)

```
geometric-moderation/
├── CLAUDE.md
├── PLAN.md                    ← this file
├── pyproject.toml
├── src/
│   └── geomod/
│       ├── __init__.py
│       ├── manifold/
│       │   ├── __init__.py
│       │   ├── poincare.py    # Poincaré ball operations
│       │   ├── spd.py         # SPD manifold + Cholesky
│       │   ├── lorentz.py     # Lorentz model (numerically stable)
│       │   └── utils.py       # Parallel transport, curvature, interp
│       ├── models/
│       │   ├── __init__.py
│       │   ├── encoder.py     # Geometric transformer encoder
│       │   ├── classifier.py  # Hyperbolic classification head
│       │   ├── attention.py   # Hyperbolic attention bias hooks
│       │   └── ensemble.py    # Fréchet mean ensemble
│       ├── policy/
│       │   ├── __init__.py
│       │   ├── taxonomy.py    # Policy tree → Poincaré embedding
│       │   ├── boundary.py    # Geodesic decision boundaries
│       │   └── severity.py    # Severity = geodesic distance
│       ├── fairness/
│       │   ├── __init__.py
│       │   ├── mahalanobis.py # Group-conditioned Mahalanobis
│       │   ├── spd_audit.py   # SPD covariance comparison
│       │   ├── geodesic_gap.py# Policy-aware fairness gap
│       │   └── report.py      # Visualization + reporting
│       ├── robustness/
│       │   ├── __init__.py
│       │   ├── mri.py         # Manifold Robustness Index
│       │   ├── attacks.py     # Geodesic adversarial attacks
│       │   ├── certify.py     # Certified robustness bounds
│       │   └── augment.py     # Manifold-neighborhood augmentation
│       └── data/
│           ├── __init__.py
│           ├── datasets.py    # Jigsaw, Civil Comments, HateXplain
│           ├── taxonomy_datasets.py  # Label → taxonomy mapping
│           ├── synthetic.py   # Synthetic test generation
│           └── splits.py      # Stratified splitting
├── tests/
│   ├── test_poincare.py
│   ├── test_spd.py
│   ├── test_taxonomy.py
│   ├── test_classifier.py
│   └── test_fairness.py
├── notebooks/
│   ├── 01_policy_taxonomy.ipynb    # Visualize policy embeddings
│   ├── 02_geometric_classifier.ipynb # Train + ablation
│   ├── 03_fairness_audit.ipynb     # Fairness analysis
│   └── 04_robustness.ipynb         # MRI + adversarial
└── paper/
    └── geometric_moderation.tex
```
