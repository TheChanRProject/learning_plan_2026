# AI Plan — Q3–Q4 2026 (Research)

Goal: research-oriented progress in **survival analysis** and **deep learning**, prioritized for AI/ML engineering work.

**Constraint:** max **1.5h/day**, 3 jobs. **Weekdays only** (weekends off). Window: **Aug 26 – Dec 31, 2026**.

**Starts:** Thu Sep 11 (first Thu after Phase 0 Rust kickoff ends Sep 8). Programming languages: [Programming_Plan.md](./Programming_Plan.md).

---

## Cadence

| Weekday | Track | Roadmap source |
|---------|-------|----------------|
| Thu | Survival analysis | [Algorithm_Roadmap.md](../../topics/machine_learning/survival_analysis/Algorithm_Roadmap.md) |
| Fri | Deep learning | [New_Architectures.md](../../topics/machine_learning/deep_learning/New_Architectures.md) → [Geometric_Deep_Learning.md](../../topics/machine_learning/deep_learning/Geometric_Deep_Learning.md) → [Gradient_Free_Algorithms.md](../../topics/machine_learning/deep_learning/Gradient_Free_Algorithms.md) |

**Session recipe (every Thu/Fri):** 45m read one paper or roadmap section → 30m 5–10 line notes (claim, method, gap) → 15m optional tiny code check (lifelines snippet, PyTorch toy, or pseudocode). No full reimplementations.

---

## Survival analysis (Thursdays)

Drive from [Algorithm_Roadmap.md](../../topics/machine_learning/survival_analysis/Algorithm_Roadmap.md).

| Month | Week-of focus | 1.5h session recipe |
|-------|---------------|---------------------|
| **Sep** | Foundations: S(t), h(t), H(t), censoring, truncation | Pick 1 foundation link per Thu; define each in your own words; sketch one toy dataset with right-censoring |
| **Oct** | Classical: Cox PH, Weibull, AFT | 45m Cox paper skim; 30m lifelines or scikit-survival notebook (fit + plot survival curve); 15m note when Cox breaks |
| **Nov** | Complex events + Bayesian skim | Competing risks, frailty, time-varying covariates; INLA / latent Gaussian models — read only, 10-line summary |
| **Dec** | Deep survival + healthcare apps | DeepSurv, DeepHit, Deep Survival Machines from roadmap; skim one healthcare app (LOS, mortality, readmission); **research memo** (1 page): classical vs deep for your domain |

### Sep Thu paper queue (foundations)

| Week of | Topic | Action |
|---------|-------|--------|
| Sep 11 | Survival function S(t) | Read roadmap link; plot hand-drawn S(t) for 3 subjects |
| Sep 18 | Hazard h(t), cumulative H(t) | Relate h to S; one worked numeric example |
| Sep 25 | Censoring | Right vs interval; why naive regression fails |

### Oct–Dec Thu queue (headings from roadmap)

- **Oct:** Cox PH → Weibull → AFT (one classical model per week + notebook week 2)
- **Nov:** Competing risks → frailty → TVC → INLA skim
- **Dec:** DeepSurv / DeepHit / DSM → healthcare applications → memo

---

## Deep learning (Fridays)

Priority order: architectures → geometric → gradient-free. Links live in repo roadmaps.

| Month | Week-of focus | 1.5h session recipe |
|-------|---------------|---------------------|
| **Sep** | Classical + Transformer core | MLP → CNN → RNN/LSTM (1 family per Fri); then Transformer encoder/decoder papers from [New_Architectures.md](../../topics/machine_learning/deep_learning/New_Architectures.md) |
| **Oct** | ViT + attention variants; start SSM | ViT + sparse/MLA skim; S4 then Mamba — focus on *why* SSM vs attention for long context |
| **Nov** | Geometric deep learning | [Geometric_Deep_Learning.md](../../topics/machine_learning/deep_learning/Geometric_Deep_Learning.md): blueprint → graph Laplacians → MPNN; optional hyperbolic skim |
| **Dec** | Gradient-free + synthesis | [Gradient_Free_Algorithms.md](../../topics/machine_learning/deep_learning/Gradient_Free_Algorithms.md): MeZO, ES; **synthesis note** linking DL themes ↔ deep survival (Dec Thu) |

### Sep Fri paper queue (New_Architectures)

| Week of | Topic | Action |
|---------|-------|--------|
| Sep 12 | MLP / CNN | One classical arch; note inductive bias |
| Sep 19 | RNN / LSTM / GRU | Sequence modeling baseline before Transformers |
| Sep 26 | Transformer encoder–decoder | Read attention paper; diagram Q/K/V in notes |

### Oct–Dec Fri queue

- **Oct:** ViT → sparse/MLA → S4 → Mamba
- **Nov:** Geometric DL blueprint → MPNN → (optional) hyperbolic embeddings
- **Dec:** MeZO → evolution strategies → synthesis with survival deep models

---

## Cross-links to programming tracks

| When | Link |
|------|------|
| Dec Thu survival memo | Reference deep survival models you read |
| Dec Fri synthesis | Connect to DeepSurv / transformer survival (SurvTRACE, MOTOR from roadmap) |
| Dec Mon Rust capstone | Parquet feature pipeline feeding survival experiment |
| Dec Wed Mojo capstone | Kernel accelerating a bottleneck from DL or survival notebook |
| Dec Tue TypeScript capstone | Typed API wrapping mock survival or inference scores |

---

## Out of scope

- Weekend sessions, >1.5h blocks
- Full paper reimplementations
- MLOps course work (see [mlops Udemy](../../topics/machine_learning/mlops/Udemy_Courses.md) only if spare capacity after Dec)
