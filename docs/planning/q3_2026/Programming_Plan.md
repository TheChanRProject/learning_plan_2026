# Programming Plan — Q3–Q4 2026

Goal: year-end fluency in **Rust**, **TypeScript**, and **Mojo** for **AI/ML engineering** — data pipelines, typed serving, hot-path kernels, agents.

**Constraint:** max **1.5h/day**, 3 jobs. **Weekdays only** (weekends off). Window: **Aug 26 – Dec 31, 2026** (~92 weekdays ≈ 138h).

Visual: [2-week Rust roadmap](../../topics/programming_languages/rust/learning_topics/assets/rust-2week-roadmap.png)

---

## Cadence (Sep 9 onward)

One track per weekday so 1.5h stays usable. Research tracks live in [AI_Plan.md](./AI_Plan.md).

| Weekday | Track | Section |
|---------|-------|---------|
| Mon | Rust | below |
| Tue | TypeScript | below |
| Wed | Mojo | below |
| Thu | Survival analysis | [AI_Plan.md](./AI_Plan.md) |
| Fri | Deep learning | [AI_Plan.md](./AI_Plan.md) |

**Phase 0 exception:** Rust daily block **Aug 26 – Sep 8** (below). No parallel languages during Phase 0. Mojo / TypeScript / AI tracks start **Tue Sep 9**.

---

## Rust

### Phase 0 — Daily kickoff (Aug 26 – Sep 8)

Window: **Wed Aug 26 – Tue Sep 8, 2026** (10 weekdays). Guides: [`learning_topics/`](../../topics/programming_languages/rust/learning_topics/00-roadmap.md).

| Date | Day | Guide | Do (1.5h) |
|------|-----|-------|-----------|
| 2026-08-26 | Wed | [01 Setup & Core](../../topics/programming_languages/rust/learning_topics/01-setup-core.md) | Grider Foundations + Core. `cargo new day01_core`; `User` struct + greeting; `cargo check` clean. |
| 2026-08-27 | Thu | [02 Ownership & Borrowing](../../topics/programming_languages/rust/learning_topics/02-ownership-borrowing.md) | Grider Ownership. `longest_len` / `append_bang`; break borrow checker once, then fix. |
| 2026-08-28 | Fri | [03 Lifetimes](../../topics/programming_languages/rust/learning_topics/03-lifetimes.md) | Grider Lifetimes (+ skim Advanced). Implement `longer` + `Excerpt<'a>`; fix invalid return. |
| 2026-08-31 | Mon | [04 Enums & Pattern Matching](../../topics/programming_languages/rust/learning_topics/04-enums-pattern-matching.md) | Grider Enums. CLI `Cmd` enum; parse `&str` → `Option`; exhaustive `match`, no naked `unwrap`. |
| 2026-09-01 | Tue | [05 Modules & Errors](../../topics/programming_languages/rust/learning_topics/05-modules-errors.md) | Grider Modules + Errors. `parse` module + `Result` + `?` in `run()`. |
| 2026-09-02 | Wed | [06 Iterators, Generics & Traits](../../topics/programming_languages/rust/learning_topics/06-iterators-generics-traits.md) | Grider Iterators + Generics/Traits (Codestars if needed). Iterator chain + `Score` trait + `best`. |
| 2026-09-03 | Thu | [07 Async & Concurrency](../../topics/programming_languages/rust/learning_topics/07-async-concurrency.md) | Codestars concurrency/async. `mpsc` thread demo + Tokio `join!` sleeps; note agent I/O choice. |
| 2026-09-04 | Fri | [08 DataFusion](../../topics/programming_languages/rust/learning_topics/08-datafusion-spark-analog.md) | [Rust Learning Plan §1](../../topics/programming_languages/rust/Rust_Learning_Plan.md). Run one DataFusion SQL; 3 bullets DataFusion vs Spark. |
| 2026-09-07 | Mon | [09 Polars & Arrow](../../topics/programming_languages/rust/learning_topics/09-polars-arrow.md) | Plan §2 + §4. Polars filter/group_by; note Polars vs DataFusion for your workflow. |
| 2026-09-08 | Tue | [10 Streaming & Agents](../../topics/programming_languages/rust/learning_topics/10-streaming-agents-ml.md) | Plan §3–5 + McDonogh AutoGPT. Agent loop sketch; 5-bullet lead pitch (stack image in guide). |

**Off:** Sat–Sun Aug 29–30, Sat–Sun Sep 5–6.

### Phase 1+ — Mondays (Sep 14 – Dec 28)

After guides 01–10: deepen ML/AI stack. ~1 Mon/week ≈ 16 sessions.

| Month | Week-of focus | 1.5h session recipe |
|-------|---------------|---------------------|
| **Sep** | Candle or Burn hello-train; Polars/DataFusion on real Parquet | 30m read crate docs; 45m `cargo new` + load Parquet + one transform; 15m notes on when Rust beats Python for this step |
| **Oct** | Arrow FFI / Python interop; agent runtime polish | 30m PyO3 or Arrow FFI skim; 45m expose one Rust fn to Python or polish Tokio agent tool trait; 15m log blockers |
| **Nov** | Embed DataFusion; batch feature pipeline | 30m DataFusion embed example; 45m tiny service: SQL over Parquet → feature row; 15m compare to Spark job you know |
| **Dec** | Capstone | 60m binary: Parquet → transform → mock inference stub; 30m update lead-pitch from guide 10 |

**Sources:** [learning_topics/](../../topics/programming_languages/rust/learning_topics/00-roadmap.md) · [Rust_Learning_Plan.md](../../topics/programming_languages/rust/Rust_Learning_Plan.md) · [Udemy_Courses.md](../../topics/programming_languages/rust/Udemy_Courses.md)

---

## TypeScript

**Starts:** Tue Sep 9 (cadence). **Ends:** Dec 2026. AI/ML focus: typed inference I/O, backend serving, light UI — not React Native primary path.

Guide order (de-emphasize [08-react-native](../../topics/programming_languages/typescript/08-react-native.md)):

1. [00–06](../../topics/programming_languages/typescript/00-roadmap.md) — language + tooling
2. [10-backend](../../topics/programming_languages/typescript/10-backend.md) — Zod, Fastify/Express, typed API
3. [11–12](../../topics/programming_languages/typescript/11-production-patterns.md) — production + tests
4. Light [07-react](../../topics/programming_languages/typescript/07-react.md) / [09-nextjs](../../topics/programming_languages/typescript/09-nextjs.md) — ML demo UI / API routes only

| Month | Week-of focus | 1.5h session recipe |
|-------|---------------|---------------------|
| **Sep** | [01–04](../../topics/programming_languages/typescript/01-foundations.md) foundations → advanced types | 45m guide + Udemy slice (Schwarzmuller/Grider); 30m Zod schema for `{ features: number[] }` → `{ score: number }`; 15m commit notes |
| **Oct** | [05–06](../../topics/programming_languages/typescript/05-objects-classes.md), [10-backend](../../topics/programming_languages/typescript/10-backend.md) | 45m modules + backend guide; 30m Fastify/Express route validates predict payload with Zod; 15m curl test |
| **Nov** | [11–12](../../topics/programming_languages/typescript/11-production-patterns.md) production + tests | 45m Result/branded types; 30m Vitest for schema/parser; 15m optional Next route handler wrapping mock model |
| **Dec** | Capstone | 60m typed ML service stub: input schema → fake scores → typed output; 30m README: how this pairs with Rust/Mojo backends |

**Sources:** [TypeScript_Learning_Plan.md](../../topics/programming_languages/typescript/TypeScript_Learning_Plan.md) · [Udemy_Courses.md](../../topics/programming_languages/typescript/Udemy_Courses.md)

---

## Mojo

**Starts:** Wed Sep 9 (cadence). **Ends:** Dec 2026. Plan rows only — no `learning_topics` series yet.

| Month | Week-of focus | 1.5h session recipe |
|-------|---------------|---------------------|
| **Sep** | Udemy beginner basics | 45m [Touseef Arif course](../../topics/programming_languages/mojo/Udemy_Courses.md); 30m syntax drills (`def`/`fn`, types); 15m compare to Python you use daily |
| **Oct** | Python interop | 30m interop docs; 45m port one NumPy-ish loop to Mojo; 15m time or profile note |
| **Nov** | Micro-kernel | 30m SIMD/GPU-oriented Mojo patterns; 45m matmul or reduce kernel; 15m document vs NumPy baseline |
| **Dec** | Capstone | 60m hot-path kernel callable from Python notebook used in survival/DL experiments; 30m link to [AI_Plan.md](./AI_Plan.md) Dec threads |

**Sources:** [Udemy_Courses.md](../../topics/programming_languages/mojo/Udemy_Courses.md)

---

## Cross-links

- Research (survival + deep learning): [AI_Plan.md](./AI_Plan.md)
- Dec capstones across Rust / TS / Mojo should share one small experiment thread where possible (e.g. survival feature pipeline)
