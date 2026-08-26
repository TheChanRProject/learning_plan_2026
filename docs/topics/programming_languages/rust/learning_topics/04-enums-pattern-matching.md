# Day 4 — Enums & Pattern Matching

> **Goal:** Model variants with enums; handle `Option` and custom enums with `match` / `if let`.

**Date:** Mon Aug 31, 2026

## What you'll learn

- Enum variants with data
- `Option<T>` instead of null
- Exhaustive `match`, `if let`, `while let`
- Destructuring patterns

## Prerequisites

[03 — Lifetimes](./03-lifetimes.md)

## Watch / read

- Udemy: [Grider — Enums Unleashed](../Udemy_Courses.md)

---

## Mental model

Enum = **tagged union**. Compiler forces you to handle every variant. `Option` / `Result` are std enums — use them at API boundaries.

```rust
enum Msg {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
}

fn handle(m: Msg) {
    match m {
        Msg::Quit => println!("bye"),
        Msg::Move { x, y } => println!("go {x},{y}"),
        Msg::Write(s) => println!("{s}"),
    }
}

fn first(v: &[i32]) -> Option<i32> {
    v.get(0).copied()
}
```

Prefer `match` when multiple arms; `if let` for one interesting case.

---

## Day 4 exercise

1. `cargo new day04_enums`
2. Model a tiny CLI command enum (`Run`, `Stop`, `Status { verbose: bool }`)
3. Parse from `&str` → `Option<Cmd>`; `match` to print behavior
4. Chain `.map` / `.unwrap_or` on an `Option` without panic paths in happy case

**Done when:** exhaustive match compiles; no naked `unwrap` in exercise code.

**Next:** [05 — Modules & Errors](./05-modules-errors.md)
