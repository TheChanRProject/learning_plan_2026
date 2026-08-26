# Day 5 — Modules & Errors

> **Goal:** Organize crates with modules; propagate errors with `Result` and `?`.

**Date:** Tue Sep 1, 2026

## What you'll learn

- `mod`, `use`, `pub`, file layout (`src/lib.rs` / modules)
- `Result<T, E>`, `?` operator
- `thiserror` / `anyhow` — know they exist; use std first today
- Converting error types at boundaries

## Prerequisites

[04 — Enums & Pattern Matching](./04-enums-pattern-matching.md)

## Watch / read

- Udemy: [Grider — Project Architecture + Errors and Results](../Udemy_Courses.md)

---

## Mental model

**Modules** hide internals; `pub` is the API. **Errors** are values (`Result`), not exceptions. `?` early-returns `Err` up the call stack.

```rust
use std::fs;
use std::io;

fn read_len(path: &str) -> Result<usize, io::Error> {
    let s = fs::read_to_string(path)?;
    Ok(s.len())
}

fn main() {
    match read_len("Cargo.toml") {
        Ok(n) => println!("{n} bytes"),
        Err(e) => eprintln!("fail: {e}"),
    }
}
```

Layout sketch:

```
src/
  main.rs      // binary entry
  lib.rs       // library root (optional)
  parse.rs     // mod parse;
```

---

## Day 5 exercise

1. `cargo new day05_errors`
2. Add `src/parse.rs` with `pub fn parse_u32(s: &str) -> Result<u32, String>`
3. `main` reads a CLI arg (or hardcode), uses `?` via a `fn run() -> Result<(), String>`
4. Return distinct error strings for empty vs non-numeric input

**Done when:** bad input prints clear `Err`; good input prints number; modules compile.

**Next:** [06 — Iterators, Generics & Traits](./06-iterators-generics-traits.md)
