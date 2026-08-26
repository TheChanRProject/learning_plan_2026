# Day 3 — Lifetimes

> **Goal:** Read and write lifetime annotations so returned references stay valid.

**Date:** Fri Aug 28, 2026

## What you'll learn

- Why lifetimes exist (refs must not outlive data)
- `'a` syntax on fns and structs
- Elision rules (when you can omit)
- Hook into advanced lifetimes (Grider bonus section) — skim only

## Prerequisites

[02 — Ownership & Borrowing](./02-ownership-borrowing.md)

## Watch / read

- Udemy: [Grider — Lifetimes Explored](../Udemy_Courses.md) (+ skim Advanced Lifetimes)

---

## Mental model

Lifetime = **region of code where a reference is valid**. Annotation links input refs to output refs so the compiler can prove safety. Most code needs no explicit `'a` — elision covers obvious cases.

```rust
// Output lives as long as the shorter of a and b
fn longer<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() >= b.len() { a } else { b }
}

struct Excerpt<'a> {
    text: &'a str,
}
```

When you need `'a`:

- Function returns a reference derived from inputs
- Struct holds a reference
- Multiple input lifetimes and the relationship is not obvious

---

## Day 3 exercise

1. `cargo new day03_lifetimes`
2. Implement `longer` as above; call with two string literals and two `String`s (via `.as_str()`)
3. Add `struct Excerpt<'a>` holding a slice of a local `String` — make it fail if you return `Excerpt` from a fn that owns the `String`, then fix by returning owned `String` instead

**Done when:** you can explain one lifetime error in plain English.

**Weekend off** (Aug 29–30). Resume Mon → [04 — Enums & Pattern Matching](./04-enums-pattern-matching.md)
