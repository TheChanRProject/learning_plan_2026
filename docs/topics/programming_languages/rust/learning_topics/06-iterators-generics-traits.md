# Day 6 — Iterators, Generics & Traits

> **Goal:** Process collections with iterators; write generic functions and trait-based APIs.

**Date:** Wed Sep 2, 2026

## What you'll learn

- `Iterator` adapters: `map`, `filter`, `fold`, `collect`
- Generic type params + trait bounds
- Defining and implementing traits
- `impl Trait` in argument / return position (skim)

## Prerequisites

[05 — Modules & Errors](./05-modules-errors.md)

## Watch / read

- Udemy: [Grider — Iterator Deep Dive + Generics and Traits](../Udemy_Courses.md)
- Codestars: matching generics/traits sections if Grider feels thin

---

## Mental model

**Iterators** = lazy pipelines over items. **Generics** = code parameterized by type. **Traits** = shared behavior (like interfaces). Bounds say "T must be able to…".

```rust
fn sum_even(xs: &[i32]) -> i32 {
    xs.iter().copied().filter(|n| n % 2 == 0).sum()
}

fn largest<T: PartialOrd + Copy>(xs: &[T]) -> Option<T> {
    xs.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap())
}

trait Describe {
    fn describe(&self) -> String;
}

impl Describe for i32 {
    fn describe(&self) -> String { format!("int {self}") }
}
```

Prefer iterators over manual index loops for transforms — clearer and often as fast.

---

## Day 6 exercise

1. `cargo new day06_iter`
2. From a `Vec<String>`, collect uppercase names longer than 3 chars via iterator chain
3. Define trait `Score { fn score(&self) -> i32 }` for a `struct Model { name: String, accuracy: f64 }` (score = accuracy * 100 as i32)
4. Generic fn `best<T: Score>(items: &[T]) -> Option<&T>`

**Done when:** iterator chain has no `for` loop; trait method used by `best`.

**Next:** [07 — Async & Concurrency](./07-async-concurrency.md)
