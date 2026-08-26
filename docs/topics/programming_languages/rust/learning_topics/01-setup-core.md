# Day 1 — Setup & Core Concepts

> **Goal:** Install Rust toolchain, create Cargo projects, use primitives, structs, and control flow.

**Date:** Wed Aug 26, 2026

## What you'll learn

- `rustup`, `cargo new`, `cargo run` / `check` / `test`
- Scalars, tuples, arrays, `String` vs `&str`
- Structs + `impl` methods
- `if`, `loop`, `while`, `for`

## Prerequisites

Read [00-roadmap](./00-roadmap.md). Have a terminal.

## Watch / read

- Udemy: [Stephen Grider — Foundations + Core Concepts](../Udemy_Courses.md)
- Optional: Codestars install/basics sections

---

## Mental model

Rust is **expression-oriented**. Almost everything returns a value. Types are static and checked before run. Cargo owns build + deps — treat it like `npm`/`pip` for binaries and libraries.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustc --version
cargo new hello_rust && cd hello_rust && cargo run
```

```rust
struct Point { x: f64, y: f64 }

impl Point {
    fn origin() -> Self { Self { x: 0.0, y: 0.0 } }
    fn dist(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
}

fn main() {
    let p = Point { x: 3.0, y: 4.0 };
    println!("dist={}", p.dist());
}
```

---

## Day 1 exercise

1. `cargo new day01_core`
2. Define a `User { name: String, age: u32 }` with `impl` that returns a greeting string
3. In `main`, create two users, print greetings in a `for` over a `Vec`
4. `cargo check` clean; commit locally if you want

**Done when:** project builds, greeting prints, you can explain `String` vs `&str` in one sentence.

**Next:** [02 — Ownership & Borrowing](./02-ownership-borrowing.md)
