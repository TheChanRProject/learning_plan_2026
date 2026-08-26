# Day 2 — Ownership & Borrowing

> **Goal:** Internalize move semantics, shared borrows (`&T`), and exclusive mutable borrows (`&mut T`).

**Date:** Thu Aug 27, 2026

## What you'll learn

- Ownership rules: one owner; drop when owner leaves scope
- Move vs copy (`Copy` types)
- `&T` (many) vs `&mut T` (one, exclusive)
- Slices (`&[T]`, `&str`)

## Prerequisites

[01 — Setup & Core](./01-setup-core.md)

## Watch / read

- Udemy: [Grider — Ownership and Borrowing](../Udemy_Courses.md)

---

## Mental model

**Move** transfers ownership. After move, old name is dead. **Borrow** lets you look (or mutate) without taking ownership. Compiler rejects aliases that would race.

```rust
fn takes_ownership(s: String) { println!("{s}"); } // s dropped here

fn borrows(s: &str) { println!("{s}"); }

fn main() {
    let name = String::from("rust");
    borrows(&name);           // ok, still own name
    takes_ownership(name);    // moved
    // println!("{name}");    // compile error
}
```

Rules cheat sheet:

| Want | Use |
|------|-----|
| Read without take | `&T` |
| Mutate without take | `&mut T` (no other borrows live) |
| Give away | move by value |
| Keep both | `.clone()` (pay cost) or rethink design |

---

## Day 2 exercise

1. `cargo new day02_ownership`
2. Write `fn longest_len(a: &str, b: &str) -> usize` — no ownership taken
3. Write `fn append_bang(s: &mut String)` that pushes `'!'`
4. Intentionally break the borrow checker once, read the error, fix it

**Done when:** you can predict move vs borrow before `cargo check`.

**Next:** [03 — Lifetimes](./03-lifetimes.md)
