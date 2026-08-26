# Day 7 — Async & Concurrency

> **Goal:** Spawn work with threads and `async`/await; pass messages on channels — base for agents.

**Date:** Thu Sep 3, 2026

## What you'll learn

- `std::thread::spawn` + `JoinHandle`
- `mpsc` channels (message passing > shared mut)
- `async fn`, `.await`, Tokio runtime basics
- When threads vs async (CPU vs many I/O waits)

## Prerequisites

[06 — Iterators, Generics & Traits](./06-iterators-generics-traits.md)

## Watch / read

- Udemy: [Codestars — concurrency / async sections](../Udemy_Courses.md)
- Skim Tokio tutorial "Hello Tokio" if course thin on async

---

## Mental model

**Threads** = OS parallelism. **Async** = cooperative concurrency on few threads — great for network/agent tool calls. Prefer **channels** over `Mutex` until you must share state.

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        tx.send(String::from("ping")).unwrap();
    });
    println!("got {}", rx.recv().unwrap());
}
```

```rust
// Cargo.toml: tokio = { version = "1", features = ["full"] }
#[tokio::main]
async fn main() {
    let h = tokio::spawn(async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        42
    });
    println!("{}", h.await.unwrap());
}
```

---

## Day 7 exercise

1. `cargo new day07_async` — add Tokio dep
2. Thread version: worker sends 3 messages on `mpsc`; main prints them
3. Async version: `tokio::spawn` two sleeps; `tokio::join!` both; print elapsed roughly
4. One-sentence note in a comment: when you'd pick async for an LLM tool-calling agent

**Done when:** both thread + async paths run; no data races (compiler quiet).

**Next:** [08 — DataFusion](./08-datafusion-spark-analog.md)
