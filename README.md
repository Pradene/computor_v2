# computor_v2

A mathematical expression interpreter written in Rust. Built as part of the 42 School curriculum, computor_v2 is an interactive REPL that can evaluate expressions, solve polynomial equations, and handle variables, functions, and matrices.

## Features

- **Interactive REPL** with command history and line editing (via `rustyline`)
- **Variable assignment** — store and reuse values: `x = 42`
- **Function definition** — define and call custom functions: `f(x) = x^2 + 3`
- **Polynomial solver** — solve equations up to degree 2: `x^2 + 3x - 4 = 0 ?`
- **Matrix support** — basic matrix operations and arithmetic
- **Expression evaluation** — handles operator precedence, parentheses, and rational numbers

## Usage
```bash
cargo build --release
./target/release/computor_v2
```

Then type expressions directly into the prompt:
```
> x = 5
> f(x) = x^2 + 2*x + 1
> f(3) = ?
16
> 2*x^2 - 5*x + 2 = 0 ?
Discriminant: 9
Solutions: x = 2, x = 0.5
```

## Implementation

The interpreter is built from scratch in Rust with no math-specific external libraries:

- **Lexer** — tokenizes raw input into numbers, operators, identifiers
- **Parser** — builds an AST respecting operator precedence
- **Evaluator** — walks the AST, resolves variables/functions, computes results
- **Polynomial solver** — discriminant-based quadratic resolution with complex number output when Δ < 0

## Requirements

- Rust 2021
