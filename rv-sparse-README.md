# RV-Sparse: Sparse Matrix-Vector Multiplication

Implementation of `sparse_multiply()` for the RV-Sparse LFX coding challenge.

---

## Build & Run

```bash
gcc -lm -O2 -o run challenge.c
./run
```

Expected output: **7 / 7 tests passed**

---

## What it does

`sparse_multiply(A, rows, cols, x, y, csr)` performs three steps in one call:

**1. Compress** — scans the row-major dense matrix `A` and extracts non-zero
elements into Compressed Sparse Row (CSR) format using caller-provided buffers.

**2. Store** — fills the CSR struct:
- `values[k]`    — value of the k-th non-zero
- `col_idx[k]`   — its column index
- `row_ptr[i]`   — index into `values[]` where row `i` begins; `row_ptr[rows] = nnz`

**3. Multiply** — computes `y = A * x` directly from the CSR data in O(nnz) time.

### Critical constraint met
**Zero dynamic memory allocation.** No `malloc`, `calloc`, or `realloc` is called
anywhere inside `sparse_multiply()`. All buffers are pre-allocated by the caller
and passed in via the `CSRMatrix` struct.

---

## Complexity

| Phase    | Time          | Space  |
|----------|---------------|--------|
| Compress | O(rows × cols)| O(1) extra |
| Multiply | O(nnz)        | O(1) extra |

For a matrix with density `d`, `nnz = d × rows × cols`, so the multiply phase
is `d` times faster than naive dense multiplication.

---

## Test cases

| # | Description                    | Checks                          |
|---|--------------------------------|---------------------------------|
| 1 | Basic 3×3 sparse matrix        | Correctness + CSR structure     |
| 2 | All-zeros matrix               | nnz == 0, y == zero vector      |
| 3 | Fully dense matrix             | nnz == rows×cols                |
| 4 | Diagonal matrix                | nnz == rows, one nz per row     |
| 5 | Non-square 2×5                 | Handles cols > rows             |
| 6 | Negative & floating-point vals | Numerical correctness           |
| 7 | Large 100×100 (~5% density)    | Stress test, 500 non-zeros      |

All results are verified against a reference dense multiplication within a
tolerance of `1e-8`.

---

## Design decisions

- **Single-pass compression + multiply** — two tightly scoped loops keep the
  code cache-friendly and easy to reason about.
- **Standard CSR layout** — identical to SuiteSparse/MKL/cuSPARSE convention,
  making the output directly usable by other sparse libraries.
- **Exact zero test** (`val != 0.0`) — matches the challenge spec; for
  numerical matrices a small epsilon threshold could be added trivially.
- **No headers beyond `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<string.h>`** —
  self-contained, no external dependencies.

---

## License

MIT
