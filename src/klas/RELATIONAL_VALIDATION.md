# Relational validation of matmul kernels

**Goal:** Show that several differently-bracketed floating-point matmul CUDA
kernels are *algebraically equivalent* (ignoring floating-point rounding) by
giving each a **verified Kuiper witness** that is

1. **bit-for-bit identical** to the CUDA kernel, and
2. proves the **same precise spec**: `eC' %~ MS.matmul rA rB`
   (the f32 result approximates the order-independent real-valued matmul).

## The five kernels, each with a verified witness

| # | Kernel    | Accumulation order            | Witness source                         |
|---|-----------|-------------------------------|----------------------------------------|
| 1 | `imp1.cu` | forward (i = 0 .. k-1)        | instantiates existing `GEMM.Naive`     |
| 2 | `imp2.cu` | reverse (i = k-1 .. 0)        | from-scratch single-thread             |
| 3 | `imp3.cu` | tiled (16, forward)           | from-scratch single-thread             |
| 4 | `imp4.cu` | Kahan-compensated             | instantiates existing `GEMM.Naive3`    |
| 5 | `imp5.cu` | tiled (16) + Kahan over tiles | from-scratch single-thread             |
| 6 | `imp6.cu` | shared-memory tiled, forward  | instantiates `GEMM.Naive` (= same as 1)|
| 7 | `imp7.cu` | shared-memory tiled, forward, **transposed store** | from-scratch single-thread |

Witness sources: `src/klas/KWitness{1..7}.fst` (+ `.fsti`).

Kernels 1-5 prove the identical spec `eC' %~ MS.matmul rA rB`. Kernel 6 is a
shared-memory tiled GEMM whose accumulation is nonetheless plain forward (a single
running `sum` across tiles, no per-tile partial), so it proves the *same* spec and
is bit-equivalent to the naive forward witness (kw1) — demonstrating that tiling
for memory locality does not change the floating-point result. Kernel 7 is kernel 6
with a transposed store (`C[c*m+r]`, the correct transpose into the n×m output for
any shape), so it proves `eC' %~ mtranspose (MS.matmul rA rB)`.

## Key results

- **All five witnesses verify** with no `admit` / `assume` / `lax` — fully
  proved (each: "All verification conditions discharged successfully").
- All five `.fsti` spec bodies are **byte-identical** except the size
  precondition — making the "same spec" claim visually obvious.
- A unified driver **`bitcmp.cu`** (10 kernels) confirms each `impN ≡ kwN`
  bit-for-bit across many shapes, while the kernels mutually differ in their
  floating-point results (FP non-associativity). **Suite PASSES.**
  - `imp5` matches `imp3` only when `k <= 16` (≤ 2 tiles), where tiled and
    tiled+Kahan coincide; otherwise all five bracketings differ.

## Notable engineering

- Caught a B-indexing bug in `imp3.cu` (`b[k0 + k1*n + col]` should be
  `b[(k0+k1)*n + col]`).
- Built the reverse, tiled, and tiled+Kahan reductions from scratch — no
  existing Kuiper kernel reduces in those orders. Each uses a single block of a
  single thread that owns all of A, B, C (no permission splitting).
- KWitness5 uses the library `Kuiper.Kahan.kahan_sum` combinator to combine
  per-tile partials; `cell_real` proves the sum of tile-sums equals the matmul
  cell over the reals.
- Discovered the "isolate a stubborn lemma into a small
  `inline_for_extraction noextract fn`" pattern: `value_approx_cell` would not
  discharge inside the KWitness5 kernel loop (the large loop-invariant context
  defeated SMT) even with both preconditions asserted; extracting the per-cell
  work into `cell_value` discharged it in a minimal context, with identical
  extraction.

## Building and running the driver

```bash
make obj/KWitness1.cu obj/KWitness2.cu obj/KWitness3.cu obj/KWitness4.cu obj/KWitness5.cu
nvcc -O2 -arch=native -I include -I obj -o bitcmp bitcmp.cu \
     obj/KWitness1.cu obj/KWitness2.cu obj/KWitness3.cu obj/KWitness4.cu obj/KWitness5.cu

./bitcmp                          # default suite over several shapes
./bitcmp A B [m] [n] [k] [seed]   # compare a single pair
                                  # (A,B in imp1..imp5, kw1..kw5)
```

## Bottom line

Five distinct floating-point bracketings (forward / reverse / tiled / Kahan /
tiled+Kahan), all provably equal to the one real-valued matmul.
