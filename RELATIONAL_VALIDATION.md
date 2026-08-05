# Relational validation of matmul kernels

**Goal:** Show that a range of differently-implemented floating-point matmul CUDA
kernels are *algebraically equivalent* (ignoring floating-point rounding) by
giving each a **verified Kuiper witness** that is

1. **bit-for-bit identical** to the CUDA kernel, and
2. proves a **precise real-valued spec**. For most kernels this is
   `eC' %~ MS.matmul rA rB` (the f32 result approximates the order-independent
   real matmul); the two variant kernels prove a simple function of it — a
   transpose (`mtranspose (MS.matmul rA rB)`, #7) or the general affine SGEMM
   result (`α·(MS.matmul rA rB) + β·C`, #8).

## The eight kernels, each with a verified witness

| # | Kernel    | Accumulation order            | Witness source                         |
|---|-----------|-------------------------------|----------------------------------------|
| 1 | `imp1.cu` | forward (i = 0 .. k-1)        | instantiates existing `GEMM.Naive1`     |
| 2 | `imp2.cu` | reverse (i = k-1 .. 0)        | from-scratch single-thread             |
| 3 | `imp3.cu` | tiled (16, forward)           | from-scratch single-thread             |
| 4 | `imp4.cu` | Kahan-compensated             | instantiates existing `GEMM.Naive3`    |
| 5 | `imp5.cu` | tiled (16) + Kahan over tiles | from-scratch single-thread             |
| 6 | `imp6.cu` | shared-memory tiled, forward  | instantiates `GEMM.Naive1` (= same as 1)|
| 7 | `imp7.cu` | shared-memory tiled, forward, **transposed store** | from-scratch single-thread |
| 8 | `imp8.cu` | 2D block-tiling SGEMM, forward, **α·(A·B)+β·C** | instantiates `GEMM.Naive1` (custom combine) |

Witness sources: `src/klas/KWitness{1..8}.fst` (+ `.fsti`). Kernel 7 also has an
alternative witness `KWitness7b` (transpose by zero-cost view shift).

Kernels 1-5 prove the identical spec `eC' %~ MS.matmul rA rB`. Kernel 6 is a
shared-memory tiled GEMM whose accumulation is nonetheless plain forward (a single
running `sum` across tiles, no per-tile partial), so it proves the *same* spec and
is bit-equivalent to the naive forward witness (kw1) — demonstrating that tiling
for memory locality does not change the floating-point result. Kernel 7 is kernel 6
with a transposed store (`C[c*m+r]`, the correct transpose into the n×m output for
any shape), so it proves `eC' %~ mtranspose (MS.matmul rA rB)`. Kernel 8 is the
optimized 2D block-tiling SGEMM (Boehm's kernel): its register-tiled accumulation is
still the forward dot product, followed by the general affine combine, so it proves
`eC' %~ α·(MS.matmul rA rB) + β·C_old` — reusing `GEMM.Naive1` with the per-cell
combine instantiated to `fun old prod -> α·prod + β·old`.

## Key results

- **All eight witnesses verify** (nine modules, counting `KWitness7b`) with no
  `admit` / `assume` / `lax` — fully proved (each: "All verification conditions
  discharged successfully").
- The **`.fsti` spec bodies of kernels 1–6 are byte-identical** except the size
  precondition — making the "same spec" claim visually obvious. Kernels 7 and 8
  deliberately differ: #7 ensures `mtranspose (MS.matmul rA rB)` and #8 ensures
  the affine `gemm_real α β rA rB C`.
- A unified driver **`bitcmp.cu`** covers kernels **1–5** (10 kernels: `imp1..5`
  + `kw1..5`) and confirms each `impN ≡ kwN` bit-for-bit across many shapes,
  while the kernels mutually differ in their floating-point results (FP
  non-associativity). **Suite PASSES.** Kernels 6, 7, and 8 are validated by
  standalone bit-equivalence harnesses (see their notes below), not the driver.
  - `imp5` matches `imp3` only when `k <= 16` (a single 16-wide tile), where the
    Kahan compensation has nothing to correct; otherwise all five bracketings
    differ.

## Notable engineering

- **Keep the public witness and specification audit surface minimal.** Temporary
  buffers that have no semantic meaning at the top level should be allocated and
  freed inside the verified function rather than supplied by its caller. They
  should not appear as public parameters, preconditions, or postconditions. For
  example, the UGS/SwiGLU witness internally owns both its FP32 projection
  accumulator and FP16 rounded-projection buffer; its public contract exposes
  only the input matrices and final output state. Exact intermediate properties
  remain in the verified component contracts rather than the top-level contract.
  This removes irrelevant runtime and ghost state from the portion a human must
  review.
- **Prefer Kuiper's `sz`-to-`nat` coercion in specifications.** A value of type
  `sz`/`szp` can usually be written directly in arithmetic, chest dimensions,
  bounds, and pure specifications. Use explicit `SZ.v` only when coercion does
  not apply or when distinguishing the runtime `sz` representation is genuinely
  important. Avoiding redundant `SZ.v` calls makes proof contracts substantially
  smaller and easier to audit.

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
- **KWitness6** (shared-memory tiled) accumulates into a single running `sum`
  across tiles — plain forward order — so it reuses the naive forward kernel and
  is bit-equivalent to kw1: tiling for locality does not change the result.
- **KWitness7** proves the transposed-store kernel; caught that the original
  `C[c*n+r]` store was a correct transpose only for square output and fixed it to
  `C[c*m+r]` (transpose of an m×n product is n×m). `KWitness7b` gives an
  alternative that needs no transpose kernel or copy at all: it matmuls into a
  **col-major view** of the output (`Kuiper.Ghost.TensorTranspose`) — a zero-cost
  ghost view shift that erases, leaving a single matmul kernel.
- **KWitness8** reuses the naive kernel with a custom per-cell combine
  `fun old prod -> α·prod + β·old` for the affine SGEMM; the `approx2` refinement
  on the real combiner needed a small explicit congruence lemma
  (`a_mul`/`a_add`/`to_real_ok`) since SMT will not beta-reduce the combiner
  lambdas on its own.

## Building and running the driver

```bash
make obj/KWitness1.cu obj/KWitness2.cu obj/KWitness3.cu obj/KWitness4.cu obj/KWitness5.cu
nvcc -O2 -arch=native -I include -I obj -o bitcmp bitcmp.cu \
     obj/KWitness1.cu obj/KWitness2.cu obj/KWitness3.cu obj/KWitness4.cu obj/KWitness5.cu

./bitcmp                          # default suite over several shapes
./bitcmp A B [m] [n] [k] [seed]   # compare a single pair
                                  # (A,B in imp1..imp5, kw1..kw5)
```

Kernels 6, 7, and 8 are checked with standalone one-off harnesses (each compiles
`impN.cu` against the extracted `obj/KWitnessN.cu` and bit-compares outputs across
shapes — and, for #8, across `alpha`/`beta` values).

## Bottom line

Eight differently-implemented f32 matmul kernels — forward, reverse, tiled,
Kahan, tiled+Kahan, shared-memory tiled, transposed-store, and a fully optimized
2D block-tiling SGEMM with α/β — each with a verified Kuiper witness that is
bit-for-bit identical and proves a precise real-valued spec (the order-independent
matmul, or a transpose / affine function of it).
