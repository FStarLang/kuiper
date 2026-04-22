# Kuiper — Copilot Instructions

Kuiper is a DSL for programming and verifying safe GPU kernels, built on F\* and Pulse. Code is written in Pulse (a separation-logic language embedded in F\*), verified for properties like data race freedom and functional correctness, then extracted to CUDA via Karamel.

For detailed guidance on writing, reviewing, and debugging Kuiper kernel code, see [`.github/agents/kuiper-kernel-expert.agent.md`](.github/agents/kuiper-kernel-expert.agent.md).

## Build & Verify

The project uses Make. Two git submodules (`FStar`, `karamel`) must be initialized first, both on their `gpu` branches.

```bash
# First-time setup: build F* and Karamel
make prepare

# Full build: verify all modules, extract to CUDA, compile with nvcc (if available)
make

# Verify only (no extraction/compilation)
make verify

# Verify + extract without TensorCore modules
make minimal
```

### Verifying a single file

```bash
./fstar.sh path/to/Module.fst
```

This invokes F\* with all necessary include paths and flags. Verification can take 5–10 minutes for complex files. Use `ADMIT=1` to skip SMT queries during development:

```bash
make ADMIT=1
```

### Extracting and compiling

```bash
make extract-all    # F* → .krml → .cu via Karamel
make dist           # copy generated .cu/.h to dist/
```

### Tests

Tests compile extracted CUDA code and compare output against expected files in `test/`:

```bash
make test                    # run all tests (requires GPU)
make obj/Test_Kuiper_GEMM_Naive__F32.test   # run a single test
make accept                  # accept current output as new expected
```

### Linting

```bash
make lint           # both C and F* linting
make lint-fstar     # remove unused opens, trailing whitespace, check attrs
make lint-c         # indent test/*.cu files
```

## Architecture

### Pipeline

1. **Verify**: F\*/Pulse source in `src/` is type-checked and verified (`obj/*.checked`)
2. **Extract**: Verified modules are extracted to KreMLin IR (`obj/*.krml`) using an extraction plugin from `extraction/`
3. **Compile to CUDA**: KreMLin (Karamel) translates `.krml` → `.cu`/`.h`, post-processed by `scripts/fixup.sed` and `indent`
4. **Build**: `nvcc` compiles the generated CUDA code (`nvcc.mk`)

### Source layout

- **`src/lib/kuiper/`** — Core library: arrays, refs, barriers, atomics, kernels, separation logic combinators (`ForEvery`, `Bijection`, `Injection`), math utilities
- **`src/lib/data/`** — Matrix and array data structures, tiling, vectorized access
- **`src/lib/spec/`** — Pure specifications (e.g., `Kuiper.Spec.GEMM` for matmul)
- **`src/lib/kernel/`** — Polymorphic (type-generic) kernel implementations
- **`src/lib/inst/`** — Monomorphic instantiations of polymorphic kernels (these get extracted to CUDA)
- **`src/lib/views/`** — Array views for zero-copy sub-arrays
- **`src/lib/ghost/`** — Ghost (erased) utilities
- **`src/examples/`** — Example kernels (simple to complex)
- **`extraction/`** — OCaml plugin for custom F\*-to-Karamel extraction
- **`include/`** — C/CUDA headers (`kuiper.h`, atomics, vector ops, tensor cores) included in all generated code
- **`test/`** — CUDA test drivers and `.output.expected` files
- **`bench/`** — Benchmarking infrastructure

### Submodules

Both on their `gpu` branches — these are forked/branched versions with GPU-specific extensions:

- `FStar/` — F\* compiler, standard library, and Pulse separation logic framework
- `karamel/` — KreMLin compiler (F\* → C/CUDA)

## Key Conventions

- All Pulse files start with `#lang-pulse`
- The main `Kuiper` module re-exports core types and combinators — most files just `open Kuiper`
- Concrete functions must be `inline_for_extraction noextract`; only top-level kernels are non-inlined (with `__global__`)
- Typeclass instances used concretely (e.g., `clayout`) must also be `inline_for_extraction noextract`
- Interfaces (`.fsti`) need at least one `inline_for_extraction` item or extraction/inlining will fail; use `inline_for_extraction let () = ()` as a workaround
- Avoid `erased (natlt z)` — instead write `n:(erased nat){n < z}` or use `enatlt` (erased is invariant w.r.t. types, causing brittle typechecking)
- Module naming: `Kuiper.Foo.Bar` maps to file `Kuiper.Foo.Bar.fst` (dots in filenames, not directories)
- `.fst` = implementation, `.fsti` = interface; some `.fst` files are auto-generated from `.fst.sh` scripts
- F\* flags include `--ext kuiper` (Kuiper-specific compiler extension) and `--z3version 4.13.3`
