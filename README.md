# Kuiper: GPU Kernel Verification in Pulse

[![CI](https://github.com/FStarLang/kuiper/actions/workflows/ci.yml/badge.svg)](https://github.com/FStarLang/kuiper/actions/workflows/ci.yml)

Kuiper is a language for safe and verified efficient CPU/GPU programming.  It
builds over F* and Pulse to provide a language based on dependent types and
separation logic to model the intricacies of GPU programming.  All Kuiper
programs are by construction safe, data-race free, and optionally verified to be
functionally correct.

Several GPU features are supported by Kuiper, including: shared memory,
barriers, atomics, vectorized memory operations, and tensor cores. Kuiper also
safely models CPU-GPU interaction via a notion of located resources, as well as
asynchronous kernels launches.

There are also libraries to make programming with matrices more convenient, by,
e.g., abstracting layouts and matrix tiling. This allows most algorithms to be
written in "layout-polymorphic" style, where they can later be instantiated to
any layout of choice.

Kuiper currently only generates CUDA code.

**NOTE**: Some modules are still work in progress and therefore
contain admitted proofs.

## Publications

- **Kuiper: Correct and Efficient GPU Programming with Dependent Types and Separation Logic**
*Guido Martínez, Bastian Köpcke, Jonáš Fiala, Gabriel Ebner, Tahina Ramananandro, Michel Steuwer, Tyler Sorensen, Nikhil Swamy*.
PLDI 2026. https://doi.org/10.1145/3808280

- **The Next Frontier for AI-Generated Kernels: Correctness**
*Guido Martínez, Tyler Sorensen*.
PAgE 2026. https://doi.org/10.1145/3819802.3820580

## Using Verified Kernels

You can find the current set of verified kernels in the `dist/` directory, which
contains only the resulting CUDA code. This snapshot is updated routinely from
the verified kernels in `src/`, but may at times be slightly stale.

There are test drivers for some of these files in `test/`. You also need to
include the relevant Kuiper header files in `include/`.

## Getting Started

### devcontainer (codespace)

The easiest way to get started is via the devcontainer. Open the repository in a
GitHub Codespace or in VS Code with the Dev Containers extension. The container
includes OCaml, OPAM, and Z3 pre-installed.

Once the container starts, submodules are fetched automatically. You then need to
build F\* and Karamel:

```bash
eval $(opam env)
make prepare       # builds F* and Karamel (~10 min with -j)
```

The [F\* VS Code extension](https://github.com/FStarLang/fstar-vscode-assistant/)
is included. Use `Ctrl+.` to verify the file at the cursor position.

### Requirements
- OCaml 5.4.0, OPAM, and some packages. (Other OCaml versions may work, but this is the one we test, YMMV.)
- Z3 version 4.13.3
- NVCC (if you wish to _compile_ the kernels)
- An Nvidia GPU (if you wish to _run_ the kernels)

### Manual Setup

First, clone the Kuiper repository with submodules:
```bash
git clone https://github.com/FStarLang/kuiper
cd kuiper
git submodule update --init --recursive
```

You can use a script provided by F* to set up Z3 in your system:
```
sudo ./FStar/.scripts/get_fstar_z3.sh /usr/local/bin
```
otherwise, you can grab Z3 4.13.3 from [its
releases](https://github.com/Z3Prover/z3/releases/tag/z3-4.13.3) and make it
available in your PATH (as `z3` or `z3-4.13.3`).

If you do not have OCaml 5.4.0 installed, you can run the following commands
to set it up.
```
sudo apt-get install opam
opam init --compiler=5.4.0
```
Then make sure the necessary packages are installed:
```
opam install batteries zarith stdint yojson dune menhir menhirLib pprint sedlex ppxlib process ppx_deriving ppx_deriving_yojson memtrace visitors uucp wasm fix mtime
```

### Building

Kuiper includes F\* and Karamel as submodules. First, build them:

```bash
eval $(opam env)
make prepare -j$(nproc)
```

Then build Kuiper itself:

```bash
make -j$(nproc)    # verify all files, extract to CUDA, compile tests
```

This will verify every file in `src/`, build the extraction plugin, extract all
examples into CUDA files, and (if `nvcc` is available) build the test drivers.
All build artifacts go into `obj/`. On a modern machine (e.g., AMD 7950X with
`-j32`), verification takes roughly 10 minutes wall-clock.

There is a simple `./configure` script that detects if nvcc is installed, and
whether it supports tensor cores.  If nvcc is not installed, `make` will stop
after code generation. If tensor cores are not enabled, tensor core examples
will be skipped. You can edit `.configure.output` manually if need be.

Other useful targets:

| Target | Description |
|---|---|
| `make verify` | Verify only (no extraction or compilation) |
| `make minimal` | Verify + extract without TensorCore modules |
| `make test` | Run tests and compare against expected output |
| `make accept` | Accept current test output as new expected |
| `make dist` | Update the `dist/` snapshot from freshly extracted code |
| `make lint` | Run C and F\* linters |
| `make list-admits` | Find any `admit`/`assume`/`magic` in source |
| `make wc` | Line counts for F\* source and generated CUDA |

To verify a single file:
```bash
./fstar.sh src/path/to/Module.fst
```

### Project Structure

Kuiper source lives under `src/`. The core library (`src/lib/kuiper/`) provides
the DSL primitives: arrays, barriers, atomics, shared memory, tensor cores, and
separation-logic combinators. Supporting libraries handle matrix data structures
and layouts (`data/`), pure functional specifications (`spec/`), array views
(`views/`), and ghost utilities (`ghost/`).

Some kernels are written in highly-polymorphic style and later instantiated.
Modules in `src/lib/kernel/` are polymorphic over an element type `et` and perhaps
layouts, tile sizes, etc.  Modules in `src/klas/` are instantiations that
bind `et` to concrete types; these are what actually get extracted to CUDA.

Some kernels have a large number of instantiations, so we generate them via a
`.fst.sh` script. Any file with this extension gets run and piped into the
proper `.fst`.

Also:
- `extraction/`: contains the F* extraction plugin
- `include/`: C/CUDA headers, needed to compile Kuiper code
- `test/`: CUDA test drivers with expected-output files
- `dist/`: a CUDA snapshot of the verified kernels
- `bench/`: benchmarking infrastructure
