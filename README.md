# Kuiper: GPU Kernel Verification in Pulse

Kuiper is a language for safe and verified efficient CPU/GPU programming.  It
builds over F* and Pulse to provide a language based on dependent types and
separation logic to model the intricacies of GPU programming.  All Kuiper
programs are by construction safe, data-race free, and optionally verified to be
functionally correct.

Several GPU features are supported by Kuiper: shared memory, barriers, atomics,
vectorized memory operations, and tensor cores. Kuiper also safely models
CPU-GPU interaction via a notion of located resources, as well as asynchronous
kernels launches.

There are also libraries to make programming with matrices more convenient, by,
e.g., abstracting layouts and matrix tiling. This allows most algorithms to be
written in "layout-polymorphic" style, where they can later be instantiated to
any layout of choice.

Kuiper currently only generates CUDA code.

## Publications

To appear at PLDI 2026.

## Using Verified Kernels

You can find the current set of verified kernels in the `dist/` directory, which
contains only the resulting CUDA code. This snapshot is updated routinely from
the verified kernels in `src/`, but may at times be slightly stale.

There are test drivers for some of these files in `test/`. You also need to
include the relevant Kuiper header files in `include/`.

## Getting Started

### Requirements
- OCaml 5.3.0, OPAM, and some packages
- Z3 version 4.13.3
- NVCC (if you wish to _compile_ the kernels)
- An Nvidia GPU (if you wish to _run_ the kernels)

First, clone the Kuiper repository with submodules included:
```bash
git clone https://github.com/FStarLang/kuiper
cd kuiper
git submodule init
git submodule update
```

You can use a script provided by F* to set up Z3 in your system:
```
sudo ./FStar/.scripts/get_fstar_z3.sh /usr/local/bin
```
otherwise, you can grab Z3 4.13.3 from [its
releases](https://github.com/Z3Prover/z3/releases/tag/z3-4.13.3) and make it
available in your PATH (as `z3` or `z3-4.13.3`).

If you do not have OCaml 5.3.0 installed, you can run the following commands
to set it up.
```
sudo apt-get install opam
opam init --compiler=5.3.0
```
Then make sure the necessary packages are installed:
```
opam install batteries zarith stdint yojson dune menhir menhirLib pprint sedlex ppxlib process ppx_deriving ppx_deriving_yojson memtrace visitors uucp wasm fix mtime
```

### Building

Kuiper includes F*, karamel, and Pulse as submodules. The top-level makefile
drives their build, so you shouldn't have to interact with them, except for the
occasional `git submodule update` if you update Kuiper.

Running `make` (use parallelism if possible) will build the submodules, verify
every file in `src/`, build the extraction plugin, extract all examples into
CUDA files, and build the test drivers. All build artifacts go into `obj/`.
Running `make test` will run the tests and fail if any of them fails.

There is a simple `./configure` script that detects if nvcc is installed, and
whether it supports tensor cores.  If nvcc is not installed, `make test` will
not run any tests. If tensor cores are not enabled, `make test` will skip the
examples that require them. This script is probably not very robust.  Please
file an issue if it does not work as expected.  You can edit `.configure.output`
manually if need be.

### Project Structure

Kuiper source lives under `src/`. The core library (`src/lib/kuiper/`) provides
the DSL primitives: arrays, barriers, atomics, shared memory, tensor cores, and
separation-logic combinators. Supporting libraries handle matrix data structures
and layouts (`data/`), pure functional specifications (`spec/`), array views
(`views/`), and ghost utilities (`ghost/`). Example kernels live in
`src/examples/`.

Some kernels are written in highly-polymorphic style and later instantiated.
Modules in `src/lib/poly/` are polymorphic over an element type `et` and perhaps
layouts, tile sizes, etc.  Modules in `src/lib/inst/` are instantiations that
bind `et` to concrete types; these are what actually get extracted to CUDA.

Some kernel have a large number of instantiations, so we generate them via a
`.fst.sh` script. Any file with this extension gets run and piped into the
proper `.fst`.

Also:
- `extraction/`: contains the F* extraction plugin
- `include/`: C/CUDA headers, needed to compile Kuiper code
- `test/`: CUDA test drivers with expected-output files
- `dist/`: a CUDA snapshot of the verified kernels
