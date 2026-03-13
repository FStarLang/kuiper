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

Requirements:
- OCaml, OPAM, and some packages
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

### Run the Benchmarks
(UPDATE)
To run the benchmarks and test the compiled kernels, execute the provided script:
```bash
./scripts/bench.sh
```
This will run the benchmarks and provide performance and correctness outputs for the compiled code.
