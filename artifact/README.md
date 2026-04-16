# Kuiper artifact (PLDI 2026)

This artifact includes the full Kuiper framework (including F*, karamel, Pulse),
its extraction pipeline, and several examples. We provide three files:

- `kuiper-src.tar.gz`: a source package for Kuiper, including sources for
  F*, karamel, and Pulse. Only OCaml and Z3 are needed as dependencies
  to use this to verify and extract kernels (and nvcc to build them)
- `kuiper-pldi2026-docker.tar.gz`: a docker container with everything set up,
  pre-built dependencies and nvcc. Running `make` here should
  just work. This container also has a VS code webserver for
  easy interactive use.
- `kuiper-bench.tar.gz`: A "bench package" with CUDA sources for every extracted
  kernel. A reviewer can use this package to inspect the result and benchmark the
  kernels, later confirming that generating them via the source package or docker
  image produces the same results.

# Using docker + webserver

The docker container starts a vscode server on port 8080.

  1. To run this container you first need to load it:

    docker load < kuiper-pldi2026-docker.tar.gz

  2. Then you can start it:

    docker run -p 127.0.0.1:8080:8080 -it kuiper-pldi2026

  3. You can now access the vscode server by opening http://localhost:8080 in
  your browser. Note: if you run this command in a VS code terminal on a remote
  computer via SSH, VS code sets up a tunnel automatically so you can still open
  localhost.

  4. Open a terminal inside vscode using the keyboard shortcut `` ctrl+` ``

  5. Run `make -j$(nproc)`.  This will verify every Kuiper source file, build
  the extraction plugin, and extract the verified kernels to CUDA.  The
  F\*/Karamel/Pulse submodules are pre-built in the container; if you desire,
  you can run `make clean-full` to clean them.

  You should inspect the kernels for unsafe features like admit/magic/assume. We
  provide a `make list-admits` command to find them. The output is non-empty:
  the existing admits are intentional and related to the model.

  `make wc` provides a line count for F* and CUDA files (in dist/). It's
  currently around 35k vs 28.5k. (The paper will be edited to make it more
  explicit that the large number of CUDA lines comes from extracting many
  variants of the same kernels.)

  6. To check that everything works as expected, open a source file such as
  `src/examples/Kuiper.Example1.fst`.  Put the cursor at the end of the
  file and press `ctrl+.`, this instructs F\* to verify the file until the
  cursor position (i.e., the whole file).  After a few seconds, you should see a
  green bar on the left of the file, indicating that the file has been
  successfully verified.

  The [F* VS Code extension
  homepage](https://github.com/FStarLang/fstar-vscode-assistant/?tab=readme-ov-file#features-and-basic-usage-guide)
  contains a detailed explanation on how to use the F* mode in VS Code.

Optional: We include a `fstar.sh` script that sets all the right options for F\*
to verify a particular Kuiper file.  You can run `./fstar.sh
src/examples/Kuiper.Example1.fst` to verify a single file.

# Step-By-Step Instructions

  1. To verify all source files and extract CUDA code, run `make -j$(nproc)`.
  Note: this step can take a long time on slower machines. On an AMD 7950X
  processor, it takes about 10 minutes (-j32) wall clock time, and around
  50 minutes overall CPU time.

  2. If nvcc is available in the container, run `make test` to compile the
  extracted CUDA files and compare their output against expected results.
  (Note: running the compiled kernels requires an Nvidia GPU.)

  3. Open the files listed in the "Connection to Paper Text" section in this
  README and check that they contain the corresponding concepts from the paper.
  You can use go-to-definition to navigate around the code (ctrl+click).

## Structure of the Artifact

- `src/lib/kuiper/`: Core Kuiper library - arrays, refs, barriers, atomics,
  kernel combinators, separation logic combinators (`ForEvery`, `Bijection`,
  `Injection`), and math utilities.

- `src/lib/data/`: Matrix and array data structures, views, tiling, vectorized
  access.

- `src/lib/spec/`: Pure specifications (e.g., `Kuiper.Spec.GEMM` for matrix
  multiplication).

- `src/lib/poly/`: Polymorphic (layout-generic) kernel implementations.

- `src/lib/inst/`: Monomorphic instantiations of polymorphic kernels (these
  get extracted to CUDA).

- `src/examples/`: Example kernels from simple to complex.

- `extraction/`: OCaml plugin for custom F\*-to-Karamel extraction.

- `include/`: C/CUDA headers included in all generated code.

- `test/`: CUDA test drivers and `.output.expected` files.

- `dist/`: Snapshot of generated CUDA code from verified kernels.

There is also:

- `FStar/`, `karamel/`, `pulse/`: submodules providing the F\* compiler, Karamel
   extraction tool, and Pulse framework.  These are not contributions of this
   paper.

- `artifact/`: contains Dockerfile and scripts to generate this artifact.

- `obj/`: build artifacts (created by `make`).

## Connection to Paper Text

For every `.fst` file mentioned here, there is usually also a matching
`.fsti` with the interface for the module.

### Section 2

The naive matrix multiplication shown in Figure 2 corresponds to
`src/lib/poly/gemm/Kuiper.Poly.GEMM.Naive.fst`, with some minor differences
(e.g. it calls into a function to do the computation instead of having an
inlined while loop). The GEMM specification (dot products, matrix multiply) is
in `src/lib/spec/Kuiper.Spec.GEMM.fst`.

### Section 3

- s3.1 Located resources: (`loc`, `on`, `is_send_across`, `placeless`):
  `pulse/lib/core/Pulse.Lib.Loc.fst`.

- s3.2 forall+: `src/lib/kuiper/Kuiper.ForEvery.fst`.

- s3.3 Kernel launch semantics:
  `src/lib/kuiper/Kuiper.Kernel.Desc.fst` (kernel description type),
  `src/lib/kuiper/Kuiper.Kernel.Sync.fst` (reference semantics),
  `src/lib/kuiper/Kuiper.Kernel.Base.fst` (kernel launch functions).
  Barriers: `src/lib/kuiper/Kuiper.Barrier.fsti`.
  Shared memory: `src/lib/kuiper/Kuiper.SHMem.fst`.

### Section 4

- s4.1 Views, matrices, and tiling:
  Views: `src/lib/views/Kuiper.View.fst`.
  Virtual arrays: `src/lib/data/array/Kuiper.VArray.fst`.
  Matrices: `src/lib/data/Kuiper.Matrix.fst`.
  Tiling: `src/lib/data/Kuiper.Matrix.Tiling.fst`.
  Vectorized access: `src/lib/data/Kuiper.Matrix.Vectorized.fst`.

- s4.2 Approximate reasoning: 
  `src/lib/kuiper/Kuiper.Approximates.fst` (main module),
  `src/lib/kuiper/Kuiper.Approximates.Base.fst` (class definition).
  Example integer instance: `src/lib/kuiper/Kuiper.Approximates.U32.fst`.
  Examples floating-point instance: `src/lib/kuiper/Kuiper.Approximates.F32.fsti` (assumed).
  See `src/examples/Kuiper.IntApprox.fst` for an example of
  recovering a precise spec from an approximate one.

- s4.3 Tensor cores and 2D block-tiled matmul:
  Tensor core operations: `src/lib/kuiper/Kuiper.TensorCore.fst`.
  TensorCore2D kernel:
  `src/lib/poly/gemm/Kuiper.Poly.GEMM.TensorCore2D.fst` and
  `src/lib/poly/gemm/Kuiper.Poly.GEMM.TensorCore2D.KernelDesc.fst`.

### Section 5

All polymorphic (layout-generic) kernels are in `src/lib/poly/`; their
monomorphic instantiations (which get extracted to CUDA) are in
`src/lib/inst/`.

- **Naive GEMM**: `src/lib/poly/gemm/Kuiper.Poly.GEMM.Naive.fst`
- **BlockTiling2D GEMM**: `src/lib/poly/gemm/Kuiper.Poly.GEMM.BlockTiling2D.fst`
- **TensorCore2D GEMM**: `src/lib/poly/gemm/Kuiper.Poly.GEMM.TensorCore2D.fst`
- **Tree reduction**: `src/lib/poly/Kuiper.Poly.HReduce.fst`
- **Dot product**: `src/lib/poly/Kuiper.Poly.DotProduct.fst`
- **Stencil**: `src/lib/poly/Kuiper.Poly.Stencil.fst`
- **Softmax**: `src/lib/poly/Kuiper.Poly.Softmax.fst`

# Benchmarking (NVIDIA GPU needed)

The bench package contains everything needed to reproduce the performance
results.  Note: performance can vary significantly if the reviewers are using
different GPUs (we would actually appreciate it if you send us your results with
details about your system in your review). The procedure changes according to
which "bar" of the graphs is being measured.

Note, the makefile and configure script tries to detect if your setup supports
tensor cores, disabling them if not. If this does not work properly, please
manually comment out the inclusion of `tensorcores.h` in `kuiper.h`.

## K: kuiper kernels

The first step is obtaining a *tuned* implementation for your particular system.
We provide (configurable) scripts to automatically sweep the parameter space
and print results. The maximum perf number is the one that should be taken.
Running the script can take a *very* long time, possibly even days if the space
is too big.

We provide here the variants that gave the best performance for the GPUs we tried:

For BlockTiling2D:
- RTX 3050: `Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x8_cr 4579.162 GFLOPS`
- A6000: `Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_16x8_cr 20447.934 GFLOPS`
- A100: `Kuiper_GEMM_BlockTiling2D_g_gemm_f32_64x128x16_8x8_cr   17509.739 GFLOPS`

For TensorCore2D:
- RTX 3050: `Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x4 20854.286 GFLOPS`
- A6000: `Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x4 88071.211 GFLOPS`
- A100: `Kuiper_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x16_8x32x16_16x2 103542.872 GFLOPS`

To find the best-tuned variant in your configuration there are two scripts:
`./tuning/tune_b2d.sh` and `./tuning/tune_tc2d.sh`, for BlockTiling2D and
TensorCore2D respectively. They will take _all_ implementations present in the
relevant instantiation modules
(`src/lib/inst/gemm/Kuiper.GEMM.BlockTiling2D.fst` and
`src/lib/inst/gemm/Kuiper.GEMM.TensorCore2D.fst`) and measure them.  To change
the space, edit the relevant `.fst.sh` scripts.

## H: handwritten implementation

You can find the files in `handwritten-bench`. These are handwritten versions of
the kernels we've verified through Kuiper, but written idiomatically in CUDA
like a regular programmer would do. These kernels (`BlockTiling2D.cu` and
`TensorCore2D.cu`) have macros at the top to tweak the metaparameters. For each
kernel, the reviewer should plug in the parameters obtained from the (K) case
above and verify that performance is similar, at most a few percent lower or
higher.

The directory contains a Makefile.

## C: cuBLAS

In `bench/cublas` you'll find sources to measure the performance of cuBLAS doing
GEMMs over FP16 and FP32. The former may use tensor cores if available. The
performance numbers you get here may vary significantly compared to the previous
two, as cuBLAS implements a possibly very different algorithm. Each directory
has a makefile, you can build and run the program with `make && ./main`.

## Histogram

If you'd like to generate a histogram like Figure 5, you can run
`./tuning/histo.sh` on the output of a tuning run.
