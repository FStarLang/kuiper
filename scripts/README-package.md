# Kuiper (binary package)

This is a self-contained [Kuiper](https://github.com/FStarLang/kuiper) binary
package. It bundles everything needed to write, verify, extract, and compile
GPU kernels in Kuiper:

- the F\* and Karamel toolchain (`inst/bin/fstar.exe`, `inst/bin/krml`),
- Z3 (bundled under `inst/lib/fstar/`),
- the verified Kuiper library (`obj/*.checked`) and its extraction plugin,
- all sources, CUDA headers (`include/`), and the build system.

See `BUILD_INFO` for the exact commits and platform this package was built from.

## Requirements

- A 64-bit Linux or macOS system matching this package's platform (see `BUILD_INFO`).
- GNU Make (>= 4.4) and a POSIX shell.
- `nvcc` (the CUDA toolkit) and an NVIDIA GPU are only needed to *compile* and
  *run* kernels. You can verify and extract to CUDA source without them.

On **macOS**, the build system needs the GNU tools rather than the BSD ones
that ship with the system. Install them with

```bash
./setup-mac.sh     # brew install gnu-sed coreutils gnu-indent make
```

No `PATH` juggling is needed: the makefiles look the tools up under their
Homebrew names (`gsed`, `grealpath`, `gindent`, `gnproc`). The one exception is
`make` itself, which Homebrew installs as `gmake` — use that instead of the
system `make` (3.81), which cannot parse these makefiles:

```bash
gmake -j$(sysctl -n hw.ncpu)
```

No OPAM/OCaml installation is required: the toolchain ships prebuilt.

## Quick start

From the root of this package:

```bash
# Verify + extract the whole library to CUDA (and build tests if nvcc is present)
make -j$(nproc)

# Verify only (no extraction/compilation)
make verify

# Verify a single file, using the bundled toolchain
./fstar.sh src/examples/Kuiper.Example.Add.fst
```

The `.packaged` marker in this directory tells the build system that the
toolchain, checked library, and extraction plugin are prebuilt, so `make` will
**not** try to rebuild F\*/Karamel from source.

## Writing a new kernel

1. Add your module under `src/` (e.g. `src/klas/Klas.MyKernel.fst`). Kernels that
   should be extracted to CUDA typically live in `src/klas/` or `src/examples/`
   and instantiate the polymorphic library to concrete element types.
2. Verify it:
   ```bash
   ./fstar.sh src/klas/Klas.MyKernel.fst
   ```
3. Extract it to CUDA and (if `nvcc` is available) compile:
   ```bash
   make obj/Klas_MyKernel.cu     # F* -> .krml -> .cu/.h
   ```
   The generated `.cu`/`.h` land in `obj/`. Compile them against the headers in
   `include/`.

The verified library is already checked (`obj/*.checked`), so re-verification
only touches your new files.

## Using the toolchain directly

`fstar.sh` and `krml.sh` wrap the bundled binaries with the correct include
paths and flags. You can also call the binaries directly:

```bash
./inst/bin/fstar.exe --version
./inst/bin/krml -version
```
