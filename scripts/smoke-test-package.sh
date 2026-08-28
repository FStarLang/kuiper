#!/usr/bin/env bash

# Smoke-test a Kuiper binary package produced by mk-package.sh.
#
# Extracts the tarball into a scratch directory and, using ONLY the bundled
# toolchain (no opam env, no submodules), checks that:
#   1. the bundled toolchain and formatter execute,
#   2. the bundled F* + Z3 can verify an existing example,
#   3. a brand-new kernel module can be written and verified,
#   4. a kernel can be extracted and formatted as CUDA.
#
# Usage: scripts/smoke-test-package.sh <package.tar.gz>

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <package.tar.gz>" >&2
  exit 1
fi

PKG="$(realpath "$1")"
[ -f "$PKG" ] || { echo "error: $PKG not found" >&2; exit 1; }

WORK="$(mktemp -d --tmpdir kuiper-smoke.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo ">>> Extracting $PKG"
tar xzf "$PKG" -C "$WORK"
cd "$WORK/kuiper"

# Deliberately run without opam: a proper package must be self-contained.
echo ">>> (1) Checking bundled executables"
./inst/bin/fstar.exe --version
./inst/bin/krml -version || true
./inst/bin/clang-format --version
test -f ./inst/share/licenses/clang-format/LICENSE.md

echo ">>> (2) Verifying a bundled example"
./fstar.sh src/examples/Kuiper.Example.Add.fst

echo ">>> (3) Verifying a brand-new kernel"
cat > src/examples/Kuiper.Smoke.Test.fst <<'EOF'
module Kuiper.Smoke.Test
#lang-pulse
open Kuiper

fn smoke_incr (x : f32) returns f32 {
  add x one;
}
EOF
./fstar.sh src/examples/Kuiper.Smoke.Test.fst

# macOS needs GNU make (as gmake) since the system make is 3.81. See
# setup-mac.sh.
if [ "$(uname -s)" = Darwin ]; then
  MAKE="${MAKE:-gmake}"
else
  MAKE="${MAKE:-make}"
fi

echo ">>> (4) Extracting a kernel to CUDA"
"$MAKE" obj/Kuiper_Example_Add.cu
test -f obj/Kuiper_Example_Add.cu
echo ">>> Generated CUDA:"
head -n 20 obj/Kuiper_Example_Add.cu

echo ">>> Smoke test passed."
