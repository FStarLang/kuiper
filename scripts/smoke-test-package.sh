#!/usr/bin/env bash

# Smoke-test a Kuiper binary package produced by mk-package.sh.
#
# Extracts the tarball into a scratch directory and, using ONLY the bundled
# toolchain (no opam env, no submodules), checks that:
#   1. the bundled F* + Z3 can verify an existing example,
#   2. a brand-new kernel module can be written and verified,
#   3. (if `indent` is available) a kernel can be extracted to CUDA.
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
echo ">>> Toolchain versions"
./inst/bin/fstar.exe --version
./inst/bin/krml -version || true

echo ">>> (1) Verifying a bundled example"
./fstar.sh src/examples/Kuiper.Example.Add.fst

echo ">>> (2) Verifying a brand-new kernel"
cat > src/examples/Kuiper.Smoke.Test.fst <<'EOF'
module Kuiper.Smoke.Test
#lang-pulse
open Kuiper

fn smoke_incr (x : f32) returns f32 {
  add x one;
}
EOF
./fstar.sh src/examples/Kuiper.Smoke.Test.fst

if command -v indent >/dev/null 2>&1; then
  echo ">>> (3) Extracting a kernel to CUDA"
  make obj/Kuiper_Example_Add.cu
  test -f obj/Kuiper_Example_Add.cu
  echo ">>> Generated CUDA:"
  head -n 20 obj/Kuiper_Example_Add.cu
else
  echo ">>> (3) Skipping CUDA extraction ('indent' not installed)"
fi

echo ">>> Smoke test passed."
