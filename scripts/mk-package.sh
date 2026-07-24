#!/usr/bin/env bash

# Assemble a self-contained Kuiper binary package.
#
# The package is a ready-to-use Kuiper tree bundling:
#   - the F* + Karamel toolchain (inst/), prebuilt
#   - Z3 4.13.3, bundled where F* expects it (inst/lib/fstar/z3-4.13.3/bin/z3)
#   - the verified Kuiper library (obj/*.checked) and its extraction plugin
#   - all sources, headers, scripts and makefiles needed to write, verify,
#     extract, and compile new kernels.
#
# It is modeled on F*'s .scripts/{bin-install,mk-package}.sh.
#
# Prerequisites (run `make prepare && make verify extract-all` first):
#   - inst/bin/fstar.exe and inst/bin/krml exist
#   - obj/*.checked exist (verified library)
#   - extraction/dune/_build/default/kuiper_extr.cmxs exists (plugin)
#
# Usage:
#   scripts/mk-package.sh [OUTPUT_BASENAME]
#
# The output archive is <OUTPUT_BASENAME>.tar.gz. If OUTPUT_BASENAME is not
# given, it defaults to kuiper$KUIPER_TAG, where KUIPER_TAG defaults to
# -$(uname -s)-$(uname -m), e.g. kuiper-Linux-x86_64.tar.gz.
#
# Environment:
#   KUIPER_TAG            Suffix for the default basename (default -<kernel>-<arch>)
#   KUIPER_PACKAGE_Z3     If "false", do not bundle Z3.
#   Z3_VERSION            Z3 version to bundle (default 4.13.3)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

KERNEL="$(uname -s)"
ARCH="$(uname -m)"
Z3_VERSION="${Z3_VERSION:-4.13.3}"

# Default tag and basename
KUIPER_TAG="${KUIPER_TAG:--$KERNEL-$ARCH}"
BASENAME="${1:-kuiper$KUIPER_TAG}"

CURL="curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors"

msg() { printf '>>> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- Sanity checks --------------------------------------------------------

[ -x inst/bin/fstar.exe ] || die "inst/bin/fstar.exe not found. Run 'make prepare' first."
[ -x inst/bin/krml ]      || die "inst/bin/krml not found. Run 'make prepare' first."
PLUGIN=extraction/dune/_build/default/kuiper_extr.cmxs
[ -f "$PLUGIN" ]          || die "extraction plugin ($PLUGIN) not found. Run 'make prepare' first."
if ! ls obj/*.checked >/dev/null 2>&1; then
  die "no obj/*.checked files found. Run 'make verify' first."
fi

# --- Staging directory ----------------------------------------------------

STAGE="$(mktemp -d --tmpdir kuiper-pkg.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/kuiper"
mkdir -p "$PKG"

msg "Staging Kuiper package in $PKG"

# --- Toolchain (F* + Karamel) --------------------------------------------

msg "Copying toolchain (inst/)"
cp -a inst "$PKG/inst"

# --- Z3 -------------------------------------------------------------------

bundle_z3() {
  local ver="$1" dest="$PKG/inst/lib/fstar"
  local url base
  case "$KERNEL-$ARCH-$ver" in
    Linux-x86_64-4.13.3)   url="https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-x64-glibc-2.35.zip" ;;
    Linux-aarch64-4.13.3|Linux-arm64-4.13.3)
                           url="https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-arm64-glibc-2.34.zip" ;;
    Darwin-x86_64-4.13.3)  url="https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-x64-osx-13.7.zip" ;;
    Darwin-arm64-4.13.3|Darwin-aarch64-4.13.3)
                           url="https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-arm64-osx-13.7.zip" ;;
    *) die "no known Z3 $ver download for $KERNEL-$ARCH. Set KUIPER_PACKAGE_Z3=false to skip." ;;
  esac

  msg "Bundling Z3 $ver from $url"
  local tmp; tmp="$(mktemp -d)"
  base="$(basename "$url")"
  $CURL "$url" -o "$tmp/$base"
  ( cd "$tmp" && unzip -q "$base" )
  mkdir -p "$dest/z3-$ver"
  cp -a "$tmp/${base%.zip}"/* "$dest/z3-$ver/"
  # Also expose it next to the binaries for PATH-based discovery.
  install -m0755 "$dest/z3-$ver/bin/z3" "$PKG/inst/bin/z3-$ver"
  rm -rf "$tmp"
}

if [ "${KUIPER_PACKAGE_Z3:-}" != "false" ]; then
  bundle_z3 "$Z3_VERSION"
else
  msg "Skipping Z3 bundling (KUIPER_PACKAGE_Z3=false)"
fi

# --- Kuiper library and sources ------------------------------------------

msg "Copying Kuiper sources and verified library"
cp -a src         "$PKG/src"
cp -a include     "$PKG/include"
cp -a scripts     "$PKG/scripts"
cp -a test        "$PKG/test"
cp -a tuning      "$PKG/tuning" 2>/dev/null || true

# Only the checked files from obj/ (the verified library). Users regenerate
# .cu/.h/.krml themselves; shipping the checked files avoids re-verification.
mkdir -p "$PKG/obj"
cp -a obj/*.checked "$PKG/obj/"

# Extraction plugin: ship sources + only the built .cmxs (not the whole _build).
mkdir -p "$PKG/extraction/dune/_build/default"
cp -a extraction/*.fst extraction/*.fsti extraction/*.json "$PKG/extraction/" 2>/dev/null || true
cp -a extraction/Makefile "$PKG/extraction/" 2>/dev/null || true
cp -a extraction/dune/dune extraction/dune/dune-project "$PKG/extraction/dune/" 2>/dev/null || true
cp -a "$PLUGIN" "$PKG/extraction/dune/_build/default/"

# Build system and helpers.
cp -a Makefile verify.mk nvcc.mk .common.mk .configure.mk configure \
      fstar.sh krml.sh "$PKG/"
cp -a Cfg.fst.config.json "$PKG/" 2>/dev/null || true
cp -a FOOTGUNS.txt "$PKG/" 2>/dev/null || true
# Ship the dependency graph so the first `make` needn't regenerate it.
# (.configure.output is intentionally NOT shipped: it is host-specific and is
# regenerated by ./configure on first use.)
cp -a .depend "$PKG/" 2>/dev/null || true

# --- Package-mode marker --------------------------------------------------
#
# The `.packaged` marker tells verify.mk that the toolchain/plugin ship
# prebuilt (see the ifeq ($(PACKAGED),) guards there). The touch files below
# are (re)created as the very last staging step so that make treats the
# toolchain, plugin, and dependency graph as already up to date.

touch "$PKG/.packaged"

# --- Strip binaries to save space -----------------------------------------

if command -v strip >/dev/null 2>&1; then
  msg "Stripping binaries"
  strip "$PKG"/inst/bin/* 2>/dev/null || true
  strip "$PKG"/inst/lib/fstar/z3-*/bin/* 2>/dev/null || true
fi

# --- Metadata -------------------------------------------------------------

commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
fstar_commit="$(git -C FStar rev-parse HEAD 2>/dev/null || echo unknown)"
krml_commit="$(git -C karamel rev-parse HEAD 2>/dev/null || echo unknown)"
cat > "$PKG/BUILD_INFO" <<EOF
Kuiper binary package [https://github.com/FStarLang/kuiper]
Built:          $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Platform:       $KERNEL-$ARCH
Kuiper commit:  $commit
F* commit:      $fstar_commit
Karamel commit: $krml_commit
Z3 version:     $Z3_VERSION
EOF

cp -a scripts/README-package.md "$PKG/README.md" 2>/dev/null || \
  cp -a README.md "$PKG/README.md"

# --- Freshen toolchain markers (must be the newest files) -----------------
#
# Created last, in dependency order, so that `make` in the extracted package
# never tries to rebuild the toolchain, plugin, or regenerate .depend.
touch "$PKG"/.fstar.src.touch "$PKG"/.fstar.touch
touch "$PKG"/.krml.src.touch "$PKG"/.krml.touch
touch "$PKG"/.plugin.touch
[ -f "$PKG/.depend" ] && touch "$PKG/.depend"

# --- Tar it up ------------------------------------------------------------

OUT="$ROOT/$BASENAME.tar.gz"
rm -f "$OUT"
msg "Creating $OUT"
# -h resolves symlinks so the package is self-contained.
tar czhf "$OUT" -C "$STAGE" kuiper

msg "Done."
ls -l "$OUT"
