#!/usr/bin/env bash

# Install Kuiper locally from GitHub releases.
#
# A Kuiper package is a self-contained tree bundling the F*/Karamel toolchain,
# Z3, and the verified Kuiper library. This script autodetects your OS and
# architecture, downloads the matching package, and unpacks it.
#
# Supports two sources:
#   - Official releases from FStarLang/kuiper
#   - Nightly builds from FStarLang/kuiper-nightly
#
# Can be run directly or via curl:
#   curl -fsSL https://raw.githubusercontent.com/FStarLang/kuiper/main/scripts/install-kuiper.sh | bash -s -- --nightly

main() {

set -euo pipefail

RELEASE_REPO=FStarLang/kuiper
# Nightlies are published as prereleases in the same repo, tagged nightly-*.
NIGHTLY_REPO=FStarLang/kuiper

usage() {
  cat <<'EOF'
Usage: install-kuiper.sh [OPTIONS]

Install Kuiper locally from GitHub releases.

Source (pick one):
  --release            Install from official releases (default)
  --nightly            Install a nightly build

Version:
  --version VER        Install a specific version instead of the latest.
                         For releases: a tag, e.g. v2026.03.24
                         For nightlies: a date, e.g. 2026-03-31

Destination:
  --dest DIR           Install Kuiper into DIR (default: ~/.local/kuiper)
  --link-dir DIR       Symlink the bundled binaries (fstar.exe, krml, z3) into
                         DIR (default: ~/.local/bin)
  --no-link            Don't create symlinks

Other:
  --list               List available versions and exit
  -v, --verbose        Show detailed output
  -h, --help           Show this help

Examples:
  install-kuiper.sh                                   # latest release
  install-kuiper.sh --nightly                         # latest nightly
  install-kuiper.sh --release --version v2026.03.24   # specific release
  install-kuiper.sh --nightly --version 2026-03-31    # specific nightly date
  install-kuiper.sh --dest ~/my-kuiper --no-link      # custom location
  install-kuiper.sh --nightly --list                  # list nightly versions

Via curl:
  curl -fsSL <url>/install-kuiper.sh | bash -s -- --nightly
EOF
}

# Defaults
SOURCE=release
VERSION=latest
DEST="$HOME/.local/kuiper"
LINK_DIR="$HOME/.local/bin"
DO_LINK=true
VERBOSE=false
LIST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)        SOURCE=release; shift ;;
    --nightly)        SOURCE=nightly; shift ;;
    --version)
      if [[ $# -lt 2 ]]; then echo "Error: --version requires an argument" >&2; exit 1; fi
      VERSION="$2"; shift 2 ;;
    --dest)
      if [[ $# -lt 2 ]]; then echo "Error: --dest requires an argument" >&2; exit 1; fi
      DEST="$2"; shift 2 ;;
    --link-dir)
      if [[ $# -lt 2 ]]; then echo "Error: --link-dir requires an argument" >&2; exit 1; fi
      LINK_DIR="$2"; shift 2 ;;
    --no-link)        DO_LINK=false; shift ;;
    --list)           LIST=true; shift ;;
    -v|--verbose)     VERBOSE=true; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)                echo "Error: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if $VERBOSE; then
  set -x
fi

# --- Helper: curl for GitHub API ---

gh_curl() {
  local -a headers=(
    -H "Accept: application/vnd.github+json"
    -H "X-GitHub-Api-Version: 2022-11-28"
  )
  curl -sL "${headers[@]}" "$@"
}

# --- Lightweight JSON helpers (no jq dependency) ---

json_field() {
  grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | head -1 | sed 's/.*:[[:space:]]*"//;s/"$//' || true
}

json_fields() {
  grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | sed 's/.*:[[:space:]]*"//;s/"$//' || true
}

# --- List mode ---

list_versions() {
  local repo
  case "$SOURCE" in
    release)        repo="$RELEASE_REPO" ;;
    nightly)        repo="$NIGHTLY_REPO" ;;
  esac

  echo "Available versions ($SOURCE):"
  echo ""
  for page in 1 2 3; do
    local json tags count
    json=$(gh_curl "https://api.github.com/repos/$repo/releases?per_page=30&page=$page")
    tags=$(echo "$json" | json_fields "tag_name")
    case "$SOURCE" in
      nightly)        tags=$(echo "$tags" | grep "^nightly-" || true) ;;
    esac
    if [[ -z "$tags" ]]; then
      break
    fi
    echo "$tags"
    count=$(echo "$json" | grep -c '"tag_name"' || true)
    if [[ "$count" -lt 30 ]]; then
      break
    fi
  done
}

if $LIST; then
  list_versions
  exit 0
fi

# --- Check required tools ---

for tool in curl tar; do
  if ! command -v "$tool" &>/dev/null; then
    echo "Error: '$tool' is required but not found." >&2
    exit 1
  fi
done

# --- Detect OS and architecture ---

kernel="$(uname -s)"
case "$kernel" in
  CYGWIN*|MINGW*|MSYS*) kernel=Windows_NT ;;
esac

arch="$(uname -m)"

# --- Construct download URL ---

# Build the asset filename for a given tag.
asset_filename() {
  # e.g. kuiper-Linux-x86_64.tar.gz  (nightly, no version in filename)
  #      kuiper-v2026.03.24-Linux-x86_64.tar.gz  (release)
  local tag="$1"
  case "$SOURCE" in
    release)  echo "kuiper-${tag}-${kernel}-${arch}.tar.gz" ;;
    nightly)  echo "kuiper-${kernel}-${arch}.tar.gz" ;;
  esac
}

asset_url() {
  local tag="$1"
  local repo filename
  case "$SOURCE" in
    release)  repo="$RELEASE_REPO" ;;
    nightly)  repo="$NIGHTLY_REPO" ;;
  esac
  filename=$(asset_filename "$tag")
  echo "https://github.com/${repo}/releases/download/${tag}/${filename}"
}

resolve_tag_and_url() {
  if [[ "$VERSION" != "latest" ]]; then
    case "$SOURCE" in
      release)  TAG="$VERSION" ;;
      nightly)  TAG="nightly-$VERSION" ;;
    esac
    ASSET_URL=$(asset_url "$TAG")
    return
  fi

  case "$SOURCE" in
    nightly)
      # Nightlies are prereleases tagged nightly-*, so /releases/latest (which
      # skips prereleases) won't find them. Scan the releases list for the
      # newest nightly-* tag instead.
      local repo="$NIGHTLY_REPO" json
      json=$(gh_curl "https://api.github.com/repos/$repo/releases?per_page=50")
      TAG=$(echo "$json" | json_fields "tag_name" | grep "^nightly-" | head -1)
      if [[ -z "$TAG" ]]; then
        echo "Error: could not find any nightly release in $repo." >&2
        exit 1
      fi
      ;;
    release)
      local repo="$RELEASE_REPO" release_json
      release_json=$(gh_curl "https://api.github.com/repos/$repo/releases/latest")
      TAG=$(echo "$release_json" | json_field "tag_name")
      if [[ -z "$TAG" ]]; then
        echo "Error: could not find the latest release." >&2
        local msg
        msg=$(echo "$release_json" | json_field "message")
        if [[ -n "$msg" ]]; then
          echo "GitHub API: $msg" >&2
        fi
        exit 1
      fi
      ;;
  esac
  ASSET_URL=$(asset_url "$TAG")
}

echo "Looking for Kuiper $SOURCE${VERSION:+ ($VERSION)} for ${kernel}-${arch}..."

TAG=""
ASSET_URL=""
resolve_tag_and_url

echo "Found: $TAG"

ASSET_NAME=$(basename "$ASSET_URL")
echo "Downloading $ASSET_NAME..."

# --- Download ---

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

if ! curl -fL "$ASSET_URL" -o "$WORKDIR/$ASSET_NAME"; then
  echo "Error: failed to download $ASSET_URL" >&2
  echo "There may be no Kuiper package for ${kernel}-${arch} in $TAG." >&2
  exit 1
fi

# --- Remove previous installation ---

if [[ -e "$DEST" ]]; then
  echo "Removing previous installation at $DEST..."
  rm -rf "$DEST"
fi
mkdir -p "$DEST"

# --- Extract ---
# The tarball has a leading kuiper/ directory; strip it into DEST.

echo "Extracting to $DEST..."
tar xzf "$WORKDIR/$ASSET_NAME" --strip-components=1 -C "$DEST"

# Sanity check
if [[ ! -x "$DEST/inst/bin/fstar.exe" ]]; then
  echo "Warning: $DEST/inst/bin/fstar.exe not found or not executable." >&2
  echo "Contents of $DEST:" >&2
  ls -la "$DEST" >&2
fi

echo "Installed Kuiper ($TAG) to $DEST"

# --- Create symlinks ---

if $DO_LINK; then
  mkdir -p "$LINK_DIR"
  for bin in "$DEST/inst/bin/"*; do
    [[ -f "$bin" ]] || continue
    name=$(basename "$bin")
    ln -sf "$(realpath "$bin")" "$LINK_DIR/$name"
    echo "  Linked $LINK_DIR/$name -> $bin"
  done
  if ! echo "$PATH" | tr ':' '\n' | grep -qxF "$LINK_DIR"; then
    echo ""
    echo "Note: $LINK_DIR is not in your PATH."
    echo "Add it with: export PATH=\"$LINK_DIR:\$PATH\""
  fi
fi

echo ""
echo "Done! Kuiper $TAG is ready in $DEST."
echo ""
echo "To verify/extract kernels, work inside the package tree:"
echo "    cd $DEST"
echo "    make -j\$(nproc)                                  # verify + extract everything"
echo "    ./fstar.sh src/examples/Kuiper.Example.Add.fst   # verify a single file"
echo ""
echo "See $DEST/README.md for how to write and build new kernels."

} # end of main()

main "$@"
