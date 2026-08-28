#!/usr/bin/env bash

# Install the pinned clang-format binary bundled in Kuiper packages.

set -euo pipefail

version="${CLANG_FORMAT_VERSION:-19.1.7}"
dest="${1:?usage: install-clang-format.sh DEST}"

if [[ "$version" != "19.1.7" ]]; then
  echo "Error: no wheel checksum is recorded for clang-format $version." >&2
  exit 1
fi

platform="$(uname -s):$(uname -m)"
case "$platform" in
  Linux:x86_64)
    url=https://files.pythonhosted.org/packages/3c/e7/0e526915a3a4a23100cc721c24226a192fa0385d394019d06920dc83fe6c/clang_format-19.1.7-py2.py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    sha256=f4906fb463dd2033032978f56962caab268c9428a384126b9400543eb667f11c
    ;;
  Linux:aarch64)
    url=https://files.pythonhosted.org/packages/b1/7d/002aa5571351ee7f00f87aae5104cdd30cad1a46f25936226f7d2aed06bf/clang_format-19.1.7-py2.py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    sha256=dac394c83a9233ab6707f66e1cdbd950f8b014b58604142a5b6f7998bf0bcc8c
    ;;
  Darwin:x86_64)
    url=https://files.pythonhosted.org/packages/5a/c3/2f1c53bc298c1740d0c9f8dc2d9b7030be4826b6f2aa8a04f07ef25a3d9b/clang_format-19.1.7-py2.py3-none-macosx_10_9_x86_64.whl
    sha256=a09f34d2c89d176581858ff718c327eebc14eb6415c176dab4af5bfd8582a999
    ;;
  Darwin:arm64)
    url=https://files.pythonhosted.org/packages/8e/9d/7c246a3d08105de305553d14971ed6c16cde06d20ab12d6ce7f243cf66f0/clang_format-19.1.7-py2.py3-none-macosx_11_0_arm64.whl
    sha256=776f89c7b056c498c0e256485bc031cbf514aaebe71e929ed54e50c478524b65
    ;;
  *)
    echo "Error: no pinned clang-format wheel for $platform." >&2
    exit 1
    ;;
esac

for tool in curl python3; do
  if ! command -v "$tool" >/dev/null; then
    echo "Error: '$tool' is required to install clang-format." >&2
    exit 1
  fi
done

workdir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$workdir"
}
trap cleanup EXIT

wheel="$workdir/clang-format.whl"
curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors "$url" -o "$wheel"

if command -v sha256sum >/dev/null; then
  printf '%s  %s\n' "$sha256" "$wheel" | sha256sum -c -
elif command -v shasum >/dev/null; then
  actual="$(shasum -a 256 "$wheel" | awk '{ print $1 }')"
  if [[ "$actual" != "$sha256" ]]; then
    echo "Error: clang-format wheel checksum mismatch." >&2
    exit 1
  fi
else
  echo "Error: sha256sum or shasum is required to verify clang-format." >&2
  exit 1
fi

python3 -m zipfile -e "$wheel" "$workdir/unpacked"
mkdir -p "$dest/bin"
install -m 755 "$workdir/unpacked/clang_format/data/bin/clang-format" \
  "$dest/bin/clang-format"
mkdir -p "$dest/share/licenses/clang-format"
install -m 644 \
  "$workdir/unpacked/clang_format-$version.dist-info/licenses/LICENSE.md" \
  "$dest/share/licenses/clang-format/LICENSE.md"

actual="$("$dest/bin/clang-format" --version)"
if [[ "$actual" != "clang-format version $version" ]]; then
  echo "Error: installed formatter reports '$actual'." >&2
  exit 1
fi
