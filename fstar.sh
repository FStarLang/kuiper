#!/usr/bin/env bash

# A helper to call F* with all the relevant flags to check a Pulse
# file in this repo.

SNAME="$0"

# macOS's bundled make (3.81) cannot parse our makefiles; Homebrew's GNU make
# installs as `gmake`. Prefer it when present. See setup-mac.sh.
MAKE="${MAKE:-$(command -v gmake >/dev/null 2>&1 && echo gmake || echo make)}"

gcmd () {
	cd $(dirname $SNAME)
	V=1 "$MAKE" -s echo-fstar
}

exec $(gcmd) --already_cached '*' "$@"
