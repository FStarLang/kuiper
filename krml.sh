#!/usr/bin/env bash

SNAME="$0"

# macOS's bundled make (3.81) cannot parse our makefiles; Homebrew's GNU make
# installs as `gmake`. Prefer it when present. See setup-mac.sh.
MAKE="${MAKE:-$(command -v gmake >/dev/null 2>&1 && echo gmake || echo make)}"

gcmd () {
	cd $(dirname $0)
	"$MAKE" -s echo-krml
}

exec $(gcmd) "$@"
