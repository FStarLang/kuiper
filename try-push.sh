#!/bin/bash
#
# try-push.sh: push the FStar/karamel submodule objects to origin, then
# advance the submodule pointers in this repo and push the result.

set -euo pipefail

# Always operate from the directory containing this script (the repo root).
cd "$(dirname "$0")"

# Submodules tracked and advanced by this repo.
SUBMODULES=(FStar karamel)

fail () {
	echo "ERROR: $*" >&2
	exit 1
}

# Print a command (with a leading '$'), then run it.
run () {
	echo "\$ $*"
	"$@"
}

trap 'fail "try-push.sh aborted (see output above)."' ERR

echo "==> [1/4] Pushing submodule objects to origin"
[ -x ./push-submodules.sh ] || fail "./push-submodules.sh not found or not executable"
./push-submodules.sh || fail "failed to push submodule objects"

echo
echo "==> [2/4] Staging submodule pointer updates"
run git add -- "${SUBMODULES[@]}"

echo
echo "==> [3/4] Committing submodule advance"
if git diff --cached --quiet -- "${SUBMODULES[@]}"; then
	echo "Nothing to commit; submodule pointers are unchanged."
else
	run git commit -m 'Advance submodules'
fi

echo
echo "==> [4/4] Pushing to origin"
run git push || fail "git push failed"

echo
echo "==> Done."
