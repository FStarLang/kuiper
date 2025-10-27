#!/bin/bash

if [ $# -ne 2 ]; then
  echo "usage: $0 <actual_output> <expected_output>" >&2
  exit 1
fi

ACTUAL="$1"
EXPECTED="$2"

DIFF="diff -u --strip-trailing-cr"

# Doing the diff in this order is more natural for error
# messages, as '+' means new lines and '-' lines that are
# no longer appearing.
if $DIFF -q "$EXPECTED" "$ACTUAL" ; then
  # OK
  exit 0
else
  # We're gonna fail. If we're running in CI, emit a Github
  # error message.
  if [ -v GITHUB_ENV ]; then
    DIFFTEXT=$($DIFF "$EXPECTED" "$ACTUAL" | sed 's/$/%0A/' | tr -d '\n')
    ACTUAL=$(realpath "$ACTUAL")
    ACTUAL="${ACTUAL#$FSTAR_ROOT}"
    EXPECTED=$(realpath "$EXPECTED")
    EXPECTED="${EXPECTED#$FSTAR_ROOT}"
    echo "::error::File $ACTUAL did not match expected $EXPECTED:%0A%0A$DIFFTEXT"
  else
    echo "error: File $ACTUAL did not match expected $EXPECTED. Diff follows." >&2
    $DIFF "$EXPECTED" "$ACTUAL"
  fi
  exit 1
fi
