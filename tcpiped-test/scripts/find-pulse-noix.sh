#!/bin/bash

find_fsti () {
  fn="$1"
  fn="$(basename "$fn")"
  find . -type f -name "$fn" | grep -q .
}

for fn in $(git grep -l '#lang-pulse')
do
  if [[ "$fn" =~ .fst$ ]] && ! find_fsti "$fn"
  then
    echo "No interface for: $fn"
  elif [[ "$fn" =~ .fsti$ ]]
  then
    # Try to find Pulse *code* in the fsti.
    if grep -q '^ *\(unfold|fold\).*; *$' "$fn" \
       || grep -q 'rewrite each' "$fn" \
       || grep -q '^ *let.*=.*; *$' "$fn"
    then
      echo "Interface seems to have Pulse code: $fn"
    fi
  fi
done

git grep 'GTot prop'
git grep 'GTot Type0'
git grep 'GTot (erased'

exit 0
