#!/bin/bash

set -eu

# Call this with no arguments and feeding a build log to stdin,
# or with arguments that are filenames to buildogs.

cat "$@" | while IFS= read msg; do
  if ! echo "$msg" | grep -q '^{'; then
    continue
  fi
  level=$(jq -r .level <<< $msg)
  file=$(jq -r .range.def.file_name <<< $msg)
  # -m: do not fail if file does not exist
  file=$(realpath -m --relative-to=. "${file}")
  line=$(jq -r .range.def.start_pos.line <<< $msg)
  endLine=$(jq -r .range.def.end_pos.line <<< $msg)
  body=$(cat <<< $msg | jq -r '.msg | join (". ")' | tr '\n' ' ')
  if [ "${level}" == "Error" ]; then
    glevel=error
  elif [ "${level}" == "Warning" ]; then
    glevel=warning
  else
    # Info message, probably
    continue
  fi
  echo "::${glevel} file=$file,line=$line,endLine=$endLine::$body"
done
