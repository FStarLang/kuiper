#!/bin/bash

set -eu

while read msg; do
  if ! echo "$msg" | grep -q '^{'; then
    continue
  fi
  level=$(jq .level <<< $msg)
  file=$(jq -r .range.def.file_name <<< $msg)
  file=$(realpath --relative-to=. "${file}")
  line=$(jq .range.def.start_pos.line <<< $msg)
  endLine=$(jq .range.def.end_pos.line <<< $msg)
  body=$(cat <<< $msg | jq '.msg | join (". ")' | tr '\n' ' ')
  if [ "${level}" == "Error" ]; then
    glevel=error
  elif [ "${level}" == "Warning" ]; then
    glevel=warning
  fi
  echo "::${glevel} file=$file,line=$line,endLine=$endLine::$body"
done
