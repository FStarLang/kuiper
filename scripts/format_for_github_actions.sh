#!/bin/bash

set -eu

while read msg; do
  if ! echo "$msg" | grep -q '^{'; then
    continue
  fi
  echo "chec"
  if [ x$(echo "$msg" | jq .level) == 'x"Error"' ]; then
    file=$(echo "$msg" | jq .range.def.file_name)
    line=$(echo "$msg" | jq .range.def.start_pos.line)
    endLine=$(echo "$msg" | jq .range.def.end_pos.line)
    body=$(echo "$msg" | jq .msg | tr -d '\n')
    echo "::error file=$file,line=$line,endLine=$endLine::$body"
  elif [ x$(echo "$msg" | jq .level) == 'x"Warning"' ]; then
    file=$(echo "$msg" | jq .range.def.file_name)
    line=$(echo "$msg" | jq .range.def.start_pos.line)
    endLine=$(echo "$msg" | jq .range.def.end_pos.line)
    body=$(echo "$msg" | jq .msg | tr -d '\n')
    echo "::warning file=$file,line=$line,endLine=$endLine::$body"
  fi
done
