#!/bin/bash

# A helper to call F* with all the relevant flags to check a Pulse
# file in this repo.

FSTAR=$(make echo-fstar)

exec ${FSTAR} "$@"
