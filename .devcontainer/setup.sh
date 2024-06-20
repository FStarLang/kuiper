#!/bin/bash

eval $(opam env)

make -j$(nproc) update-pulse

# In the background, start building .checked files
nice make -j$(nproc) &
