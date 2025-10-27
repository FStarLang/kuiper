#!/bin/bash

# Don't use a space after @@
git grep '^\[@@ '

# Very crappy, trying to make sure we mark kernels as KrmlPrivate
git grep __global__ | grep -v KrmlPrivate

exit 0
