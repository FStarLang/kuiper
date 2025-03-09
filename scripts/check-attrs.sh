#!/bin/bash

# Very crappy, trying to make sure we mark kernels as KrmlPrivate

git grep __global__ | grep -v KrmlPrivate

exit 0
