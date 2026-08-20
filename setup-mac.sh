#!/usr/bin/env bash

# `make` here is GNU Make 4.x, installed as `gmake`. It is not optional: the
# makefiles use `define VAR =` (make >= 3.82) and .NOTINTERMEDIATE (make >=
# 4.4), which macOS's bundled make 3.81 cannot even parse. Build with `gmake`.
brew install gnu-sed coreutils gnu-indent make
