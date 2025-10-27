#!/bin/bash

if [ ! -x ./bench ]; then
    echo "Run make to create the bench binary or make it excutable.";
else
    ./bench 0 |& tee L00
    ./bench 1 |& tee L01
    ./bench 2 |& tee L02
    ./bench 3 |& tee L03
    ./bench 4 |& tee L04
    ./bench 5 |& tee L05
    ./bench 6 |& tee L06
    ./bench 7 |& tee L07
    ./bench 8 |& tee L08
    ./bench 9 |& tee L09
    ./bench 10 |& tee L10
    ./bench 11 |& tee L11
fi
