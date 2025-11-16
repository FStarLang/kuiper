#!/bin/bash
#
# Constructing a very basic histogram from a tuning log.

step=$1
file=$2

points=$(sed -n 's/.*	\([0-9]*\)\.[0-9]* GFLOPS.*/\1/p' < $file)
buckets=()
max=0

echo $points

for p in $points; do
	echo "p=$p"
	b=$(($p / $step))
	buckets["$b"]=$((1 + ${buckets["$b"]:-0}))
	if [ "$b" -gt "$max" ]; then max=$b; fi
done

for i in $(seq 0 $max); do
	echo \($((i * step)), ${buckets[$i]:-0}\)
done
