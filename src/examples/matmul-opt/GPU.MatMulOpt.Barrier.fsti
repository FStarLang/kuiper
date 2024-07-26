module GPU.MatMulOpt.Barrier

open Pulse.Lib.Pervasives

let barrier_mm
    (n: nat)
    (it: nat)
    (from: nat { 0 <= from /\ from < n })
    (to: nat { 0 <= to /\ to < n })
    : slprop = emp
