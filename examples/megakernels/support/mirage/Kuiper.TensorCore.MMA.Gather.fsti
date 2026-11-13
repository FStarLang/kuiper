module Kuiper.TensorCore.MMA.Gather

#lang-pulse

open Kuiper

module S = FStar.Seq.Base
module F = Kuiper.Floating.Base
module C = Kuiper.Chest
module MMA = Kuiper.TensorCore.MMA

inline_for_extraction noextract
let score_a (input:lseq bf16 32768) (head:natlt 8) (step:natlt 8)
  : GTot (C.chest2 bf16 16 16) =
  C.mk2 (fun row col ->
    if row < 14 then
      S.index input
        ((row / 2) * 4096 + head * 512 + (row % 2) * 128 + step * 16 + col)
    else F.zero)

inline_for_extraction noextract
let score_b (input:lseq bf16 32768) (head:natlt 8) (step:natlt 8)
  : GTot (C.chest2 bf16 16 8) =
  C.mk2 (fun col key ->
    if key < 7 then
      S.index input (key * 4096 + head * 512 + 256 + step * 16 + col)
    else F.zero)

inline_for_extraction noextract
let value_a (probability:lseq bf16 1024) (step:natlt 4)
  : GTot (C.chest2 bf16 16 16) =
  C.mk2 (fun row col -> S.index probability (row * 64 + step * 16 + col))

inline_for_extraction noextract
let value_b (input:lseq bf16 32768)
  (head:natlt 8) (block:natlt 16) (step:natlt 4)
  : GTot (C.chest2 bf16 16 8) =
  C.mk2 (fun row col ->
    let key = step * 16 + row in
    if key < 7 then
      S.index input (key * 4096 + head * 512 + 384 + block * 8 + col)
    else F.zero)

(* Explicit native correspondence for reduced.cu's register loads and one
   m16n8k16 instruction, not a new hardware result or an attention theorem.
   As with MMA.mma_sync, all 32 live lanes must call convergently with the
   same parameters and published inputs. Callers prove those obligations.
   No allocation, memory copy, launch or barrier occurs here. *)
fn score
  (input:larray bf16 32768)
  (head:szlt 8) (step:szlt 8) (fr:MMA.fragment)
  (#values:erased (lseq bf16 32768))
  (#fraction:perm) (#before:C.chest2 float 16 8)
  preserves gpu ** input |-> Frac fraction values
  requires fr |-> before
  ensures fr |-> MMA.emma before (score_a values head step) (score_b values head step)

fn value
  (input:larray bf16 32768) (probability:larray bf16 1024)
  (head:szlt 8) (block:szlt 16) (step:szlt 4) (fr:MMA.fragment)
  (#values:erased (lseq bf16 32768))
  (#probabilities:erased (lseq bf16 1024))
  (#fi #fp:perm) (#before:C.chest2 float 16 8)
  preserves gpu ** input |-> Frac fi values ** probability |-> Frac fp probabilities
  requires fr |-> before
  ensures fr |-> MMA.emma before (value_a probabilities step) (value_b values head block step)
