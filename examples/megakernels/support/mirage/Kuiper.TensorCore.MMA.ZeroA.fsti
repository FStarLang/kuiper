module Kuiper.TensorCore.MMA.ZeroA

#lang-pulse

open Kuiper
open Kuiper.Shape

module S = FStar.Seq.Base
module F = Kuiper.Floating.Base
module C = Kuiper.Chest
module MMA = Kuiper.TensorCore.MMA
module Registers = Kuiper.TensorCore.MMA.Registers

inline_for_extraction noextract
let operand_a : C.chest2 bf16 16 16 =
  C.const (16 @| 16 @| INil) (F.zero #bf16)

inline_for_extraction noextract
let operand_b7 (hidden:lseq bf16 7168) (step:natlt 64)
  : GTot (C.chest2 bf16 16 7) =
  C.mk2 (fun col row -> S.index hidden (row * 1024 + step * 16 + col))

(* Explicit native correspondence for reduced.cu's layout3 padded A tiles.
   All32 live lanes use the same step and published hidden-state inputs.
   This binds register operands to the existing MMA result, not a zero-result
   identity or an entire projection theorem. No memory is used for operand A. *)
fn mma_sync7
  (hidden:larray bf16 7168) (step:szlt 64) (fr:MMA.fragment)
  (#values:erased (lseq bf16 7168)) (#fraction:perm)
  (#before:C.chest2 float 16 8)
  preserves gpu ** hidden |-> Frac fraction values
  requires fr |-> before
  ensures fr |-> MMA.emma before operand_a (Registers.pad_b7 (operand_b7 values step))
