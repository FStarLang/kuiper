module Kuiper.TensorCore.MMA.Registers

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorCore.MMA

module Cast = Kuiper.Float.Casts.Base

let pad_b7 (b : chest2 bf16 16 7) : chest2 bf16 16 8 =
  mk2 #bf16 #16 #8 (fun i j -> if j < 7 then acc2 b i j else zero #bf16)

let rounded7 (v : chest2 float 16 8) : chest2 bf16 16 7 =
  mk2 #bf16 #16 #7 (fun i j -> Cast.cast_f32_to_bf16 (acc2 v i j))

let add_values (d s : chest2 float 16 8) : chest2 float 16 8 =
  chest_comb (Kuiper.Scalars.Base.add #float) d s

(* Native correspondence boundary, not derived Pulse implementations.
   These operations use the EXISTING MMA.fragment and MMA.emma, with the
   same sm_80+, 32-live-lane convergence and input-publication obligations.
   The gpu capability alone does not establish those obligations.

   mma_sync7 loads only the 16-by-7 B view, using literal-zero packed B
   registers in group 7; there is no eighth memory column.
   copy/add act on the four native FP32 registers, preserving register order.
   store7_bf16 converts and writes only the 16-by-7 column-major view.
   No operation allocates global storage, launches work or synchronizes.
   ExtractKuiper and include/kuiper/mma.h implement this boundary. *)

fn mma_sync7
  (#la : layout2 16 16) {| strided_row_major la |}
  (a : array2 bf16 la)
  (#lb : layout2 16 7) {| strided_col_major lb |}
  (b : array2 bf16 lb)
  (fr : fragment)
  (#fa #fb : perm)
  (#va : chest2 bf16 16 16)
  (#vb : chest2 bf16 16 7)
  (#vc : chest2 float 16 8)
  preserves gpu
  preserves a |-> Frac fa va
  preserves b |-> Frac fb vb
  requires fr |-> vc
  ensures fr |-> emma vc va (pad_b7 vb)

fn copy
  (dst src : fragment)
  (#vd #vs : chest2 float 16 8)
  preserves gpu
  preserves src |-> vs
  requires dst |-> vd
  ensures dst |-> vs

fn add
  (dst src : fragment)
  (#vd #vs : chest2 float 16 8)
  preserves gpu
  preserves src |-> vs
  requires dst |-> vd
  ensures dst |-> add_values vd vs

fn store7_bf16
  (fr : fragment)
  (#l : layout2 16 7) {| strided_col_major l |}
  (c : array2 bf16 l)
  (#v : chest2 float 16 8)
  (#vc : chest2 bf16 16 7)
  preserves gpu
  preserves fr |-> v
  requires c |-> Frac (1.0R /. warp_size) vc
  ensures c |-> Frac (1.0R /. warp_size) (rounded7 v)

inline_for_extraction let () = ()
