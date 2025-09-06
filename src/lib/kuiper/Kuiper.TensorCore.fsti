module Kuiper.TensorCore

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs { row_major, col_major }
open Kuiper.Spec.GEMM { mma }

// inline_for_extraction noextract
type fragment_kind =
  | FragA
  | FragB
  | FragAcc

// inline_for_extraction noextract
type fragment_layout =
  | FragLRM
  | FragLCM
  | FragLAcc

// this ignores checking for a valid compute capability
let valid_frag_et_dims
  (et : Type0)
  (knd : fragment_kind)
  (m n k : nat) : prop
= 
  ((((knd == FragA \/ knd == FragB) /\ (et == half \/ et == u8 \/ et == i8)) \/
     (knd == FragAcc /\ (et == float \/ et == half \/ et == int))) /\
     ((m == 16 /\ n == 16 /\ k == 16) \/
      (m == 32 /\ n == 8 /\ k == 16)  \/
      (m == 8 /\ n == 32 /\ k == 16))) \/
  // ignore alternate fp and experimental sub-byte ops and add
  // special case for double
  (et == double /\ m == 8 /\ n == 8 /\ k == 8) \/
  False

let valid_frag_et_comb
  (et_ab et_acc : Type0) : prop
=
  // requires sm_70+
  (et_ab == half /\ (et_acc == half \/ et_acc == float)) \/
  // requires sm_72+
  ((et_ab == u8 \/ et_ab == i8) /\ et_acc == int) \/
  // skip alternate fp (sm_80+) and experimental sub-byte ops (sm_75+)
  // double requires (sm_80+)
  (et_ab == double /\ et_acc == double) \/
  False

// [@@erasable]
// class valid_frag (et : Type0) (knd : fragment_kind) (m n k : nat) = { x: unit }
// instance valid_fragA_half_16x16x16 : valid_frag half FragA 16 16 16 = { x = () }
// instance valid_fragB_half_16x16x16 : valid_frag half FragB 16 16 16 = { x = () }
// instance valid_fragAcc_half_16x16x16 : valid_frag half FragAcc 16 16 16 = { x = () }

// instance valid_fragA_half_32x8x16 : valid_frag half FragAcc 32 8 16 = { x = () }
// instance valid_fragB_half_32x8x16 : valid_frag half FragAcc 32 8 16 = { x = () }
// instance valid_fragAcc_half_32x8x16 : valid_frag half FragAcc 32 8 16 = { x = () }

new
val fragment
  (et : Type0)
  (knd : fragment_kind)
  (m n k : nat)
  (layout : fragment_layout)
  : Type0

let value_for et knd m n k =
  match knd with
  | FragA   -> ematrix et m k
  | FragB   -> ematrix et k n
  | FragAcc -> ematrix et m n

val fragment_pts_to
  (#et : Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#fl : fragment_layout)
  ([@@@mkey] f : fragment et knd m n k fl)
  (em  : value_for et knd m n k)
  : slprop

unfold
instance has_pts_to_fragment
  (#et : Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#fl : fragment_layout)
  : has_pts_to (fragment et knd m n k fl) (value_for et knd m n k) =
{
  pts_to = (fun r #f v -> fragment_pts_to r v);
}

ghost
fn fragment_pts_to_ref
  (#et : Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#fl : fragment_layout)
  (f   : fragment et knd m n k fl)
  (em  : value_for et knd m n k)
  preserves
    fragment_pts_to f em
  ensures
    pure (valid_frag_et_dims et knd m n k) **
    pure (knd == FragAcc <==> fl == FragLAcc) // useful?

fn mma_sync'
  (#et_ab #et_acc : Type)
  // not *really* necessary I think
  {| scalar et_ab, scalar et_acc |}
  (#m #n #k : erased nat)
  (#la #lb : fragment_layout)
  (fa : fragment et_ab FragA   m n k la)
  (fb : fragment et_ab FragB   m n k lb)
  (fc : fragment et_acc FragAcc m n k FragLAcc)
  (#ea : ematrix et_ab m k)
  (#eb : ematrix et_ab k n)
  (#ec : ematrix et_acc m n)
  preserves fa |-> ea
  preserves fb |-> eb
  // this is trivially true because we currently do not allow mix-precision;
  // mix-precision would have to be handled in the specs (matplus)
  preserves pure (valid_frag_et_comb et_ab et_acc)
  requires
    fc |-> ec
  ensures
    fc |-> mma ec ea eb 

fn mma_loadA
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragA m n k FragLRM)
  (#l : mlayout m k) {| strided_row_major l |}
  (gm : gpu_matrix et l)
  (#f : perm)
  (#m0 : ematrix et m k)
  (#f0 : erased (value_for et FragA m n k))
  preserves
    gm |-> Frac f m0
  requires
    fr |-> f0
  ensures
    fr |-> m0

fn mma_loadB
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragB m n k FragLRM)
  (#l : mlayout k n) {| strided_row_major l |}
  (gm : gpu_matrix et l)
  (#f : perm)
  (#m0 : ematrix et k n)
  (#f0 : erased (value_for et FragB m n k))
  preserves
    gm |-> Frac f m0
  requires
    fr |-> f0
  ensures
    fr |-> m0

fn mma_loadAccum
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragAcc m n k FragLAcc)
  (#l : mlayout m n) {| strided_row_major l |}
  (gm : gpu_matrix et l)
  (#f : perm)
  (#m0 : ematrix et m n)
  (#f0 : erased (value_for et FragAcc m n k))
  preserves
    gm |-> Frac f m0
  requires
    fr |-> f0
  ensures
    fr |-> m0

let fill_value
  (#et : Type)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (i : et)
  : value_for et knd m n k
=
  match knd with
  | FragA     -> EMatrix.const_matrix #_ #m #k i
  | FragB     -> EMatrix.const_matrix #_ #k #n i
  | FragAcc -> EMatrix.const_matrix #_ #m #n i

fn mma_fill
  (#et : Type)
  (#knd : fragment_kind)
  (#m #n #k : erased nat)
  (#ly : fragment_layout)
  (fr : fragment et knd m n k ly)
  (i : et)
  (#v0 : erased (value_for et knd m n k))
  requires fr |-> v0
  ensures  fr |-> fill_value i

fn mma_store
  (#et : Type)
  (#m #n #k : erased nat)
  (fr : fragment et FragAcc m n k FragLAcc)
  (#l : mlayout m n) {| strided_row_major l |}
  (gm : gpu_matrix et l)
  (#f0 : erased (value_for et FragAcc m n k))
  (#m0 : ematrix et m n)
  preserves
    fr |-> f0
  requires
    gm |-> m0
  ensures
    gm |-> f0

(* We should add checker support for this. *)
fn with_fragment u#r
  (#pre   : slprop)
  (#ret_t : Type u#r)
  (#post  : ret_t -> slprop)
  (et : Type0) (knd : fragment_kind)
  (m n k : sz)
  (fl : fragment_layout)
  (body : (
    (fr : fragment et knd m n k fl) ->
      stt ret_t (
        requires pure (valid_frag_et_dims et knd m n k) **
                 pre  ** (exists* em. fragment_pts_to fr em))
        (ensures  fun r ->
                 pure (valid_frag_et_dims et knd m n k) **
                 post r ** (exists* em. fragment_pts_to fr em))
  ))
  preserves pure (valid_frag_et_dims et knd m n k)
  requires pre
  returns  r : ret_t
  ensures  post r

(* Unsound, clearly *)
fn __alloc_fragment
  (et : Type0) (knd : fragment_kind)
  (m n k : sz)
  (fl : fragment_layout)
  preserves pure (valid_frag_et_dims et knd m n k)
  returns  fr : fragment et knd m n k fl
  ensures  exists* v. fr |-> v
