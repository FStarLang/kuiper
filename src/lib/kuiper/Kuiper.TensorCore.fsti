module Kuiper.TensorCore

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs { row_major, col_major }
open Kuiper.Spec.GEMM { matmul, matplus }

type fragment_kind = | FragA | FragB | FragAccum

type fragment_layout =
  | FragLRM
  | FragLCM
  | FragLAccum

let valid_frag_dimensions et (m n k : nat) : prop =
  (et == half /\ m == 16 /\ n == 16 /\ k == 16) \/
  False

new
val fragment
  (et : Type0)
  (k : fragment_kind)
  (m n k : nat)
  (layout : fragment_layout)
  : Type0

let value_for et knd m n k =
  match knd with
  | FragA     -> ematrix et m k
  | FragB     -> ematrix et k n
  | FragAccum -> ematrix et m n

val fragment_pts_to
  (#et : Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#fl : fragment_layout)
  (f   : fragment et knd m n k fl)
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
    pure (valid_frag_dimensions et m n k) **
    pure (knd == FragAccum <==> fl == FragLAccum) // useful?

fn mma_sync'
  (#et : Type) {| scalar et |}
  (#m #n #k : nat)
  (#la #lb : fragment_layout)
  (fa : fragment et FragA     m n k la)
  (fb : fragment et FragB     m n k lb)
  (fc : fragment et FragAccum m n k FragLAccum)
  (#ea : ematrix et m k)
  (#eb : ematrix et k n)
  (#ec : ematrix et m n)
  preserves fa |-> ea
  preserves fb |-> eb
  requires
    fc |-> ec
  ensures
    fc |-> matplus ec (matmul ea eb)

fn mma_loadA
  (#et : Type)
  (#m #n #k : nat)
  (fr : fragment et FragA m n k FragLRM)
  (gm : gpu_matrix et (row_major m k))
  (#m0 : ematrix et m k)
  (#f0 : value_for et FragA m n k)
  preserves
    gm |-> m0
  requires
    fr |-> f0
  ensures
    fr |-> m0

fn mma_loadB
  (#et : Type) {| scalar et |}
  (#m #n #k : nat)
  (fr : fragment et FragB m n k FragLRM)
  (gm : gpu_matrix et (row_major k n))
  (#m0 : ematrix et k n)
  (#f0 : value_for et FragB m n k)
  preserves
    gm |-> m0
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
  | FragA     -> EMatrix.mkM #_ #m #k (fun _ _ -> i)
  | FragB     -> EMatrix.mkM #_ #k #n (fun _ _ -> i)
  | FragAccum -> EMatrix.mkM #_ #m #n (fun _ _ -> i)

fn mma_fill
  (#et : Type)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#ly : fragment_layout)
  (fr : fragment et knd m n k ly)
  (i : et)
  (#v0 : value_for et knd m n k)
  requires fr |-> v0
  ensures  fr |-> fill_value i

fn mma_store
  (#et : Type)
  (#m #n #k : nat)
  (fr : fragment et FragAccum m n k FragLAccum)
  (gm : gpu_matrix et (row_major m n))
  (#f0 : value_for et FragAccum m n k)
  (#m0 : ematrix et m n)
  preserves
    fr |-> f0
  requires
    gm |-> m0
  ensures
    gm |-> f0

fn use_wmma_ker
  (m1 : gpu_matrix half (row_major 16 16))
  (m2 : gpu_matrix half (row_major 16 16))
  (m3 : gpu_matrix half (row_major 16 16))
  (fa : fragment   half FragA     16 16 16 FragLRM)
  (fb : fragment   half FragB     16 16 16 FragLRM)
  (fc : fragment   half FragAccum 16 16 16 FragLAccum)
  preserves
    (exists* v. m1 |-> v) **
    (exists* v. m2 |-> v) **
    (exists* v. m3 |-> v) **
    (exists* v. fa |-> v) **
    (exists* v. fb |-> v) **
    (exists* v. fc |-> v)
{
  mma_loadA fa m1;
  mma_loadB fb m2;
  mma_fill fc zero;
  mma_sync' fa fb fc;
  mma_store fc m3;
  ()
}

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
      stt ret_t (requires          pre  ** (exists* em. fragment_pts_to fr em))
                (ensures  fun r -> post r ** (exists* em. fragment_pts_to fr em))
  ))
  requires pre
  returns  r : ret_t
  ensures  post r

(* Unsound, clearly *)
fn __alloc_fragment
  (et : Type0) (knd : fragment_kind)
  (m n k : sz)
  (fl : fragment_layout)
  returns  fr : fragment et knd m n k fl
  ensures  exists* v. fr |-> v


// fn wmma_ker
//   (m1 : gpu_matrix half (row_major 16 16))
//   (m2 : gpu_matrix half (row_major 16 16))
//   (m3 : gpu_matrix half (row_major 16 16))
//   preserves exists* v. m1 |-> v
//   preserves exists* v. m2 |-> v
//   preserves exists* v. m3 |-> v
// {
//   with_fragment
//     #((exists* v. gpu_matrix_pts_to m1 #1.0R v) ** (exists* v. gpu_matrix_pts_to m2 #1.0R v) ** (exists* v. gpu_matrix_pts_to m3 #1.0R v))
//     #_
//     #(fun _ -> (exists* v. gpu_matrix_pts_to m1 #1.0R v) ** (exists* v. gpu_matrix_pts_to m2 #1.0R v) ** (exists* v. gpu_matrix_pts_to m3 #1.0R v))
//     half FragA 16sz 16sz 16sz FragLRM (fun fa ->
//       with_fragment
//         #((exists* v. gpu_matrix_pts_to m1 #1.0R v) ** (exists* v. gpu_matrix_pts_to m2 #1.0R v) ** (exists* v. gpu_matrix_pts_to m3 #1.0R v) ** (exists* v. fragment_pts_to fa v))
//         #_
//         #(fun _ -> (exists* v. gpu_matrix_pts_to m1 #1.0R v) ** (exists* v. gpu_matrix_pts_to m2 #1.0R v) ** (exists* v. gpu_matrix_pts_to m3 #1.0R v) ** (exists* v. fragment_pts_to fa v))
//         half FragB 16sz 16sz 16sz FragLRM (fun fb ->
//           with_fragment
//             #((exists* v. gpu_matrix_pts_to m1 #1.0R v) ** (exists* v. gpu_matrix_pts_to m2 #1.0R v) ** (exists* v. gpu_matrix_pts_to m3 #1.0R v) ** (exists* v. fragment_pts_to fa v) ** (exists* v. fragment_pts_to fb v))
//             #_
//             #(fun _ -> (exists* v. gpu_matrix_pts_to m1 #1.0R v) ** (exists* v. gpu_matrix_pts_to m2 #1.0R v) ** (exists* v. gpu_matrix_pts_to m3 #1.0R v) ** (exists* v. fragment_pts_to fa v) ** (exists* v. fragment_pts_to fb v))
//             half FragAccum 16sz 16sz 16sz FragLAccum (fun fc ->
//             use_wmma_ker m1 m2 m3 fa fb fc)))
// }
