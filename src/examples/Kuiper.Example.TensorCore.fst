module Kuiper.Example.TensorCore

#lang-pulse
open Kuiper
open Kuiper.TensorCore
open Kuiper.Matrix
open Kuiper.Matrix.Tiling
open Kuiper.Matrix.Reprs { row_major, col_major }
open Kuiper.Spec.GEMM
open Kuiper.EMatrix

inline_for_extraction noextract instance c16 : concrete_sz 16 = { x = 16sz; }
inline_for_extraction noextract instance c1 : concrete_sz 1 = { x = 1sz; }
inline_for_extraction noextract instance c48 : concrete_sz 48 = { x = 48sz; }

inline_for_extraction noextract
fn use_wmma_ker
  (m1 : gpu_matrix half (row_major 16 16))
  (m2 : gpu_matrix half (row_major 16 16))
  (m3 : gpu_matrix half (row_major 16 16))
  (fa : fragment   half FragA     16 16 16 FragLRM)
  (fb : fragment   half FragB     16 16 16 FragLRM)
  (fc : fragment   half FragAcc 16 16 16 FragLAcc)
  preserves
    (exists* v. m1 |-> v) **
    (exists* v. m2 |-> v) **
    (exists* v. pts_to m3 #(1.0R /. 32.0R) v) **
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

let matplus_zero_lem (#et:Type) {| scalar et |}
  (#m #n : nat)
  (mm : ematrix et m n)
  : Lemma (requires (forall (x:et). zero `add` x == x))
          (ensures matplus (const_matrix zero) mm == mm)
  = assert (equal (matplus (const_matrix zero) mm) mm);
    ()

[@@CPrologue "inline";
 CPrologue "__device__"]
fn test
  (m1 : gpu_matrix half (row_major 16 16))
  (m2 : gpu_matrix half (row_major 16 16))
  (m3 : gpu_matrix half (row_major 16 16))
  preserves
    m1 |-> 'v1 **
    m2 |-> 'v2
  requires
    // m3 |-> 'v3
   pts_to m3 #(1.0R /. 32.0R) 'v3
  ensures
    (* m3 |-> matmul #half 'v1 'v2 *)
    live m3 #(1.0R /. 32.0R)
{
  let fa = __alloc_fragment half FragA 16sz 16sz 16sz FragLRM;
  let fb = __alloc_fragment half FragB 16sz 16sz 16sz FragLRM;
  let fc = __alloc_fragment half FragAcc 16sz 16sz 16sz FragLAcc;

  // use_wmma_ker m1 m2 m3 fragA fragB fragC;
  mma_loadA fa m1;
  mma_loadB fb m2;
  mma_fill fc zero;
  mma_sync' fa fb fc;
  mma_store fc m3;

  (* lemma_mma_is_matmul_add (fill_value #half #FragAcc #16 #16 #16 zero) 'v1 'v2; *)
  (* assume (pure (forall (x:half). zero `add` x == x)); *)
  (* matplus_zero_lem (matmul 'v1 'v2); *)
  (* assert m3 |-> matmul 'v1 'v2; *)

  with x. assert fa |-> x; drop_ (fa |-> x);
  with x. assert fb |-> x; drop_ (fb |-> x);
  with x. assert fc |-> x; drop_ (fc |-> x);
  ()
}

[@@CPrologue "inline";
 CPrologue "__device__"]
fn test2
  (m1 : gpu_matrix half (row_major 48 48))
  (m2 : gpu_matrix half (row_major 48 48))
  (m3 : gpu_matrix half (row_major 48 48))
  (#v1 #v2 #v3 : ematrix half 48 48)
  preserves
    m1 |-> v1 **
    m2 |-> v2
  requires
    m3 |-> v3
  ensures
    live m3
    (* m3 |-> *)
    (*   update_tile #half v3 16 16 1 1 *)
    (*     (matplus (const_matrix #half #16 #16 zero) *)
    (*       (matmul #half *)
    (*         (ematrix_subtile v1 16 16 1 1) *)
    (*         (ematrix_subtile v2 16 16 1 1))) *)
{
  (* This is hacky due to the fractional permission on the matrix tiles. *)

  let fa = __alloc_fragment half FragA 16sz 16sz 16sz FragLRM;
  let fb = __alloc_fragment half FragB 16sz 16sz 16sz FragLRM;
  let fc = __alloc_fragment half FragAcc 16sz 16sz 16sz FragLAcc;

  gpu_matrix_extract_tile_ro m1 16 16 1 1;
  let t1 = gpu_matrix_subtile m1 16 16 1 1;
  assert (rewrites_to t1 (gpu_matrix_subtile m1 16 16 1 1));

  gpu_matrix_extract_tile_ro m2 16 16 1 1;
  let t2 = gpu_matrix_subtile m2 16 16 1 1;
  assert (rewrites_to t2 (gpu_matrix_subtile m2 16 16 1 1));

  gpu_matrix_extract_tile m3 16 16 1 1;
  let t3 = gpu_matrix_subtile m3 16 16 1 1;
  assert (rewrites_to t3 (gpu_matrix_subtile m3 16 16 1 1));

  with vm3. assert gpu_matrix_pts_to t3 vm3;
  rewrite
    gpu_matrix_pts_to t3 vm3
  as
    gpu_matrix_pts_to t3 #(1.0R /. 32.0R) vm3 ** gpu_matrix_pts_to t3 #(31.0R /. 32.0R) vm3
  by tadmit();

  mma_loadA fa t1;
  mma_loadB fb t2;
  mma_fill fc zero;
  mma_sync' fa fb fc;
  mma_store fc t3;

  with vm3'. assert gpu_matrix_pts_to t3 #(1.0R /. 32.0R) vm3';
  rewrite
    gpu_matrix_pts_to t3 #(1.0R /. 32.0R) vm3' ** gpu_matrix_pts_to t3 #(31.0R /. 32.0R) vm3
  as
    gpu_matrix_pts_to t3 vm3'
  by tadmit();

  (* lemma_mma_is_matmul_add *)
  (*   (fill_value #half #FragAcc #16 #16 #16 zero) *)
  (*   (ematrix_subtile v1 16 16 1 1) *)
  (*   (ematrix_subtile v2 16 16 1 1); *)

  with x1.
    assert t1 |-> x1;
    Pulse.Lib.Trade.elim_trade (t1 |-> x1) (m1 |-> v1);
  with x2.
    assert t2 |-> x2;
    Pulse.Lib.Trade.elim_trade (t2 |-> x2) (m2 |-> v2);
  with x3.
    assert t3 |-> x3;
    Pulse.Lib.Forall.elim_forall x3;
    Pulse.Lib.Trade.elim_trade (t3 |-> x3) _;

  with x. assert fa |-> x; drop_ (fa |-> x);
  with x. assert fb |-> x; drop_ (fb |-> x);
  with x. assert fc |-> x; drop_ (fc |-> x);
  ()
}
