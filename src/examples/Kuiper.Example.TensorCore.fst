module Kuiper.Example.TensorCore

#lang-pulse
open Kuiper
open Kuiper.TensorCore
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as row_major, l2_col_major as col_major }
open Kuiper.Spec.GEMM
open Kuiper.EMatrix

inline_for_extraction noextract instance c16 : concrete_sz 16 = { x = 16sz; }
inline_for_extraction noextract instance c1 : concrete_sz 1 = { x = 1sz; }
inline_for_extraction noextract instance c48 : concrete_sz 48 = { x = 48sz; }

inline_for_extraction noextract
fn use_wmma_ker
  (m1 : array2 half (row_major 16 16))
  (m2 : array2 half (row_major 16 16))
  (m3 : array2 half (row_major 16 16))
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

let matplus_zero_lem
  (#m #n : nat)
  (mm : chest2 real m n)
  : Lemma (ensures matplus (const_matrix 0.0R) mm == mm)
  = assert (equal (matplus (const_matrix 0.0R) mm) mm);
    ()

[@@CPrologue "inline";
 CPrologue "__device__"]
fn test
  (m1 m2 m3 : array2 half (row_major 16 16))
  (#v1 #v2 #v3 : chest2 half 16 16)
  (#r1 #r2 #r3 : chest2 real 16 16)
  preserves
    m1 |-> v1 ** pure (v1 %~ r1) **
    m2 |-> v2 ** pure (v2 %~ r2)
  requires
    m3 |-> Frac (1.0R /. 32.0R) v3
  ensures
    (exists* (v3 : chest2 half 16 16).
      m3 |-> Frac (1.0R /. 32.0R) v3 ** pure (v3 %~ matmul r1 r2))
{
  with v1. assert m1 |-> v1;
  with v2. assert m2 |-> v2;
  with v3. assert tensor_pts_to m3 #(1.0R /. 32.0R) v3;

  let fa = __alloc_fragment half FragA 16sz 16sz 16sz FragLRM;
  let fb = __alloc_fragment half FragB 16sz 16sz 16sz FragLRM;
  let fc = __alloc_fragment half FragAcc 16sz 16sz 16sz FragLAcc;

  mma_loadA fa m1;
  mma_loadB fb m2;
  mma_fill fc zero;
  mma_sync' fa fb fc;
  mma_store fc m3;

  emma_approx_lemma (fill_value #half #FragAcc #16 #16 #16 zero) v1 v2 (Kuiper.EMatrix.const_matrix 0.0R) r1 r2;

  with x. assert fa |-> x; drop_ (fa |-> x);
  with x. assert fb |-> x; drop_ (fb |-> x);
  with x. assert fc |-> x; drop_ (fc |-> x);
  ()
}

[@@CPrologue "inline";
 CPrologue "__device__"]
fn test2
  (m1 m2 m3 : array2 half (row_major 48 48))
  (#v1 #v2 #v3 : chest2 half 48 48)
  (#r1 #r2 #r3 : chest2 real 48 48)
  preserves
    m1 |-> v1 ** pure (v1 %~ r1) **
    m2 |-> v2 ** pure (v2 %~ r2)
  requires
    m3 |-> Frac (1.0R /. 32) v3 ** pure (v3 %~ r3)
  ensures
    exists* (v3' : chest2 half 48 48).
      m3 |-> Frac (1.0R /. 32) v3' **
      pure (v3' %~
        update_tile #real r3 16 16 1 1
            (matmul #real
              (ematrix_subtile r1 16 16 1 1)
              (ematrix_subtile r2 16 16 1 1)))
{
  let fa = __alloc_fragment half FragA 16sz 16sz 16sz FragLRM;
  let fb = __alloc_fragment half FragB 16sz 16sz 16sz FragLRM;
  let fc = __alloc_fragment half FragAcc 16sz 16sz 16sz FragLAcc;

  array2_extract_tile_ro m1 16 16 1 1;
  let t1 = array2_subtile m1 16 16 1 1;
  assert (rewrites_to t1 (array2_subtile m1 16 16 1 1));

  array2_extract_tile_ro m2 16 16 1 1;
  let t2 = array2_subtile m2 16 16 1 1;
  assert (rewrites_to t2 (array2_subtile m2 16 16 1 1));

  array2_extract_tile m3 16 16 1 1;
  let t3 = array2_subtile m3 16 16 1 1;
  assert (rewrites_to t3 (array2_subtile m3 16 16 1 1));

  with vm3. assert tensor_pts_to t3 #(1.0R /. 32) vm3;

  mma_loadA fa t1;
  mma_loadB fb t2;
  mma_fill fc zero;
  mma_sync' fa fb fc;
  mma_store fc t3;

  emma_approx_lemma (fill_value #half #FragAcc #16 #16 #16 zero)
    (ematrix_subtile v1 16 16 1 1)
    (ematrix_subtile v2 16 16 1 1)
    (Kuiper.EMatrix.const_matrix 0.0R)
    (ematrix_subtile r1 16 16 1 1)
    (ematrix_subtile r2 16 16 1 1);

  with x1.
    assert t1 |-> x1;
    Pulse.Lib.Trade.elim_trade (t1 |-> x1) (m1 |-> v1);
  with x2.
    assert t2 |-> x2;
    Pulse.Lib.Trade.elim_trade (t2 |-> x2) (m2 |-> v2);
  with x3.
    assert tensor_pts_to t3 #(1.0R /. 32) x3;
    Pulse.Lib.Forall.elim_forall x3;
    Pulse.Lib.Trade.elim_trade (tensor_pts_to t3 #(1.0R /. 32) x3) _;

  with x. assert fa |-> x; drop_ (fa |-> x);
  with x. assert fb |-> x; drop_ (fb |-> x);
  with x. assert fc |-> x; drop_ (fc |-> x);

  ()
}
