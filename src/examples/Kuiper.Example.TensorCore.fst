module Kuiper.Example.TensorCore

#lang-pulse
open Kuiper
open Kuiper.TensorCore
open Kuiper.Matrix
open Kuiper.Matrix.Reprs { row_major, col_major }
open Kuiper.Spec.GEMM
open Kuiper.EMatrix

inline_for_extraction noextract
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
    m3 |-> 'v3
  ensures
    m3 |-> matmul #half 'v1 'v2
{
  let fa = __alloc_fragment half FragA 16sz 16sz 16sz FragLRM;
  let fb = __alloc_fragment half FragB 16sz 16sz 16sz FragLRM;
  let fc = __alloc_fragment half FragAccum 16sz 16sz 16sz FragLAccum;

  // use_wmma_ker m1 m2 m3 fragA fragB fragC;
  mma_loadA fa m1;
  mma_loadB fb m2;
  mma_fill fc zero;
  mma_sync' fa fb fc;
  mma_store fc m3;

  assume (pure (forall (x:half). zero `add` x == x));
  matplus_zero_lem (matmul 'v1 'v2);
  assert m3 |-> matmul 'v1 'v2;

  with x. assert (fa |-> x); drop_ (fa |-> x);
  with x. assert (fb |-> x); drop_ (fb |-> x);
  with x. assert (fc |-> x); drop_ (fc |-> x);
  ()
}
