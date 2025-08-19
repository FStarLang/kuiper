module Kuiper.Example.TensorCore

#lang-pulse
open Kuiper
open Kuiper.TensorCore
open Kuiper.Matrix
open Kuiper.Matrix.Reprs { row_major, col_major }

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
  // mma_store fc m3;
  ()
}

[@@CPrologue "__device__"]
fn test
  (m1 : gpu_matrix half (row_major 16 16))
  (m2 : gpu_matrix half (row_major 16 16))
  (m3 : gpu_matrix half (row_major 16 16))
  preserves
    (exists* v. m1 |-> v) **
    (exists* v. m2 |-> v) **
    (exists* v. m3 |-> v)
{
  // let mut fragA : fragment half FragA 16 16 16 FragLRM = magic();
  let fragA = __alloc_fragment half FragA 16sz 16sz 16sz FragLRM;
  let fragB = __alloc_fragment half FragB 16sz 16sz 16sz FragLRM;
  let fragC = __alloc_fragment half FragAccum 16sz 16sz 16sz FragLAccum;

  use_wmma_ker m1 m2 m3 fragA fragB fragC;

  with x. assert (fragA |-> x); drop_ (fragA |-> x);
  with x. assert (fragB |-> x); drop_ (fragB |-> x);
  with x. assert (fragC |-> x); drop_ (fragC |-> x);
  ()
}
