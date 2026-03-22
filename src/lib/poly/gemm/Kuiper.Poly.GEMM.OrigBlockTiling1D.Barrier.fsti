module Kuiper.Poly.GEMM.OrigBlockTiling1D.Barrier

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix { gpu_matrix }
open Kuiper.Matrix.Reprs.Type

module SZ = Kuiper.SizeT

open Kuiper.Poly.GEMM.OrigBlockTiling1D.Defs

inline_for_extraction let () = ()

ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols * bn))
  (bid : natlt (mrows * mcols))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (#_ : squash (SZ.fits (mlayout_size l1)))
  (#_ : squash (SZ.fits (mlayout_size l2)))
  (it : nat)
  requires
    forall+ (tid : natlt (bm/tm * bn)).
      barrier_p tm eA eB bid m1 m2 it tid
  ensures
    forall+ (tid : natlt (bm/tm * bn)).
      barrier_q tm eA eB bid m1 m2 it tid
