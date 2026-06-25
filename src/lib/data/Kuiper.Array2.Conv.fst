
// TODO: should probably just be removed when we get rid of Array2

module Kuiper.Array2.Conv
friend Kuiper.Array2
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.EMatrix

module T = Kuiper.Tensor
module A2 = Kuiper.Array2

let a2_roundtrip (#et:Type0) (#rows #cols:erased nat) (#l:A2.layout rows cols) (a:tensor et l)
  : Lemma (ensures A2.as_tensor (A2.from_tensor a) == a)
          [SMTPat (A2.as_tensor (A2.from_tensor a))]
  = ()

inline_for_extraction noextract
fn array2_of_tensor
  (#et:Type0) (#rows #cols:nat) (#l : A2.layout rows cols)
  (g : tensor et l { is_global g })
  (#f:perm) (#s : ematrix et rows cols)
  requires on gpu_loc (g |-> Frac f s)
  returns a : (a : A2.t et l { A2.is_global a })
  ensures on gpu_loc (a |-> Frac f s) ** pure (A2.as_tensor a == g)
{
  A2.lem_as_tensor_pts_to (A2.from_tensor g) #f s;
  A2.lem_as_tensor_global (A2.from_tensor g);
  let a : A2.t et l = A2.from_tensor g;
  rewrite (on gpu_loc (g |-> Frac f s)) as (on gpu_loc (a |-> Frac f s));
  a
}

inline_for_extraction noextract
fn tensor_of_array2
  (#et:Type0) (#rows #cols:nat) (#l : A2.layout rows cols)
  (a : A2.t et l { A2.is_global a })
  (#f:perm) (#s : ematrix et rows cols)
  requires on gpu_loc (a |-> Frac f s)
  returns g : (g : tensor et l { is_global g })
  ensures on gpu_loc (g |-> Frac f s) ** pure (g == A2.as_tensor a)
{
  A2.lem_as_tensor_pts_to a #f s;
  A2.lem_as_tensor_global a;
  let g : tensor et l = A2.as_tensor a;
  rewrite (on gpu_loc (a |-> Frac f s)) as (on gpu_loc (g |-> Frac f s));
  g
}
