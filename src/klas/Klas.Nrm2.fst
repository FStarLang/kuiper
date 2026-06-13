module Klas.Nrm2

(* See Klas.Nrm2.fsti for the specification. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common { seq_map, (@!) }
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map
module K = Kuiper.Kernel.HReduce

(* The floating-point elementwise squares approximate the real elementwise
   squares, pointwise (each cell by a_mul). *)
let sq_seq_approx
  (#et:Type0) {| scalar et |} {| real_like et |}
  (#n:nat) (vx : lseq et n) (rx : lseq real n)
  : Lemma (requires vx %~ rx)
          (ensures (Map.lseq_map2 mul vx vx) %~ (Map.lseq_map2 ( *. ) rx rx))
  = let lhs = Map.lseq_map2 mul vx vx in
    let rhs = Map.lseq_map2 ( *. ) rx rx in
    let aux (i : natlt n) : Lemma ((lhs @! i) %~ (rhs @! i))
      = assert ((vx @! i) %~ (rx @! i));
        assert ((lhs @! i) == mul (vx @! i) (vx @! i));
        assert ((rhs @! i) == (rx @! i) *. (rx @! i))
    in
    Classical.forall_intro aux;
    assert (lhs %~ rhs)

inline_for_extraction noextract
fn nrm2_gen (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp { nth <= max_threads })
  (lena : szp { lena <= max_blocks * max_threads /\ SZ.fits (lena + nth) })
  (x : array1 et (l1_forward lena) { is_global x })
  (#vx : erased (lseq et lena))
  (#fx : perm)
  (rx : erased (lseq real lena))
  norewrite
  preserves
    cpu ** on gpu_loc (x |-> Frac fx vx)
  requires
    pure (vx %~ rx)
  returns
    res : et
  ensures
    pure (res %~ s_nrm2 rx)
{
  let tmp = alloc0 #et lena (l1_forward lena);
  memcpy_device_to_device tmp x lena;
  Map.map_gpu2 mul lena tmp x;
  sq_seq_approx (reveal vx) (reveal rx);
  assert pure (Seq.equal (seq_map id (Map.lseq_map2 ( *. ) rx rx))
                         (Map.lseq_map2 ( *. ) rx rx));
  let s = K.reduce id id nth lena tmp (Map.lseq_map2 ( *. ) rx rx);
  free tmp;
  let res = sqrt s;
  res
}

let nrm2_f16 = nrm2_gen #f16
let nrm2_f32 = nrm2_gen #f32
let nrm2_f64 = nrm2_gen #f64
