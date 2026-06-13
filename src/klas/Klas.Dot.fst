module Klas.Dot

(* See Klas.Dot.fsti for the specification.

   dot = Σ xᵢ·yᵢ: multiply x and y elementwise into a scratch array (copy x,
   then in-place pointwise multiply by y), then sum with the verified parallel
   reduction Kuiper.Kernel.HReduce.reduce. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common { seq_map, (@!) }
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map
module K = Kuiper.Kernel.HReduce

(* The floating-point elementwise products approximate the real elementwise
   products, pointwise (each cell by a_mul). *)
let dot_seq_approx
  (#et:Type0) {| scalar et |} {| real_like et |}
  (#n:nat) (vx vy : lseq et n) (rx ry : lseq real n)
  : Lemma (requires vx %~ rx /\ vy %~ ry)
          (ensures (Map.lseq_map2 mul vx vy) %~ (Map.lseq_map2 ( *. ) rx ry))
  = let lhs = Map.lseq_map2 mul vx vy in
    let rhs = Map.lseq_map2 ( *. ) rx ry in
    let aux (i : natlt n) : Lemma ((lhs @! i) %~ (rhs @! i))
      = assert ((vx @! i) %~ (rx @! i));
        assert ((vy @! i) %~ (ry @! i));
        assert ((lhs @! i) == mul (vx @! i) (vy @! i));
        assert ((rhs @! i) == (rx @! i) *. (ry @! i))
    in
    Classical.forall_intro aux;
    assert (lhs %~ rhs)

inline_for_extraction noextract
fn dot_gen (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp { nth <= max_threads })
  (lena : szp { lena <= max_blocks * max_threads /\ SZ.fits (lena + nth) })
  (x : array1 et (l1_forward lena) { is_global x })
  (y : array1 et (l1_forward lena) { is_global y })
  (#vx : erased (lseq et lena))
  (#vy : erased (lseq et lena))
  (#fx : perm)
  (#fy : perm)
  (rx : erased (lseq real lena))
  (ry : erased (lseq real lena))
  norewrite
  preserves
    cpu ** on gpu_loc (x |-> Frac fx vx) ** on gpu_loc (y |-> Frac fy vy)
  requires
    pure (vx %~ rx /\ vy %~ ry)
  returns
    res : et
  ensures
    pure (res %~ s_dot rx ry)
{
  let tmp = alloc0 #et lena (l1_forward lena);
  memcpy_device_to_device tmp x lena;
  Map.map_gpu2 mul lena tmp y;
  dot_seq_approx (reveal vx) (reveal vy) (reveal rx) (reveal ry);
  assert pure (Seq.equal (seq_map id (Map.lseq_map2 ( *. ) rx ry))
                         (Map.lseq_map2 ( *. ) rx ry));
  let r = K.reduce id id nth lena tmp (Map.lseq_map2 ( *. ) rx ry);
  free tmp;
  r
}

let dot_f16 = dot_gen #f16
let dot_f32 = dot_gen #f32
let dot_f64 = dot_gen #f64
