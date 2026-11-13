module KPlay.Proof.Optimized.Head

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 50"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module C = FStar.Classical
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Approx = Kuiper.Approximates.Base
module Cast = Kuiper.Float.Casts.Base
module Lit = Kuiper.Float32.Literals
module Root = FStar.Math.Sqrt
module N = KPlay.Proof.Optimized.NormMath
module Strided = KPlay.Proof.Optimized.Strided
module Masks = KPlay.Proof.Optimized.Dot
module Scalar = KPlay.Proof.Optimized.Scalar
module R = KPlay.Proof.Shared.Recurrent

let square_dot (#n:Strided.width) (x:lseq real n)
  : Lemma (
      rsum (S.init_ghost n (Strided.term #n (x,x))) == N.sum_squares x n) =
  prefix_init n (Strided.term #n (x,x));
  prefix_ext n (Strided.term #n (x,x)) (N.square_cell x);
  N.squares_prefix x n

let lift_square (#n:Strided.width) (x:lseq f32 n) (value:f32)
  : Lemma
      (requires forall (m:Strided.interpretation n). Strided.inputs x x m ==>
        value %~ rsum (S.init_ghost n (Strided.term m)))
      (ensures forall (m:lseq real n). x %~ m ==> value %~ (N.sum_squares m n <: real)) =
  let related (m:lseq real n{x %~ m})
    : Lemma (value %~ (N.sum_squares m n <: real)) =
    square_dot m;
    assert (Strided.inputs x x (m,m))
  in C.forall_intro related

noextract
fn squared (#n:erased Strided.width) (x:larray f32 n)
  (count:sz{SZ.v count == n}) (#before:erased (lseq f32 n)) (#fraction:perm)
  preserves gpu ** x |-> Frac fraction before
  returns result:f32
  ensures pure (forall (m:lseq real n).
    before %~ m ==> result %~ (N.sum_squares m n <: real))
{
  A.share x;
  let result = Strided.strided_dot x x count;
  A.gather x;
  lift_square before result;
  result
}

let sqrt128_literal ()
  : Lemma (F.of_literal #f32 "11.313708498984761" %~ (Root.sqrt 128.0R <: real)) =
  Root.sqrt_square 128.0R;
  Root.sqrt_positive 128.0R;
  assert (
    11313708498984760.0R /. 1000000000000000.0R <=. Root.sqrt 128.0R /\
    Root.sqrt 128.0R <=. 11313708498984762.0R /. 1000000000000000.0R);
  assert_norm (Lit.common_normal_cell "11.313708498984761"
    11313708498984760 1000000000000000
    11313708498984762 1000000000000000 11863283 (-20));
  FP.literal_interval_approx "11.313708498984761"
    11313708498984760 1000000000000000
    11313708498984762 1000000000000000 11863283 (-20) (Root.sqrt 128.0R)

inline_for_extraction noextract
let query_factor : f32 =
  F.div (F.one #f32) (F.of_literal #f32 "11.313708498984761")

let query_factor_approx ()
  : Lemma (query_factor %~ (Root.rsqrt 128.0R <: real)) =
  sqrt128_literal ();
  Approx.div_approx (F.one #f32) (F.of_literal #f32 "11.313708498984761")
    1.0R (Root.sqrt 128.0R)

inline_for_extraction noextract
let l2_scale (x:lseq real 128) : GTot Root.rpos =
  Root.rsqrt (N.sum_squares x 128 +. 0.000001R)

inline_for_extraction noextract
let normalized (query:bool) (x:lseq real 128) : GTot (lseq real 128) =
  S.init_ghost 128 (fun i ->
    S.index x i *. (l2_scale x *. (if query then Root.rsqrt 128.0R else 1.0R)))

let l2_finish (x:lseq f32 128) (argument scale:f32)
  : Lemma
      (requires
        (forall (m:lseq real 128). x %~ m ==>
          argument %~ (N.sum_squares m 128 +. 0.000001R <: real)) /\
        (forall (r:Root.rpos). argument %~ (r <: real) ==>
          scale %~ (1.0R /. Root.sqrt r)))
      (ensures forall (m:lseq real 128). x %~ m ==> scale %~ (l2_scale m <: real)) =
  let related (m:lseq real 128{x %~ m})
    : Lemma (scale %~ (l2_scale m <: real)) = ()
  in C.forall_intro related

(* reduced.cu:213-220 computes both scales before the interleaved stores. *)
noextract
fn normalize_qk (query key:larray f32 128)
  (#qv #kv:erased (lseq f32 128))
  preserves gpu
  requires query |-> qv ** key |-> kv
  ensures exists* (after_q after_k:lseq f32 128).
    query |-> after_q ** key |-> after_k **
    pure (
      (forall (m:lseq real 128). qv %~ m ==> after_q %~ normalized true m) /\
      (forall (m:lseq real 128). kv %~ m ==> after_k %~ normalized false m))
{
  N.epsilon_approx ();
  query_factor_approx ();
  let qq = squared query 128sz;
  let qa = FP.add_rn_ftz qq N.epsilon;
  lift_add (fun (m:lseq real 128) -> qv %~ m)
    (fun m -> (N.sum_squares m 128 <: real)) (fun _ -> 0.000001R)
    qq N.epsilon qa;
  let qs = FP.rsqrt_approx_ftz qa;
  l2_finish qv qa qs;
  let qscale = FP.mul_rn_ftz qs query_factor;
  lift_mul (fun (m:lseq real 128) -> qv %~ m)
    (fun m -> (l2_scale m <: real)) (fun _ -> (Root.rsqrt 128.0R <: real))
    qs query_factor qscale;
  let kk = squared key 128sz;
  let ka = FP.add_rn_ftz kk N.epsilon;
  lift_add (fun (m:lseq real 128) -> kv %~ m)
    (fun m -> (N.sum_squares m 128 <: real)) (fun _ -> 0.000001R)
    kk N.epsilon ka;
  let kscale = FP.rsqrt_approx_ftz ka;
  l2_finish kv ka kscale;
  let mut index = 0sz;
  while (!index <^ 128sz)
    invariant exists* (i:sz{i <= 128}) (q k:lseq f32 128).
      index |-> i ** query |-> q ** key |-> k **
      pure (forall (j:nat{SZ.v i <= j /\ j < 128}).
        S.index q j == S.index qv j /\ S.index k j == S.index kv j) **
      pure (forall (m:lseq real 128). qv %~ m ==>
        (forall (j:nat{j < SZ.v i}). S.index q j %~ S.index (normalized true m) j)) **
      pure (forall (m:lseq real 128). kv %~ m ==>
        (forall (j:nat{j < SZ.v i}). S.index k j %~ S.index (normalized false m) j))
    decreases (128 - !index)
  {
    let i = !index;
    let q = query.(i);
    let qnext = FP.mul_rn_ftz q qscale;
    lift_mul (fun (m:lseq real 128) -> qv %~ m)
      (fun m -> S.index m (SZ.v i))
      (fun m -> l2_scale m *. Root.rsqrt 128.0R) q qscale qnext;
    query.(i) <- qnext;
    let k = key.(i);
    let knext = FP.mul_rn_ftz k kscale;
    lift_mul (fun (m:lseq real 128) -> kv %~ m)
      (fun m -> S.index m (SZ.v i)) (fun m -> (l2_scale m <: real))
      k kscale knext;
    key.(i) <- knext;
    index := i +^ 1sz;
  }
}

inline_for_extraction noextract
let warp_square (x:lseq real 128) (warp:nat) : GTot real =
  if warp < 4 then prefix 32 (fun lane -> N.square_cell x (warp * 32 + lane))
  else 0.0R

let warp_square_at (warp:nat{warp < 4}) (x:lseq real 128)
  : Lemma (warp_square x warp ==
      prefix 32 (fun lane -> N.square_cell x (warp * 32 + lane))) = ()

let gated_squares (x:lseq real 128)
  : Lemma (prefix 32 (warp_square x) == N.sum_squares x 128) =
  prefix_ext 4 (warp_square x)
    (fun warp -> prefix 32 (fun lane -> N.square_cell x (warp * 32 + lane)));
  prefix_blocks 4 32 (N.square_cell x);
  N.squares_prefix x 128;
  prefix_split 4 28 (warp_square x);
  prefix_ext 28 (fun i -> warp_square x (4 + i)) (fun _ -> 0.0R);
  prefix_zero 28

let gated_finish (x:lseq f32 128) (argument result:f32)
  : Lemma
      (requires
        (forall (m:lseq real 128). x %~ m ==>
          argument %~ (N.sum_squares m 128 /. 128.0R +. 0.000001R <: real)) /\
        (forall (r:Root.rpos). argument %~ (r <: real) ==>
          result %~ (1.0R /. Root.sqrt r)))
      (ensures forall (m:lseq real 128). x %~ m ==> result %~ (N.scale m <: real)) =
  let related (m:lseq real 128{x %~ m})
    : Lemma (result %~ (N.scale m <: real)) = ()
  in C.forall_intro related

noextract
fn gated_rstd (input:larray f32 128)
  (#before:erased (lseq f32 128)) (#fraction:perm)
  preserves gpu ** input |-> Frac fraction before
  returns result:f32
  ensures pure (forall (m:lseq real 128).
    before %~ m ==> result %~ (N.scale m <: real))
{
  let mut warps:A.array f32 = [| 32sz |];
  let mut lanes:A.array f32 = [| 32sz |];
  let mut warp = 0sz;
  while (!warp <^ 4sz)
    invariant exists* (w:sz{w <= 4}) (wv:lseq (option f32) 32).
      warp |-> w ** A.pts_to_mask warps wv Masks.all_cells **
      pure (Masks.initialized wv (SZ.v w)) **
      pure (forall (m:lseq real 128). before %~ m ==>
        (forall (j:nat{j < SZ.v w}).
          Masks.option_models (S.index wv j) (warp_square m j)))
    invariant exists* (lv:lseq (option f32) 32). A.pts_to_mask lanes lv Masks.all_cells
    decreases (4 - !warp)
  {
    let w = !warp;
    let mut lane = 0sz;
    while (!lane <^ 32sz)
      invariant exists* (i:sz{i <= 32}) (lv:lseq (option f32) 32).
        lane |-> i ** A.pts_to_mask lanes lv Masks.all_cells **
        pure (Masks.initialized lv (SZ.v i)) **
        pure (forall (m:lseq real 128). before %~ m ==>
          (forall (j:nat{j < SZ.v i}). Masks.option_models (S.index lv j)
            (N.square_cell m (SZ.v w * 32 + j))))
      decreases (32 - !lane)
    {
      let i = !lane;
      let v = input.(w *^ 32sz +^ i);
      let square = FP.fma_rn_ftz v v (zero #f32);
      lift_fma (fun (m:lseq real 128) -> before %~ m)
        (fun m -> S.index m (SZ.v w * 32 + SZ.v i))
        (fun m -> S.index m (SZ.v w * 32 + SZ.v i))
        (fun _ -> 0.0R) v v (zero #f32) square;
      A.mask_write lanes i square;
      lane := i +^ 1sz;
    };
    A.from_mask lanes;
    with values. assert lanes |-> values;
    let warp_total = sum32 lanes;
    lift_sum32 (fun (m:lseq real 128) -> before %~ m)
      (fun m lane -> N.square_cell m (SZ.v w * 32 + lane)) values warp_total;
    C.forall_intro (warp_square_at (SZ.v w));
    assert pure (forall (m:lseq real 128). before %~ m ==>
      warp_total %~ warp_square m (SZ.v w));
    A.to_mask lanes;
    A.mask_write warps w warp_total;
    warp := w +^ 1sz;
  };
  while (!warp <^ 32sz)
    invariant exists* (w:sz{4 <= w /\ w <= 32}) (wv:lseq (option f32) 32).
      warp |-> w ** A.pts_to_mask warps wv Masks.all_cells **
      pure (Masks.initialized wv (SZ.v w)) **
      pure (forall (m:lseq real 128). before %~ m ==>
        (forall (j:nat{j < SZ.v w}).
          Masks.option_models (S.index wv j) (warp_square m j)))
    decreases (32 - !warp)
  {
    let w = !warp;
    A.mask_write warps w (zero #f32);
    warp := w +^ 1sz;
  };
  A.from_mask warps;
  with values. assert warps |-> values;
  let squared_total = sum32 warps;
  lift_sum32 (fun (m:lseq real 128) -> before %~ m) warp_square values squared_total;
  C.forall_intro gated_squares;
  N.inverse_width_approx 128;
  N.epsilon_approx ();
  let argument = FP.fma_rn_ftz squared_total (N.inverse_width 128) N.epsilon;
  lift_fma (fun (m:lseq real 128) -> before %~ m)
    (fun m -> (N.sum_squares m 128 <: real))
    (fun _ -> 1.0R /. 128.0R) (fun _ -> 0.000001R)
    squared_total (N.inverse_width 128) N.epsilon argument;
  let result = FP.rsqrt_approx_ftz argument;
  gated_finish before argument result;
  A.to_mask warps;
  result
}

inline_for_extraction noextract
let gated_model (w x z:lseq real 128) : GTot (lseq real 128) =
  S.init_ghost 128 (fun i ->
    ((S.index x i *. N.scale x) *. S.index w i) *. R.silu (S.index z i))

inline_for_extraction noextract
let gated_interpretation = lseq real 128 & lseq real 128 & lseq real 128

inline_for_extraction noextract
let gated_inputs (w:lseq bf16 128) (x z:lseq f32 128)
  (m:gated_interpretation) : prop =
  w %~ m._1 /\ x %~ m._2 /\ z %~ m._3

noextract
fn gate_readout (weights:larray bf16 128) (readout gates:larray f32 128)
  (#wv:erased (lseq bf16 128)) (#xv #zv:erased (lseq f32 128)) (#fw #fz:perm)
  preserves gpu ** weights |-> Frac fw wv ** gates |-> Frac fz zv
  requires readout |-> xv
  ensures exists* (after:lseq f32 128). readout |-> after **
    pure (forall (m:gated_interpretation).
      gated_inputs wv xv zv m ==> after %~ gated_model m._1 m._2 m._3)
{
  let scale = gated_rstd readout;
  let mut index = 0sz;
  while (!index <^ 128sz)
    invariant exists* (i:sz{i <= 128}) (values:lseq f32 128).
      index |-> i ** readout |-> values **
      pure (forall (j:nat{SZ.v i <= j /\ j < 128}). S.index values j == S.index xv j) **
      pure (forall (m:gated_interpretation).
        gated_inputs wv xv zv m ==> (forall (j:nat{j < SZ.v i}).
          S.index values j %~ S.index (gated_model m._1 m._2 m._3) j))
    decreases (128 - !index)
  {
    let i = !index;
    let value = readout.(i);
    let scaled = FP.mul_rn_ftz value scale;
    lift_mul #gated_interpretation (gated_inputs wv xv zv)
      (fun m -> S.index m._2 (SZ.v i)) (fun m -> (N.scale m._2 <: real))
      value scale scaled;
    let gain = Cast.cast_bf16_to_f32 (weights.(i));
    let normed = FP.mul_rn_ftz scaled gain;
    lift_mul #gated_interpretation (gated_inputs wv xv zv)
      (fun m -> S.index m._2 (SZ.v i) *. N.scale m._2)
      (fun m -> S.index m._1 (SZ.v i)) scaled gain normed;
    let z = gates.(i);
    let activated = Scalar.silu32 z;
    let result = FP.mul_rn_ftz normed activated;
    lift_mul #gated_interpretation (gated_inputs wv xv zv)
      (fun m -> (S.index m._2 (SZ.v i) *. N.scale m._2) *. S.index m._1 (SZ.v i))
      (fun m -> R.silu (S.index m._3 (SZ.v i))) normed activated result;
    readout.(i) <- result;
    index := i +^ 1sz;
  }
}
