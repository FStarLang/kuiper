module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as row_major, l2_col_major as col_major }
open Kuiper.Array2.Strided
module S = KPlay.SharedSpec
module A = Pulse.Lib.Array
module GA = Kuiper.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module MMA = Kuiper.TensorCore.MMA
module G = Kuiper.Spec.GEMM
module EM = Kuiper.EMatrix
module C = Kuiper.Float.Casts
module R = FStar.Math.Sqrt
module Proof = FStar.Tactics.V2
module IC = FStar.Int.Cast
module I64 = FStar.Int64
module BF = Kuiper.BFloat16
module E = FStar.Math.Exp
module SQ = FStar.Math.Sqrt
module Fast = Kuiper.Float32.FastMath
open Pulse.Lib.Array

#set-options "--fuel 4 --ifuel 4 --z3rlimit 10"

inline_for_extraction noextract
let sum_zero (f:S.vector 0) : Lemma (S.sum 0 f == 0.0R) =
  norm_spec [delta_once ["KPlay.SharedSpec.sum"]] (S.sum 0 f)

inline_for_extraction noextract
let sum_step (n:pos) (f:S.vector n)
  : Lemma (S.sum n f == S.sum (n - 1) (fun i -> f i) +. f (n - 1)) =
  norm_spec [delta_once ["KPlay.SharedSpec.sum"]] (S.sum n f)

inline_for_extraction noextract
let rec sum_ext (n:nat) (f g:S.vector n)
  : Lemma
      (requires forall (i:S.index n). f i == g i)
      (ensures S.sum n f == S.sum n g)
      (decreases n) =
  if n = 0 then (sum_zero f; sum_zero g)
  else (
    sum_ext (n - 1) (fun i -> f i) (fun i -> g i);
    sum_step n f;
    sum_step n g
  )

inline_for_extraction noextract
let rec sum_plus (n:nat) (f g:S.vector n)
  : Lemma (
      S.sum n (fun i -> f i +. g i) == S.sum n f +. S.sum n g)
      (decreases n) =
  if n = 0 then (
    sum_zero f;
    sum_zero g;
    sum_zero (fun i -> f i +. g i)
  ) else (
    sum_plus (n - 1) (fun i -> f i) (fun i -> g i);
    sum_step n f;
    sum_step n g;
    sum_step n (fun i -> f i +. g i)
  )

inline_for_extraction noextract
let rec sum_split (n m:nat) (f:S.vector (n + m))
  : Lemma (
      S.sum (n + m) f ==
      S.sum n (fun i -> f i) +. S.sum m (fun i -> f (n + i)))
      (decreases m) =
  if m = 0 then (
    sum_zero (fun i -> f (n + i));
    sum_ext n f (fun i -> f i)
  ) else (
    let lower : S.vector (n + m - 1) = fun i -> f i in
    sum_split n (m - 1) lower;
    sum_ext (n + m - 1) lower (fun i -> f i);
    sum_ext n (fun i -> lower i) (fun i -> f i);
    sum_ext (m - 1) (fun i -> lower (n + i)) (fun i -> f (n + i));
    assert (S.sum (n + m - 1) (fun i -> f i) ==
      S.sum n (fun i -> f i) +. S.sum (m - 1) (fun i -> f (n + i)));
    sum_step (n + m) f;
    sum_step m (fun i -> f (n + i));
    assert (f (n + m - 1) == f (n + (m - 1)));
    assert (S.sum m (fun i -> f (n + i)) ==
      S.sum (m - 1) (fun i -> f (n + i)) +. f (n + m - 1))
  )

inline_for_extraction noextract
let sum_halves (n:pos) (f:S.vector (2 * n))
  : Lemma (
      S.sum n (fun i -> f i +. f (i + n)) == S.sum (2 * n) f) =
  sum_split n n f;
  sum_plus n (fun i -> f i) (fun i -> f (i + n));
  sum_ext n (fun i -> f (i + n)) (fun i -> f (n + i))

inline_for_extraction noextract
let rec sum_groups4 (n:nat) (f:S.vector (4 * n))
  : Lemma (
      S.sum n (fun g -> S.sum 4 (fun j -> f (4 * g + j))) ==
      S.sum (4 * n) f)
      (decreases n) =
  if n = 0 then (
    sum_zero (fun g -> S.sum 4 (fun j -> f (4 * g + j)));
    sum_zero f
  ) else (
    sum_groups4 (n - 1) (fun i -> f i);
    sum_step n (fun g -> S.sum 4 (fun j -> f (4 * g + j)));
    sum_split (4 * (n - 1)) 4 f
  )

inline_for_extraction noextract
let rec sum_squares_bridge (n:nat) (x:S.vector n)
  : Lemma (S.sum n (fun i -> x i *. x i) == S.sum_squares n x)
      (decreases n) =
  norm_spec [delta_once ["KPlay.SharedSpec.sum_squares"]] (S.sum_squares n x);
  if n = 0 then sum_zero (fun i -> x i *. x i)
  else (
    sum_squares_bridge (n - 1) (fun i -> x i);
    sum_step n (fun i -> x i *. x i)
  )

inline_for_extraction noextract
let rec sum_scale (n:nat) (f:S.vector n) (c:real)
  : Lemma (S.sum n (fun i -> c *. f i) == c *. S.sum n f)
      (decreases n) =
  if n = 0 then (
    sum_zero f;
    sum_zero (fun i -> c *. f i)
  ) else (
    sum_scale (n - 1) (fun i -> f i) c;
    sum_step n f;
    sum_step n (fun i -> c *. f i)
  )

inline_for_extraction noextract
let rec positive_sum_bridge (n:pos) (f:S.index n -> R.rpos)
  : Lemma (S.sum n f == S.sum_positive n f)
      (decreases n) =
  norm_spec [delta_once ["KPlay.SharedSpec.sum_positive"]] (S.sum_positive n f);
  sum_step n f;
  if n = 1 then sum_zero (fun i -> f i)
  else (
    let lower : S.index (n - 1) -> R.rpos = fun i -> f i in
    positive_sum_bridge (n - 1) lower;
    sum_ext (n - 1) lower (fun i -> f i)
  )

inline_for_extraction noextract
let instantiate3 (p:real -> real -> real -> prop) (x y z:real)
  : Lemma (requires forall (a b c:real). p a b c)
          (ensures p x y z) =
  let first : squash (forall (b c:real). p x b c) =
    FStar.Classical.Sugar.forall_elim x ()
  in
  let second : squash (forall (c:real). p x y c) =
    FStar.Classical.Sugar.forall_elim y first
  in
  let third : squash (p x y z) =
    FStar.Classical.Sugar.forall_elim z second
  in ()

inline_for_extraction noextract
let instantiate_fma
  (x y z result:f32) (xr yr zr:real)
  : Lemma
      (requires
        (forall (ar br cr:real).
          v_approximates x ar /\ v_approximates y br /\ v_approximates z cr ==>
          v_approximates result (ar *. br +. cr)) /\
        v_approximates x xr /\ v_approximates y yr /\ v_approximates z zr)
      (ensures v_approximates result (xr *. yr +. zr)) =
  instantiate3
    (fun ar br cr ->
      v_approximates x ar /\ v_approximates y br /\ v_approximates z cr ==>
      v_approximates result (ar *. br +. cr))
    xr yr zr

inline_for_extraction noextract
let multiply_reciprocal (x:real) (y:real{y =!= 0.0R})
  : Lemma (x *. (1.0R /. y) == x /. y) = ()

inline_for_extraction noextract
let epsilon_value () : Lemma (1.0R /. 1000000.0R == 0.000001R) = ()

inline_for_extraction noextract
fn rounded_mul
  (#xr #yr:erased real)
  (x y:f32)
  preserves gpu
  requires pure (v_approximates x (reveal xr) /\ v_approximates y (reveal yr))
  returns result:f32
  ensures pure (v_approximates result (reveal xr *. reveal yr))
{
  let minus_one : f32 = of_int (-1L);
  of_int_approx #f32 (-1L);
  let negative_zero = mul minus_one (zero #f32);
  a_mul minus_one (zero #f32) (0.0R -. 1.0R) 0.0R;
  let result = Fast.fma_rn x y negative_zero;
  instantiate_fma x y negative_zero result (reveal xr) (reveal yr) 0.0R;
  result
}

inline_for_extraction noextract
fn rounded_add
  (#xr #yr:erased real)
  (x y:f32)
  preserves gpu
  requires pure (v_approximates x (reveal xr) /\ v_approximates y (reveal yr))
  returns result:f32
  ensures pure (v_approximates result (reveal xr +. reveal yr))
{
  let result = Fast.fma_rn x (one #f32) y;
  instantiate_fma x (one #f32) y result (reveal xr) 1.0R (reveal yr);
  result
}

inline_for_extraction noextract
let bf_rep (n:nat) (storage:lseq bf16 n) (logical:S.vector n) : prop =
  forall (i:S.index n). v_approximates (Seq.index storage i) (logical i)

inline_for_extraction noextract
let vector_at (#n:nat) (logical:erased (S.vector n)) (i:nat{i < n}) : GTot real =
  (reveal #(S.vector n) logical) i

inline_for_extraction noextract
let bounded_sum (#n:nat) (count:nat{count <= n}) (terms:S.vector n) : real =
  S.sum count (fun i -> terms i)

inline_for_extraction noextract
let bounded_sum_step (#n:nat) (count:nat{count < n}) (terms:S.vector n)
  : Lemma (
      bounded_sum (count + 1) terms == bounded_sum count terms +. terms count) =
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]] (bounded_sum (count + 1) terms);
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]] (bounded_sum count terms);
  sum_step (count + 1) (fun i -> terms i);
  sum_ext count (fun (i:S.index (count + 1 - 1)) -> terms i) (fun i -> terms i)

inline_for_extraction noextract
let bounded_sum_full (#n:nat) (terms:S.vector n)
  : Lemma (bounded_sum n terms == S.sum n terms) =
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]] (bounded_sum n terms);
  sum_ext n (fun i -> terms i) terms

inline_for_extraction noextract
let partial_rep
  (n:nat{n <= 256}) (storage:lseq f32 256) (logical:S.vector 256) : prop =
  forall (i:S.index n). v_approximates (Seq.index storage i) (logical i)

inline_for_extraction noextract
let partial_sum (n:nat{n <= 256}) (logical:S.vector 256) : real =
  S.sum n (fun i -> logical i)

inline_for_extraction noextract
let partial_sum_one (logical:S.vector 256)
  : Lemma (partial_sum 1 logical == logical 0) =
  sum_step 1 (fun i -> logical i);
  sum_zero (fun i -> logical i)

inline_for_extraction noextract
let tree_width (n:nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32 \/
  n = 64 \/ n = 128 \/ n = 256

inline_for_extraction noextract
let tree_width_half (n:pos{tree_width n /\ n > 1})
  : Lemma (n % 2 == 0 /\ tree_width (n / 2)) = ()

inline_for_extraction noextract
let merge_partial (half:pos{half <= 128}) (r:S.vector 256) : S.vector 256 =
  fun i -> if i < half then r i +. r (i + half) else r i

inline_for_extraction noextract
let merge_sum (half:pos{half <= 128}) (r:S.vector 256)
  : Lemma (
      partial_sum half (merge_partial half r) ==
      partial_sum (2 * half) r) =
  norm_spec [delta_once ["KPlay.Functional.partial_sum"]]
    (partial_sum half (merge_partial half r));
  norm_spec [delta_once ["KPlay.Functional.partial_sum"]]
    (partial_sum (2 * half) r);
  sum_halves half (fun i -> r i);
  sum_ext half (fun i -> merge_partial half r i) (fun i -> r i +. r (i + half))

inline_for_extraction noextract
fn halve_partial
  (partial:A.larray f32 256)
  (half:szp{half <= 128})
  (#storage:erased (lseq f32 256))
  (#logical:erased (S.vector 256))
  preserves gpu
  requires partial |-> storage
  requires pure (partial_rep (2 * half) storage logical)
  ensures exists* (next:lseq f32 256).
    partial |-> next **
    pure (partial_rep half next (merge_partial half logical))
{
  let mut i = 0sz;
  while (!i <^ half)
    invariant exists* (j:sz{j <= half}) (values:lseq f32 256).
      i |-> j ** partial |-> values **
      pure (forall (k:S.index (2 * half)).
        if k < j then
          v_approximates (Seq.index values k) (merge_partial half logical k)
        else v_approximates (Seq.index values k) (vector_at #256 logical k))
    decreases (half - !i)
  {
    let j = !i;
    let x = partial.(j);
    let y = partial.(j +^ half);
    let next = rounded_add #(vector_at #256 logical j) #(vector_at #256 logical (j + half)) x y;
    partial.(j) <- next;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn tree_reduce
  (partial:A.larray f32 256)
  (width:szp{tree_width width})
  (#storage:erased (lseq f32 256))
  (#logical:erased (S.vector 256))
  preserves gpu
  requires partial |-> storage
  requires pure (partial_rep width storage logical)
  returns result:f32
  ensures exists* (next:lseq f32 256).
    partial |-> next **
    pure (v_approximates result (partial_sum width logical))
{
  let mut active = width;
  let mut current : erased (S.vector 256) = logical;
  while (!active >^ 1sz)
    invariant exists* (n:szp{tree_width n /\ n <= width})
      (r:erased (S.vector 256)) (values:lseq f32 256).
      active |-> n ** current |-> r ** partial |-> values **
      pure (partial_rep n values r /\
        partial_sum n r == partial_sum width logical)
    decreases (SZ.v (!active))
  {
    let n = !active;
    let r = !current;
    tree_width_half n;
    let half = n /^ 2sz;
    let next : erased (S.vector 256) = hide (merge_partial half r);
    halve_partial partial half #_ #r;
    merge_sum half r;
    assert pure (SZ.v n == 2 * SZ.v half);
    current := next;
    active := half;
  };
  let r = !current;
  partial_sum_one r;
  partial.(0sz)
}

inline_for_extraction noextract
let square_terms
  (width:pos{width = 256 \/ width = 1024}) (group:nat{group < width / 4})
  (x:S.vector width) : S.vector 4 =
  fun j -> x (4 * group + j) *. x (4 * group + j)

inline_for_extraction noextract
fn square_group
  (width:szp{SZ.v width = 256 \/ SZ.v width = 1024})
  (input:A.larray bf16 width)
  (group:sz{group < width / 4})
  (#fraction:perm)
  (#storage:erased (lseq bf16 width))
  (#logical:erased (S.vector width))
  preserves gpu ** input |-> Frac fraction storage
  requires pure (bf_rep width storage logical)
  returns result:f32
  ensures pure (v_approximates result
    (S.sum 4 (square_terms width group logical)))
{
  let terms : erased (S.vector 4) = hide (square_terms width group logical);
  let mut j = 0sz;
  let mut accum : f32 = zero;
  while (!j <^ 4sz)
    invariant exists* (k:sz{k <= 4}) (acc:f32).
      j |-> k ** accum |-> acc **
      pure (v_approximates acc (bounded_sum #4 k terms))
    decreases (4 - !j)
  {
    let k = !j;
    let value = input.(4sz *^ group +^ k);
    C.fcast_approx #bf16 #f32 value (vector_at #width logical (4 * group + k));
    let value32 = C.fcast #bf16 #f32 value;
    let squared = rounded_mul
      #(vector_at #width logical (4 * group + k)) #(vector_at #width logical (4 * group + k))
      value32 value32;
    let acc = !accum;
    let next = rounded_add
      #(bounded_sum #4 k terms)
      #(vector_at #width logical (4 * group + k) *. vector_at #width logical (4 * group + k))
      acc squared;
    bounded_sum_step #4 k terms;
    accum := next;
    j := k +^ 1sz;
  };
  bounded_sum_full #4 terms;
  !accum
}

inline_for_extraction noextract
let grouped_squares (width:pos{width = 256 \/ width = 1024}) (x:S.vector width)
  : S.vector 256 =
  fun i ->
    if i < width / 4 then S.sum 4 (square_terms width i x)
    else 0.0R

inline_for_extraction noextract
let grouped_squares_sum
  (width:pos{width = 256 \/ width = 1024}) (x:S.vector width)
  : Lemma (
      partial_sum (width / 4) (grouped_squares width x) == S.sum_squares width x) =
  let point (g:S.index (width / 4))
    : Lemma (
        grouped_squares width x (g <: nat) ==
        S.sum 4 (fun j -> x (4 * g + j) *. x (4 * g + j))) =
    norm_spec [delta_once [
      "KPlay.Functional.grouped_squares"; "KPlay.Functional.square_terms"
    ]] (grouped_squares width x (g <: nat))
  in
  FStar.Classical.forall_intro #(S.index (width / 4)) #(fun g ->
    grouped_squares width x (g <: nat) ==
    S.sum 4 (fun j -> x (4 * g + j) *. x (4 * g + j))) point;
  norm_spec [delta_once ["KPlay.Functional.partial_sum"]]
    (partial_sum (width / 4) (grouped_squares width x));
  sum_ext (width / 4)
    (fun g -> grouped_squares width x (g <: nat))
    (fun g -> S.sum 4 (fun j -> x (4 * g + j) *. x (4 * g + j)));
  sum_groups4 (width / 4) (fun i -> x (i <: nat) *. x (i <: nat));
  sum_ext width
    (fun (i:S.index (4 * (width / 4))) -> x (i <: nat) *. x (i <: nat))
    (fun i -> x i *. x i);
  sum_squares_bridge width x

inline_for_extraction noextract
fn rms_scale
  (width:szp{SZ.v width = 256 \/ SZ.v width = 1024})
  (input:A.larray bf16 width)
  (partial:A.larray f32 256)
  (#fraction:perm)
  (#storage:erased (lseq bf16 width))
  (#before:erased (lseq f32 256))
  (#logical:erased (S.vector width))
  preserves gpu ** input |-> Frac fraction storage
  requires partial |-> before
  requires pure (bf_rep width storage logical)
  returns result:f32
  ensures exists* (next:lseq f32 256). partial |-> next **
    pure (v_approximates result
      (R.rsqrt (S.sum_squares width logical /. FStar.Real.of_int width +. 0.000001R)))
{
  let groups = width /^ 4sz;
  let mut i = 0sz;
  while (!i <^ groups)
    invariant exists* (j:sz{j <= groups}) (values:lseq f32 256).
      i |-> j ** partial |-> values **
      pure (partial_rep j values (grouped_squares width logical))
    decreases (groups - !i)
  {
    let j = !i;
    let group = square_group width input j #fraction #storage #logical;
    partial.(j) <- group;
    i := j +^ 1sz;
  };
  let square_total = tree_reduce partial groups #_ #(grouped_squares width logical);
  grouped_squares_sum width logical;
  let width32 : f32 =
    (if width = 256sz then of_int (256L) else of_int (1024L));
  of_int_approx #f32 256L;
  of_int_approx #f32 1024L;
  let inverse_width = div (one #f32) width32;
  div_approx (one #f32) width32 1.0R (FStar.Real.of_int width);
  let mean = rounded_mul #(S.sum_squares width logical) #(1.0R /. FStar.Real.of_int width)
    square_total inverse_width;
  multiply_reciprocal (S.sum_squares width logical) (FStar.Real.of_int width);
  let million : f32 = of_int 1000000L;
  of_int_approx #f32 1000000L;
  let epsilon = div (one #f32) million;
  div_approx (one #f32) million 1.0R 1000000.0R;
  epsilon_value ();
  let shifted = rounded_add #(S.sum_squares width logical /. FStar.Real.of_int width) #0.000001R
    mean epsilon;
  rsqrt_approx shifted (S.sum_squares width logical /. FStar.Real.of_int width +. 0.000001R);
  rsqrt shifted
}

inline_for_extraction noextract
fn rms_kernel
  (width:szp{SZ.v width = 256 \/ SZ.v width = 1024})
  (input weight:A.larray bf16 width)
  (output:A.larray bf16 width)
  (partial:A.larray f32 256)
  (#fi #fw:perm)
  (#si #sw:erased (lseq bf16 width))
  (#sp:erased (lseq f32 256))
  (#xr #wr:erased (S.vector width))
  preserves gpu ** input |-> Frac fi si ** weight |-> Frac fw sw
  requires partial |-> sp ** A.pts_to_uninit output width
  requires pure (bf_rep width si xr /\ bf_rep width sw wr)
  ensures exists* (so:lseq bf16 width) (next:lseq f32 256).
    output |-> so ** partial |-> next **
    pure (bf_rep width so (S.rms_norm width wr xr))
{
  let scale = rms_scale width input partial #fi #si #sp #xr;
  unfold A.pts_to_uninit output width;
  let mut i = 0sz;
  while (!i <^ width)
    invariant exists* (j:sz{j <= width}) (masked:Seq.seq (option bf16)).
      i |-> j ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == width /\
        (forall (k:S.index width). k < j ==>
          exists (v:bf16). Seq.index masked k == Some v /\
            v_approximates v (S.rms_norm width wr xr k)))
    decreases (width - !i)
  {
    let j = !i;
    let x = input.(j);
    let w = weight.(j);
    C.fcast_approx #bf16 #f32 x (vector_at #width xr j);
    C.fcast_approx #bf16 #f32 w (vector_at #width wr j);
    let scaled = rounded_mul #(vector_at #width xr j)
      #(R.rsqrt (S.sum_squares width xr /. FStar.Real.of_int width +. 0.000001R))
      (C.fcast #bf16 #f32 x) scale;
    let scale_weight = rounded_add #1.0R #(vector_at #width wr j) (one #f32) (C.fcast #bf16 #f32 w);
    let y = rounded_mul
      #(vector_at #width xr j *. R.rsqrt (S.sum_squares width xr /. FStar.Real.of_int width +. 0.000001R))
      #(1.0R +. vector_at #width wr j) scaled scale_weight;
    norm_spec [delta_once ["KPlay.SharedSpec.rms_norm"]] (S.rms_norm width wr xr j);
    C.fcast_approx #f32 #bf16 y (S.rms_norm width wr xr j);
    A.mask_write output j (C.fcast #f32 #bf16 y);
    i := j +^ 1sz;
  };
  A.from_mask output;
}

instance send_initialized
  (#a:Type0) (p:A.array a{GA.is_global_array p}) (#f:perm) (s:Seq.seq a)
  : is_send_across gpu_of (A.pts_to p #f s) =
  GA.is_send_pts_to p #f s

fn execute_rms
  (width:szp{SZ.v width = 256 \/ SZ.v width = 1024})
  (input:A.larray bf16 width{GA.is_global_array input})
  (weight:A.larray bf16 width{GA.is_global_array weight})
  (output:A.larray bf16 width{GA.is_global_array output})
  (#fi #fw:perm)
  (#si #sw:erased (lseq bf16 width))
  (#xr #wr:erased (S.vector width))
  preserves cpu **
    on gpu_loc (input |-> Frac fi si ** weight |-> Frac fw sw)
  requires on gpu_loc (A.pts_to_uninit output width)
  requires pure (bf_rep width si xr /\ bf_rep width sw wr)
  ensures exists* (so:lseq bf16 width).
    on gpu_loc (output |-> so) **
    pure (bf_rep width so (S.rms_norm width wr xr))
{
  let allocation = GA.gpu_array_alloc #f32 256sz;
  let partial : (p:A.larray f32 256{GA.is_global_array p}) = allocation;
  assert rewrites_to partial allocation;
  with sp. assert on gpu_loc (partial |-> sp);
  let initial : erased (lseq f32 256) = hide (sp <: lseq f32 256);
  launch_kernel_1 (fun () -> rms_kernel width input weight output partial #fi #fw #si #sw #initial #xr #wr);
  GA.gpu_array_free partial;
}

inline_for_extraction noextract
let pad16 (n:pos) : pos = ((n + 15) / 16) * 16

inline_for_extraction noextract
let pad16_bounds (n:pos) : Lemma (n <= pad16 n /\ pad16 n < n + 16 /\ pad16 n % 16 == 0) = ()

inline_for_extraction noextract
let padded_matrix (rows cols:pos) (w:S.matrix rows cols)
  : chest2 real (pad16 rows) (pad16 cols) =
  mk2 (fun r c -> if r < rows && c < cols then w r c else 0.0R)

inline_for_extraction noextract
let padded_input (cols:pos) (x:S.vector cols)
  : chest2 real (pad16 cols) 8 =
  mk2 (fun c _ -> if c < cols then x c else 0.0R)

inline_for_extraction noextract
let matrix_rep
  (rows cols:pos) (count:nat)
  (offset rs:nat) (cs:nat{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (storage:lseq bf16 count) (w:S.matrix rows cols) : prop =
  forall (r:S.index rows) (c:S.index cols).
    v_approximates (Seq.index storage (offset + r * rs + c * cs)) (w r c)

inline_for_extraction noextract
let matrix_index_bounds
  (rows cols:pos) (count:nat)
  (offset rs:nat) (cs:nat{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (r:nat{r < rows}) (c:nat{c < cols})
  : Lemma (
      r * rs < count /\ c * cs < count /\
      offset + r * rs < count /\ offset + r * rs + c * cs < count) = ()

inline_for_extraction noextract
let row_index (nr nc:pos) (r:nat{r < nr}) (c:nat{c < nc})
  : Lemma (
      r * nc + c < nr * nc /\ (r * nc + c) / nc == r /\ (r * nc + c) % nc == c) = ()

inline_for_extraction noextract
let row_rep (nr nc:pos) (storage:lseq bf16 (nr * nc)) (logical:chest2 real nr nc) : prop =
  forall (r:nat{r < nr}) (c:nat{c < nc}).
    v_approximates (Seq.index storage (r * nc + c)) (acc2 logical r c)

inline_for_extraction noextract
let flat_matrix_result
  (nr nc:pos) (storage:lseq bf16 (nr * nc)) (logical:chest2 real nr nc)
  : Lemma
      (requires forall (k:nat{k < nr * nc}).
        v_approximates (Seq.index storage k) (acc2 logical (k / nc) (k % nc)))
      (ensures row_rep nr nc storage logical) =
  let point (r:nat{r < nr}) (c:nat{c < nc})
    : Lemma (v_approximates (Seq.index storage (r * nc + c)) (acc2 logical r c)) =
    row_index nr nc r c;
    let _ : squash (v_approximates (Seq.index storage (r * nc + c))
      (acc2 logical ((r * nc + c) / nc) ((r * nc + c) % nc))) =
      FStar.Classical.Sugar.forall_elim (r * nc + c) ()
    in ()
  in
  FStar.Classical.forall_intro_2 #(r:nat{r < nr}) #(fun _ -> c:nat{c < nc})
    #(fun r c -> v_approximates (Seq.index storage (r * nc + c)) (acc2 logical r c)) point

inline_for_extraction noextract
fn read_matrix_element
  (rows:szp{rows <= 4096}) (cols:szp{cols <= 3584}) (count:szp)
  (source:A.larray bf16 count)
  (offset rs:sz) (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (r:sz{r < pad16 rows}) (c:sz{c < pad16 cols})
  (#fraction:perm) (#storage:erased (lseq bf16 count))
  (#w:erased (S.matrix rows cols))
  preserves gpu ** source |-> Frac fraction storage
  requires pure (matrix_rep rows cols count offset rs cs storage w)
  returns result:bf16
  ensures pure (v_approximates result (acc2 (padded_matrix rows cols w) r c))
{
  if (r <^ rows && c <^ cols) {
    matrix_index_bounds rows cols count offset rs cs r c;
    source.(offset +^ r *^ rs +^ c *^ cs)
  } else {
    (zero #bf16)
  }
}

inline_for_extraction noextract
fn prepare_matrix
  (rows:szp{rows <= 4096})
  (cols:szp{cols <= 3584})
  (count:szp)
  (source:A.larray bf16 count)
  (offset rs:sz) (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (dest:A.larray bf16 (pad16 rows * pad16 cols))
  (#fraction:perm)
  (#storage:erased (lseq bf16 count))
  (#before:erased (lseq bf16 (pad16 rows * pad16 cols)))
  (#w:erased (S.matrix rows cols))
  preserves gpu ** source |-> Frac fraction storage
  requires dest |-> before
  requires pure (matrix_rep rows cols count offset rs cs storage w)
  ensures exists* (next:lseq bf16 (pad16 rows * pad16 cols)).
    dest |-> next **
    pure (row_rep (pad16 rows) (pad16 cols) next (padded_matrix rows cols w))
{
  pad16_bounds rows;
  pad16_bounds cols;
  let nr = ((rows +^ 15sz) /^ 16sz) *^ 16sz;
  let nc = ((cols +^ 15sz) /^ 16sz) *^ 16sz;
  assert pure (SZ.v nr == pad16 rows /\ SZ.v nc == pad16 cols);
  let mut i = 0sz;
  while (!i <^ nr *^ nc)
    invariant exists* (j:sz{j <= nr * nc})
      (values:lseq bf16 (pad16 rows * pad16 cols)).
      i |-> j ** dest |-> values **
      pure (forall (k:nat{k < j}).
        v_approximates (Seq.index values k)
          (acc2 (padded_matrix rows cols w) (k / nc) (k % nc)))
    decreases (nr * nc - !i)
  {
    let j = !i;
    let r = j /^ nc;
    let c = j %^ nc;
    let value = read_matrix_element rows cols count source offset rs cs r c #fraction #storage #w;
    dest.(j) <- value;
    i := j +^ 1sz;
  };
  with next. assert dest |-> next;
  flat_matrix_result nr nc next (padded_matrix rows cols w);
  assert pure (row_rep (pad16 rows) (pad16 cols) next (padded_matrix rows cols w));
}

inline_for_extraction noextract
fn read_input_element
  (cols:szp{cols <= 3584}) (source:A.larray bf16 cols)
  (r:sz{r < pad16 cols}) (c:sz{c < 8})
  (#fraction:perm) (#storage:erased (lseq bf16 cols))
  (#x:erased (S.vector cols))
  preserves gpu ** source |-> Frac fraction storage
  requires pure (bf_rep cols storage x)
  returns result:bf16
  ensures pure (v_approximates result (acc2 (padded_input cols x) r c))
{
  if (r <^ cols) { source.(r) } else { (zero #bf16) }
}

inline_for_extraction noextract
fn prepare_input
  (cols:szp{cols <= 3584})
  (source:A.larray bf16 cols)
  (dest:A.larray bf16 (pad16 cols * 8))
  (#fraction:perm)
  (#storage:erased (lseq bf16 cols))
  (#before:erased (lseq bf16 (pad16 cols * 8)))
  (#x:erased (S.vector cols))
  preserves gpu ** source |-> Frac fraction storage
  requires dest |-> before
  requires pure (bf_rep cols storage x)
  ensures exists* (next:lseq bf16 (pad16 cols * 8)).
    dest |-> next **
    pure (row_rep 8 (pad16 cols) next (EM.mtranspose (padded_input cols x)))
{
  pad16_bounds cols;
  let nc = ((cols +^ 15sz) /^ 16sz) *^ 16sz;
  assert pure (SZ.v nc == pad16 cols);
  let mut i = 0sz;
  while (!i <^ nc *^ 8sz)
    invariant exists* (j:sz{j <= nc * 8}) (values:lseq bf16 (pad16 cols * 8)).
      i |-> j ** dest |-> values **
      pure (forall (k:nat{k < j}).
        v_approximates (Seq.index values k)
          (acc2 (padded_input cols x) (k % nc) (k / nc)))
    decreases (nc * 8 - !i)
  {
    let j = !i;
    let r = j %^ nc;
    let value = read_input_element cols source r (j /^ nc) #fraction #storage #x;
    dest.(j) <- value;
    i := j +^ 1sz;
  };
  with next. assert dest |-> next;
  flat_matrix_result 8 nc next (EM.mtranspose (padded_input cols x));
  assert pure (row_rep 8 (pad16 cols) next (EM.mtranspose (padded_input cols x)));
}

inline_for_extraction noextract
let rec mma_accum
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (va:chest2 bf16 nr nc) (vb:chest2 bf16 nc 8)
  (row:nat{row < nr / 16}) (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  : Tot (chest2 f32 16 8) (decreases (finish - start)) =
  if finish = start then const (16 @| 8 @| INil) (zero #f32)
  else
    MMA.emma (mma_accum va vb row start (finish - 1))
      (ematrix_subtile va 16 16 row (finish - 1))
      (ematrix_subtile vb 16 8 (finish - 1) 0)

inline_for_extraction noextract
let block_result
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (va:chest2 bf16 nr nc) (vb:chest2 bf16 nc 8)
  (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  : chest2 f32 nr 8 =
  mk2 (fun r c -> acc2 (mma_accum va vb (r / 16) start finish) (r % 16) c)

inline_for_extraction noextract
instance c8 : concrete_sz 8 = {x = 8sz}

inline_for_extraction noextract
fn tensor_kernel
  (nr:szp{nr <= 4096 /\ nr % 16 == 0})
  (nc:szp{nc <= 3584 /\ nc % 16 == 0})
  (a:array2 bf16 (row_major nr nc))
  (b:array2 bf16 (col_major nc 8))
  (c:array2 f32 (row_major nr 8))
  (start:sz) (finish:sz{start <= finish /\ finish <= nc / 16})
  (#va:chest2 bf16 nr nc)
  (#vb:chest2 bf16 nc 8)
  (#vc:chest2 f32 nr 8)
  (tid:szlt 32)
  ()
  norewrite
  requires gpu **
    (a |-> Frac (1.0R /. 32.0R) va **
     b |-> Frac (1.0R /. 32.0R) vb **
     c |-> Frac (1.0R /. 32.0R) vc) ** thread_id 32 tid
  ensures gpu **
    (a |-> Frac (1.0R /. 32.0R) va **
     b |-> Frac (1.0R /. 32.0R) vb **
     c |-> Frac (1.0R /. 32.0R) (block_result va vb start finish)) ** thread_id 32 tid
{
  let mut row = 0sz;
  while (!row <^ nr /^ 16sz)
    invariant exists* (r:sz{r <= nr / 16}) (values:chest2 f32 nr 8).
      row |-> r ** c |-> Frac (1.0R /. 32.0R) values **
      pure (forall (i:nat{i < r * 16}) (j:nat{j < 8}).
        acc2 values i j == acc2 (block_result va vb start finish) i j)
    decreases (nr / 16 - !row)
  {
    let r = !row;
    array2_extract_tile c 16 8 r 0;
    let ct = array2_subtile c 16 8 (SZ.v r) 0;
    assert rewrites_to ct (array2_subtile c 16 8 (SZ.v r) 0);
    let fragment = MMA.alloc_fragment ();
    MMA.fill fragment (zero #f32);
    let mut k = start;
    while (!k <^ finish)
      invariant exists* (kk:sz{start <= kk /\ kk <= finish}).
        k |-> kk ** fragment |-> mma_accum va vb r start kk
      decreases (finish - !k)
    {
      let kk = !k;
      array2_extract_tile_ro a 16 16 r kk;
      let at = array2_subtile a 16 16 (SZ.v r) (SZ.v kk);
      assert rewrites_to at (array2_subtile a 16 16 (SZ.v r) (SZ.v kk));
      array2_extract_tile_ro b 16 8 kk 0;
      let bt = array2_subtile b 16 8 (SZ.v kk) 0;
      assert rewrites_to bt (array2_subtile b 16 8 (SZ.v kk) 0);
      MMA.mma_sync at bt fragment;
      with av. assert at |-> Frac (1.0R /. 32.0R) av;
      Pulse.Lib.Trade.elim_trade (at |-> Frac (1.0R /. 32.0R) av) (a |-> Frac (1.0R /. 32.0R) va);
      with bv. assert bt |-> Frac (1.0R /. 32.0R) bv;
      Pulse.Lib.Trade.elim_trade (bt |-> Frac (1.0R /. 32.0R) bv) (b |-> Frac (1.0R /. 32.0R) vb);
      norm_spec [delta_once ["KPlay.Functional.mma_accum"]]
        (mma_accum va vb r start (kk + 1));
      k := kk +^ 1sz;
    };
    MMA.store fragment ct;
    with result. assert fragment |-> result;
    drop_ (fragment |-> result);
    with cv. assert ct |-> Frac (1.0R /. 32.0R) cv;
    Pulse.Lib.Forall.elim_forall cv;
    Pulse.Lib.Trade.elim_trade (ct |-> Frac (1.0R /. 32.0R) cv) _;
    row := r +^ 1sz;
  };
  with result. assert c |-> Frac (1.0R /. 32.0R) result;
  EM.lemma_equal_intro result (block_result va vb start finish);
  Kuiper.Chest.ext result (block_result va vb start finish);
  assert pure (result == block_result va vb start finish);
}

inline_for_extraction noextract
let rec real_accum
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (row:nat{row < nr / 16}) (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  : Tot (chest2 real 16 8) (decreases (finish - start)) =
  if finish = start then const (16 @| 8 @| INil) 0.0R
  else
    G.matplus (real_accum ra rb row start (finish - 1))
      (G.matmul (ematrix_subtile ra 16 16 row (finish - 1))
        (ematrix_subtile rb 16 8 (finish - 1) 0))

inline_for_extraction noextract
let rec mma_accum_approx
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (va:chest2 bf16 nr nc) (vb:chest2 bf16 nc 8)
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (row:nat{row < nr / 16}) (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  : Lemma
      (requires va %~ ra /\ vb %~ rb)
      (ensures mma_accum va vb row start finish %~ real_accum ra rb row start finish)
      (decreases (finish - start)) =
  norm_spec [delta_once ["KPlay.Functional.mma_accum"]] (mma_accum va vb row start finish);
  norm_spec [delta_once ["KPlay.Functional.real_accum"]] (real_accum ra rb row start finish);
  if finish = start then ()
  else (
    mma_accum_approx va vb ra rb row start (finish - 1);
    MMA.emma_approx_lemma
      (mma_accum va vb row start (finish - 1))
      (ematrix_subtile va 16 16 row (finish - 1))
      (ematrix_subtile vb 16 8 (finish - 1) 0)
      (real_accum ra rb row start (finish - 1))
      (ematrix_subtile ra 16 16 row (finish - 1))
      (ematrix_subtile rb 16 8 (finish - 1) 0)
  )

inline_for_extraction noextract
let dot_terms (#nr #nc:nat)
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (r:nat{r < nr}) (c:nat{c < 8}) : S.vector nc =
  fun k -> acc2 ra r k *. acc2 rb k c

inline_for_extraction noextract
let segment_sum (#n:nat) (start:nat) (finish:nat{start <= finish /\ finish <= n})
  (terms:S.vector n) : real =
  S.sum (finish - start) (fun i -> terms (start + i))

inline_for_extraction noextract
let segment_step (#n:nat) (start:nat) (middle:nat{start <= middle})
  (finish:nat{middle <= finish /\ finish <= n}) (terms:S.vector n)
  : Lemma (
      segment_sum start finish terms ==
      segment_sum start middle terms +. segment_sum middle finish terms) =
  sum_split (middle - start) (finish - middle) (fun i -> terms (start + i));
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]] (segment_sum start finish terms);
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]] (segment_sum start middle terms);
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]] (segment_sum middle finish terms);
  sum_ext (finish - start)
    (fun (i:S.index (middle - start + (finish - middle))) -> terms (start + i))
    (fun i -> terms (start + i));
  sum_ext (finish - middle)
    (fun i -> terms (start + (middle - start + i))) (fun i -> terms (middle + i))

inline_for_extraction noextract
let rec tile_dot
  (a:chest2 real 16 16) (b:chest2 real 16 8)
  (r:nat{r < 16}) (c:nat{c < 8}) (n:nat{n <= 16})
  : Lemma (
      G.__matmul_single a b r c n == bounded_sum n (dot_terms a b r c))
      (decreases n) =
  if n = 0 then (
    G.__gmatmul_single_zero_lemma 0.0R ( *. ) ( +. ) a b r c;
    norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
      (bounded_sum 0 (dot_terms a b r c))
  ) else (
    tile_dot a b r c (n - 1);
    G.matmul_single_lemma a b r c n;
    bounded_sum_step (n - 1) (dot_terms a b r c)
  )

inline_for_extraction noextract
let tile_sum
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (row:nat{row < nr / 16}) (block:nat{block < nc / 16})
  (r:nat{r < 16}) (c:nat{c < 8})
  : Lemma (
      acc2 (G.matmul (ematrix_subtile ra 16 16 row block)
        (ematrix_subtile rb 16 8 block 0)) r c ==
      segment_sum (16 * block) (16 * (block + 1)) (dot_terms ra rb (16 * row + r) c)) =
  let a = ematrix_subtile ra 16 16 row block in
  let b = ematrix_subtile rb 16 8 block 0 in
  G.lemma_matmul_index a b r c;
  tile_dot a b r c 16;
  bounded_sum_full (dot_terms a b r c);
  sum_ext 16
    (dot_terms a b r c)
    (fun k -> dot_terms ra rb (16 * row + r) c (16 * block + k));
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]]
    (segment_sum (16 * block) (16 * (block + 1)) (dot_terms ra rb (16 * row + r) c));
  sum_ext 16
    (fun (i:S.index (16 * (block + 1) - 16 * block)) ->
      dot_terms ra rb (16 * row + r) c (16 * block + i))
    (fun i -> dot_terms ra rb (16 * row + r) c (16 * block + i))

inline_for_extraction noextract
let rec real_accum_dot
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (row:nat{row < nr / 16}) (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  (r:nat{r < 16}) (c:nat{c < 8})
  : Lemma (
      acc2 (real_accum ra rb row start finish) r c ==
      segment_sum (16 * start) (16 * finish) (dot_terms ra rb (16 * row + r) c))
      (decreases (finish - start)) =
  norm_spec [delta_once ["KPlay.Functional.real_accum"]] (real_accum ra rb row start finish);
  if finish = start then (
    norm_spec [delta_once ["KPlay.Functional.segment_sum"]]
      (segment_sum (16 * start) (16 * finish) (dot_terms ra rb (16 * row + r) c))
  ) else (
    real_accum_dot ra rb row start (finish - 1) r c;
    tile_sum ra rb row (finish - 1) r c;
    EM.chest_comb_acc2 (add #real)
      (real_accum ra rb row start (finish - 1))
      (G.matmul (ematrix_subtile ra 16 16 row (finish - 1))
        (ematrix_subtile rb 16 8 (finish - 1) 0)) r c;
    segment_step (16 * start) (16 * (finish - 1)) (16 * finish)
      (dot_terms ra rb (16 * row + r) c)
  )

inline_for_extraction noextract
let block_approx
  (#nr:pos{nr % 16 == 0}) (#nc:pos{nc % 16 == 0})
  (va:chest2 bf16 nr nc) (vb:chest2 bf16 nc 8)
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  (r:nat{r < nr})
  : Lemma
      (requires va %~ ra /\ vb %~ rb)
      (ensures
        v_approximates (acc2 (block_result va vb start finish) r 0)
          (segment_sum (16 * start) (16 * finish) (dot_terms ra rb r 0))) =
  mma_accum_approx va vb ra rb (r / 16) start finish;
  real_accum_dot ra rb (r / 16) start finish (r % 16) 0;
  let actual = mma_accum va vb (r / 16) start finish in
  let logical = real_accum ra rb (r / 16) start finish in
  assert (chest_approximates actual logical);
  let _ : squash (
      acc actual (idx2 #16 #8 (r % 16) 0) %~ acc logical (idx2 #16 #8 (r % 16) 0)) =
    FStar.Classical.Sugar.forall_elim (idx2 #16 #8 (r % 16) 0) ()
  in ()

inline_for_extraction noextract
let rec zero_suffix_sum (n:nat) (count:nat{count <= n}) (terms:S.vector n)
  : Lemma
      (requires forall (i:nat{count <= i /\ i < n}). terms i == 0.0R)
      (ensures S.sum n terms == bounded_sum count terms)
      (decreases (n - count)) =
  if n = count then bounded_sum_full terms
  else (
    let lower:S.vector (n - 1) = fun i -> terms i in
    zero_suffix_sum (n - 1) count lower;
    sum_step n terms;
    norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
      (bounded_sum count lower);
    norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
      (bounded_sum count terms);
    sum_ext count (fun i -> lower i) (fun i -> terms i)
  )

inline_for_extraction noextract
let full_linear_bridge
  (rows cols:pos) (w:S.matrix rows cols) (x:S.vector cols) (r:nat{r < rows})
  : Lemma (
      segment_sum 0 (pad16 cols) (dot_terms (padded_matrix rows cols w) (padded_input cols x) r 0) ==
      S.linear rows cols w x r) =
  pad16_bounds rows;
  pad16_bounds cols;
  let terms = dot_terms (padded_matrix rows cols w) (padded_input cols x) r 0 in
  zero_suffix_sum (pad16 cols) cols terms;
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]]
    (segment_sum 0 (pad16 cols) terms);
  sum_ext (pad16 cols)
    (fun (i:S.index (pad16 cols - 0)) -> terms (0 + i))
    (fun i -> terms (0 + i));
  sum_ext (pad16 cols) (fun i -> terms (0 + i)) terms;
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
    (bounded_sum cols terms);
  sum_ext cols (fun i -> terms (i <: nat)) (fun i -> w r i *. x i);
  norm_spec [delta_once ["KPlay.SharedSpec.linear"]] (S.linear rows cols w x r)

inline_for_extraction noextract
let segment_empty (#n:nat) (position:nat{position <= n}) (terms:S.vector n)
  : Lemma (segment_sum position position terms == 0.0R) =
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]] (segment_sum position position terms);
  sum_zero (fun i -> terms (position + i))

inline_for_extraction noextract
let flat_tensor_index
  (nr nc:pos) (values:chest2 f32 nr nc)
  (r:nat{r < nr}) (c:nat{c < nc})
  : Lemma (
      Seq.index (to_seq (row_major nr nc) values) (r * nc + c) == acc2 values r c) =
  let layout = row_major nr nc in
  let coordinate = idx2 #nr #nc r c in
  assert (layout.imap.f coordinate == r * nc + c);
  Kuiper.Injection.lem_pat layout.imap coordinate

inline_for_extraction noextract
fn store_parts
  (result:Pulse.Lib.Reference.ref f32) (split:bool) (a b:f32)
  (#before:erased f32)
  (#ar #br:erased real)
  preserves gpu
  requires result |-> before
  requires pure (
    v_approximates a (FStar.Ghost.reveal #real ar) /\
    v_approximates b (FStar.Ghost.reveal #real br) /\
    (not split ==> FStar.Ghost.reveal #real ar == 0.0R))
  ensures exists* (value:f32). result |-> value **
    pure (v_approximates value (FStar.Ghost.reveal #real ar +. FStar.Ghost.reveal #real br))
{
  if split {
    let first = rounded_add #0.0R #ar (zero #f32) a;
    let combined = rounded_add #ar #br first b;
    result := combined;
  } else {
    let combined = rounded_add #0.0R #br (zero #f32) b;
    result := combined;
  }
}

inline_for_extraction noextract
fn combine_parts
  (split:bool) (a b:f32)
  (#ar #br:erased real)
  preserves gpu
  requires pure (
    v_approximates a (FStar.Ghost.reveal #real ar) /\
    v_approximates b (FStar.Ghost.reveal #real br) /\
    (not split ==> FStar.Ghost.reveal #real ar == 0.0R))
  returns combined:f32
  ensures pure (v_approximates combined
    (FStar.Ghost.reveal #real ar +. FStar.Ghost.reveal #real br))
{
  let mut result : f32 = zero;
  store_parts result split a b #_ #ar #br;
  !result
}

inline_for_extraction noextract
fn finish_matvec
  (rows:szp{rows <= 4096}) (cols:szp{cols <= 3584})
  (middle:sz{middle <= pad16 cols / 16})
  (left right:array2 f32 (row_major (pad16 rows) 8))
  (output:A.larray bf16 rows)
  (#va:chest2 bf16 (pad16 rows) (pad16 cols))
  (#vb:chest2 bf16 (pad16 cols) 8)
  (#w:erased (S.matrix rows cols)) (#x:erased (S.vector cols))
  preserves gpu
  preserves left |-> block_result va vb 0 middle
  preserves right |-> block_result va vb middle (pad16 cols / 16)
  requires A.pts_to_uninit output rows
  requires pure (va %~ padded_matrix rows cols w /\ vb %~ padded_input cols x)
  ensures exists* (result:lseq bf16 rows).
    output |-> result ** pure (bf_rep rows result (S.linear rows cols w x))
{
  pad16_bounds rows;
  pad16_bounds cols;
  tensor_concr left;
  tensor_concr right;
  let lp = core left;
  let rp = core right;
  assert rewrites_to lp (core left);
  assert rewrites_to rp (core right);
  unfold A.pts_to_uninit output rows;
  let mut i = 0sz;
  while (!i <^ rows)
    invariant exists* (j:sz{j <= rows}) (masked:Seq.seq (option bf16)).
      i |-> j ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == rows /\
        (forall (k:nat{k < j}).
          exists (v:bf16). Seq.index masked k == Some v /\
            v_approximates v (S.linear rows cols w x k)))
    decreases (rows - !i)
  {
    let j = !i;
    let a = lp.(8sz *^ j);
    let b = rp.(8sz *^ j);
    flat_tensor_index (pad16 rows) 8 (block_result va vb 0 middle) j 0;
    flat_tensor_index (pad16 rows) 8 (block_result va vb middle (pad16 cols / 16)) j 0;
    block_approx va vb (padded_matrix rows cols w) (padded_input cols x) 0 middle j;
    block_approx va vb (padded_matrix rows cols w) (padded_input cols x) middle (pad16 cols / 16) j;
    segment_empty 0 (dot_terms (padded_matrix rows cols w) (padded_input cols x) j 0);
    let combined = combine_parts (middle <> 0sz) a b
      #(segment_sum 0 (16 * middle) (dot_terms (padded_matrix rows cols w) (padded_input cols x) j 0))
      #(segment_sum (16 * middle) (pad16 cols) (dot_terms (padded_matrix rows cols w) (padded_input cols x) j 0));
    segment_step 0 (16 * middle) (pad16 cols)
      (dot_terms (padded_matrix rows cols w) (padded_input cols x) j 0);
    full_linear_bridge rows cols w x j;
    C.fcast_approx #f32 #bf16 combined (S.linear rows cols w x j);
    A.mask_write output j (C.fcast #f32 #bf16 combined);
    i := j +^ 1sz;
  };
  A.from_mask output;
  tensor_abs (row_major (pad16 rows) 8) lp #1.0R #(block_result va vb 0 middle);
  tensor_abs (row_major (pad16 rows) 8) rp #1.0R #(block_result va vb middle (pad16 cols / 16));
  lem_core_from_array left;
  lem_core_from_array right;
  rewrite each (from_array (row_major (pad16 rows) 8) (core left)) as left;
  rewrite each (from_array (row_major (pad16 rows) 8) (core right)) as right;
}

ghost
fn tensor_setup
  (#nr #nc:erased nat)
  (a:array2 bf16 (row_major nr nc))
  (b:array2 bf16 (col_major nc 8))
  (c:array2 f32 (row_major nr 8))
  (#va:chest2 bf16 nr nc) (#vb:chest2 bf16 nc 8) (#vc:chest2 f32 nr 8)
  ()
  norewrite
  requires a |-> va ** b |-> vb ** c |-> vc
  ensures (forall+ (_:natlt 32).
    a |-> Frac (1.0R /. 32.0R) va **
    b |-> Frac (1.0R /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) vc) ** emp
{
  tensor_share_n a 32;
  tensor_share_n b 32;
  tensor_share_n c 32;
  forevery_zip3
    (fun (_:natlt 32) -> a |-> Frac (1.0R /. 32.0R) va)
    (fun (_:natlt 32) -> b |-> Frac (1.0R /. 32.0R) vb)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) vc);
}

ghost
fn tensor_gather
  (#nr #nc:erased nat)
  (a:array2 bf16 (row_major nr nc))
  (b:array2 bf16 (col_major nc 8))
  (c:array2 f32 (row_major nr 8))
  (#va:chest2 bf16 nr nc) (#vb:chest2 bf16 nc 8) (#vc:chest2 f32 nr 8)
  ()
  norewrite
  requires (forall+ (_:natlt 32).
    a |-> Frac (1.0R /. 32.0R) va **
    b |-> Frac (1.0R /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) vc) ** emp
  ensures a |-> va ** b |-> vb ** c |-> vc
{
  forevery_unzip3
    (fun (_:natlt 32) -> a |-> Frac (1.0R /. 32.0R) va)
    (fun (_:natlt 32) -> b |-> Frac (1.0R /. 32.0R) vb)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) vc);
  tensor_gather_n a 32;
  tensor_gather_n b 32;
  tensor_gather_n c 32;
}

inline_for_extraction noextract
let tensor_desc
  (nr:szp{nr <= 4096 /\ nr % 16 == 0})
  (nc:szp{nc <= 3584 /\ nc % 16 == 0})
  (a:array2 bf16 (row_major nr nc){is_global a})
  (b:array2 bf16 (col_major nc 8){is_global b})
  (c:array2 f32 (row_major nr 8){is_global c})
  (start:sz) (finish:sz{start <= finish /\ finish <= nc / 16})
  (#va:chest2 bf16 nr nc) (#vb:chest2 bf16 nc 8) (#vc:chest2 f32 nr 8)
  : kernel_desc_1_n
      (a |-> va ** b |-> vb ** c |-> vc)
      (a |-> va ** b |-> vb ** c |-> block_result va vb start finish) = {
  nthr = 32sz;
  frame = emp;
  kpre = (fun _ ->
    a |-> Frac (1.0R /. 32.0R) va **
    b |-> Frac (1.0R /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) vc);
  kpost = (fun _ ->
    a |-> Frac (1.0R /. 32.0R) va **
    b |-> Frac (1.0R /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) (block_result va vb start finish));
  f = tensor_kernel nr nc a b c start finish #va #vb #vc;
  block_setup = tensor_setup a b c #va #vb #vc;
  block_teardown = tensor_gather a b c #va #vb #(block_result va vb start finish);
  full_pre_sendable = solve;
  full_post_sendable = solve;
  kpre_sendable = solve;
  kpost_sendable = solve;
}

fn run_tensor
  (nr:szp{nr <= 4096 /\ nr % 16 == 0})
  (nc:szp{nc <= 3584 /\ nc % 16 == 0})
  (a:array2 bf16 (row_major nr nc){is_global a})
  (b:array2 bf16 (col_major nc 8){is_global b})
  (c:array2 f32 (row_major nr 8){is_global c})
  (start:sz) (finish:sz{start <= finish /\ finish <= nc / 16})
  (#va:chest2 bf16 nr nc) (#vb:chest2 bf16 nc 8) (#vc:chest2 f32 nr 8)
  preserves cpu ** on gpu_loc (a |-> va ** b |-> vb)
  requires on gpu_loc (c |-> vc)
  ensures on gpu_loc (c |-> block_result va vb start finish)
{
  launch_kernel_full_sync (tensor_desc nr nc a b c start finish #va #vb #vc);
}

inline_for_extraction noextract
let row_from_seq_approx
  (nr nc:pos) (storage:lseq bf16 (nr * nc)) (logical:chest2 real nr nc)
  : Lemma
      (requires row_rep nr nc storage logical)
      (ensures from_seq (row_major nr nc) storage %~ logical) =
  EM.lemma_approximates_intro (from_seq (row_major nr nc) storage) logical

inline_for_extraction noextract
let col_from_seq_approx
  (nr:pos) (storage:lseq bf16 (nr * 8)) (logical:chest2 real nr 8)
  : Lemma
      (requires row_rep 8 nr storage (EM.mtranspose logical))
      (ensures from_seq (col_major nr 8) storage %~ logical) =
  EM.lemma_approximates_intro (from_seq (col_major nr 8) storage) logical

fn matvec
  (rows:szp{rows <= 4096}) (cols:szp{cols <= 3584}) (count:szp)
  (source:A.larray bf16 count{GA.is_global_array source})
  (offset rs:sz) (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (input:A.larray bf16 cols{GA.is_global_array input})
  (output:A.larray bf16 rows{GA.is_global_array output})
  (split:bool{split ==> cols % 32 == 0})
  (#fw #fx:perm)
  (#sw:erased (lseq bf16 count)) (#sx:erased (lseq bf16 cols))
  (#w:erased (S.matrix rows cols)) (#x:erased (S.vector cols))
  preserves cpu **
    on gpu_loc (source |-> Frac fw sw ** input |-> Frac fx sx)
  requires on gpu_loc (A.pts_to_uninit output rows)
  requires pure (matrix_rep rows cols count offset rs cs sw w /\ bf_rep cols sx x)
  ensures exists* (result:lseq bf16 rows).
    on gpu_loc (output |-> result) ** pure (bf_rep rows result (S.linear rows cols w x))
{
  pad16_bounds rows;
  pad16_bounds cols;
  let nr = ((rows +^ 15sz) /^ 16sz) *^ 16sz;
  let nc = ((cols +^ 15sz) /^ 16sz) *^ 16sz;
  assert pure (SZ.v nr == pad16 rows /\ SZ.v nc == pad16 cols);

  let raw_a = GA.gpu_array_alloc #bf16 (nr *^ nc);
  let ar : (p:A.larray bf16 (pad16 rows * pad16 cols){GA.is_global_array p}) = raw_a;
  assert rewrites_to ar raw_a;
  with a0. assert on gpu_loc (ar |-> a0);
  let initial_a : erased (lseq bf16 (pad16 rows * pad16 cols)) = hide (a0 <: lseq bf16 (pad16 rows * pad16 cols));
  launch_kernel_1 (fun () -> prepare_matrix rows cols count source offset rs cs ar #fw #sw #initial_a #w);
  with sa. assert on gpu_loc (ar |-> sa);
  row_from_seq_approx nr nc sa (padded_matrix rows cols w);
  let a = from_array (row_major nr nc) ar;
  assert rewrites_to a (from_array (row_major nr nc) ar);
  map_loc gpu_loc #(ar |-> sa) #(a |-> from_seq (row_major nr nc) sa)
    fn _ { tensor_abs' (row_major nr nc) ar; };

  let raw_b = GA.gpu_array_alloc #bf16 (nc *^ 8sz);
  let br : (p:A.larray bf16 (pad16 cols * 8){GA.is_global_array p}) = raw_b;
  assert rewrites_to br raw_b;
  with b0. assert on gpu_loc (br |-> b0);
  let initial_b : erased (lseq bf16 (pad16 cols * 8)) = hide (b0 <: lseq bf16 (pad16 cols * 8));
  launch_kernel_1 (fun () -> prepare_input cols input br #fx #sx #initial_b #x);
  with sb. assert on gpu_loc (br |-> sb);
  col_from_seq_approx nc sb (padded_input cols x);
  let b = from_array (col_major nc 8) br;
  assert rewrites_to b (from_array (col_major nc 8) br);
  map_loc gpu_loc #(br |-> sb) #(b |-> from_seq (col_major nc 8) sb)
    fn _ { tensor_abs' (col_major nc 8) br; };

  let left = Kuiper.Tensor.alloc0 #f32 (nr *^ 8sz) (row_major nr 8);
  let right = Kuiper.Tensor.alloc0 #f32 (nr *^ 8sz) (row_major nr 8);
  let middle : sz = (if split then nc /^ 32sz else 0sz);
  with la. assert on gpu_loc (left |-> la);
  run_tensor nr nc a b left 0sz middle #_ #_ #la;
  with rb. assert on gpu_loc (right |-> rb);
  run_tensor nr nc a b right middle (nc /^ 16sz) #_ #_ #rb;
  launch_kernel_1 (fun () -> finish_matvec rows cols middle left right output
    #(from_seq (row_major nr nc) sa) #(from_seq (col_major nc 8) sb) #w #x);
  Kuiper.Tensor.free left;
  Kuiper.Tensor.free right;
  Kuiper.Tensor.free a;
  Kuiper.Tensor.free b;
}

inline_for_extraction noextract
let rec sum_all_zero (n:nat) (f:S.vector n)
  : Lemma (requires forall (i:S.index n). f i == 0.0R)
          (ensures S.sum n f == 0.0R)
          (decreases n) =
  if n = 0 then sum_zero f
  else (
    sum_all_zero (n - 1) (fun i -> f i);
    sum_step n f
  )

inline_for_extraction noextract
let rec sum_swap (n m:nat) (f:S.matrix n m)
  : Lemma (
      S.sum n (fun i -> S.sum m (fun j -> f i j)) ==
      S.sum m (fun j -> S.sum n (fun i -> f i j)))
      (decreases n) =
  if n = 0 then (
    sum_zero (fun i -> S.sum m (fun j -> f i j));
    sum_all_zero m (fun j -> S.sum n (fun i -> f i j))
  ) else (
    sum_swap (n - 1) m (fun i j -> f i j);
    sum_step n (fun i -> S.sum m (fun j -> f i j));
    sum_plus m
      (fun j -> S.sum (n - 1) (fun i -> f i j))
      (fun j -> f (n - 1) j);
    let point (j:S.index m)
      : Lemma (
          S.sum n (fun i -> f i j) ==
          S.sum (n - 1) (fun i -> f i j) +. f (n - 1) j) =
      sum_step n (fun i -> f i j)
    in
    FStar.Classical.forall_intro #(S.index m) #(fun j ->
      S.sum n (fun i -> f i j) ==
      S.sum (n - 1) (fun i -> f i j) +. f (n - 1) j) point;
    sum_ext m
      (fun j -> S.sum n (fun i -> f i j))
      (fun j -> S.sum (n - 1) (fun i -> f i j) +. f (n - 1) j)
  )

inline_for_extraction noextract
let rec sum_grid (rows:nat) (cols:pos) (f:S.vector (rows * cols))
  : Lemma (
      S.sum rows (fun r -> S.sum cols (fun c -> f (r * cols + c))) ==
      S.sum (rows * cols) f)
      (decreases rows) =
  if rows = 0 then (
    sum_zero (fun r -> S.sum cols (fun c -> f (r * cols + c)));
    sum_zero f
  ) else (
    sum_grid (rows - 1) cols (fun i -> f i);
    sum_step rows (fun r -> S.sum cols (fun c -> f (r * cols + c)));
    sum_split ((rows - 1) * cols) cols f
  )

inline_for_extraction noextract
let slots (n width:pos) : pos = (n + width - 1) / width

inline_for_extraction noextract
let slots_bounds (n width:pos)
  : Lemma (n <= slots n width * width /\ slots n width * width < n + width) = ()

inline_for_extraction noextract
let padded_terms (n:pos) (extent:pos{n <= extent}) (terms:S.vector n) : S.vector extent =
  fun i -> if i < n then terms i else 0.0R

inline_for_extraction noextract
let stripe_terms
  (n width:pos) (group:nat{group < width}) (terms:S.vector n)
  : S.vector (slots n width) =
  fun q -> if q * width + group < n then terms (q * width + group) else 0.0R

inline_for_extraction noextract
let group_sums
  (n:pos) (width:pos{width <= 32}) (terms:S.vector n) : S.vector 256 =
  fun group ->
    if group < width then S.sum (slots n width) (stripe_terms n width group terms)
    else 0.0R

inline_for_extraction noextract
let stripes_sum
  (n:pos) (width:pos{width <= 32}) (terms:S.vector n)
  : Lemma (partial_sum width (group_sums n width terms) == S.sum n terms) =
  slots_bounds n width;
  let padded = padded_terms n (slots n width * width) terms in
  let stripe (group:S.index width)
    : Lemma (
        group_sums n width terms (group <: nat) ==
        S.sum (slots n width) (fun q -> padded (q * width + group))) =
    sum_ext (slots n width) (stripe_terms n width group terms)
      (fun q -> padded (q * width + group))
  in
  FStar.Classical.forall_intro #(S.index width) #(fun group ->
    group_sums n width terms (group <: nat) ==
    S.sum (slots n width) (fun q -> padded (q * width + group))) stripe;
  norm_spec [delta_once ["KPlay.Functional.partial_sum"]]
    (partial_sum width (group_sums n width terms));
  sum_ext width
    (fun group -> group_sums n width terms (group <: nat))
    (fun group -> S.sum (slots n width) (fun q -> padded (q * width + group)));
  sum_swap width (slots n width) (fun group q -> padded (q * width + group));
  sum_grid (slots n width) width padded;
  zero_suffix_sum (slots n width * width) n padded;
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
    (bounded_sum n padded);
  sum_ext n (fun i -> padded (i <: nat)) terms

inline_for_extraction noextract
let scaled_scores (n:nat) (x:S.vector n) : S.vector n =
  fun i -> x i *. 0.0625R

inline_for_extraction noextract
let exponentials (n:nat) (scores:S.vector n) (shift:real)
  : S.index n -> SQ.rpos =
  fun i -> E.exp_positive (scores i -. shift); E.exp (scores i -. shift)

inline_for_extraction noextract
let exponential_sum (n:pos) (scores:S.vector n) (shift:real) : SQ.rpos =
  S.sum_positive n (exponentials n scores shift)

inline_for_extraction noextract
let exponential_sum_bridge (n:pos) (scores:S.vector n) (shift:real)
  : Lemma (
      S.sum n (exponentials n scores shift) == exponential_sum n scores shift) =
  positive_sum_bridge n (exponentials n scores shift)

inline_for_extraction noextract
let exponential_shift (n:pos) (scores:S.vector n) (shift:real)
  : Lemma (
      exponential_sum n scores shift == exponential_sum n scores 0.0R /. E.exp shift) =
  E.exp_positive shift;
  let point (i:S.index n)
    : Lemma (
        exponentials n scores shift i ==
        (1.0R /. E.exp shift) *. exponentials n scores 0.0R i) =
    E.exp_sub shift (scores i)
  in
  FStar.Classical.forall_intro #(S.index n) #(fun i ->
    exponentials n scores shift i ==
    (1.0R /. E.exp shift) *. exponentials n scores 0.0R i) point;
  sum_ext n (exponentials n scores shift)
    (fun i -> (1.0R /. E.exp shift) *. exponentials n scores 0.0R i);
  sum_scale n (exponentials n scores 0.0R) (1.0R /. E.exp shift);
  exponential_sum_bridge n scores shift;
  exponential_sum_bridge n scores 0.0R;
  multiply_reciprocal (exponential_sum n scores 0.0R) (E.exp shift)

inline_for_extraction noextract
let softmax_denominator (n:pos) (scores:S.vector n) (i:S.index n)
  : Lemma (
      S.softmax n scores i == E.exp (scores i) /. exponential_sum n scores 0.0R) =
  let original : S.index n -> SQ.rpos =
    fun j -> E.exp_positive (scores j); E.exp (scores j)
  in
  positive_sum_bridge n original;
  sum_ext n original (exponentials n scores 0.0R);
  exponential_sum_bridge n scores 0.0R;
  norm_spec [delta_once ["KPlay.SharedSpec.softmax"]] (S.softmax n scores i)

inline_for_extraction noextract
let cancel_division (a:real) (b c:SQ.rpos)
  : Lemma ((a /. c) /. (b /. c) == a /. b) = ()

inline_for_extraction noextract
let softmax_shift_value (n:pos) (scores:S.vector n) (shift:real) (i:S.index n)
  : Lemma (
      E.exp (scores i -. shift) /. exponential_sum n scores shift ==
      S.softmax n scores i) =
  E.exp_positive shift;
  E.exp_sub shift (scores i);
  exponential_shift n scores shift;
  softmax_denominator n scores i;
  cancel_division (E.exp (scores i)) (exponential_sum n scores 0.0R) (E.exp shift)

inline_for_extraction noextract
let fp_rep (n:nat) (storage:lseq f32 n) (logical:S.vector n) : prop =
  forall (i:nat{i < n}). v_approximates (Seq.index storage i) (logical i)

inline_for_extraction noextract
fn rounded_sub (x y:f32) (#xr #yr:erased real)
  preserves gpu
  requires pure (
    v_approximates x (FStar.Ghost.reveal #real xr) /\
    v_approximates y (FStar.Ghost.reveal #real yr))
  returns result:f32
  ensures pure (v_approximates result
    (FStar.Ghost.reveal #real xr -. FStar.Ghost.reveal #real yr))
{
  let result = Fast.sub_rn x y;
  let first : squash (forall (r:real).
      v_approximates x (FStar.Ghost.reveal #real xr) /\ v_approximates y r ==>
      v_approximates result (FStar.Ghost.reveal #real xr -. r)) =
    FStar.Classical.Sugar.forall_elim (FStar.Ghost.reveal #real xr) ();
  let second : squash (
      v_approximates x (FStar.Ghost.reveal #real xr) /\
      v_approximates y (FStar.Ghost.reveal #real yr) ==>
      v_approximates result (FStar.Ghost.reveal #real xr -. FStar.Ghost.reveal #real yr)) =
    FStar.Classical.Sugar.forall_elim (FStar.Ghost.reveal #real yr) first;
  result
}

inline_for_extraction noextract
fn maximum_step (maximum:Pulse.Lib.Reference.ref f32) (value:f32) (#before:erased f32)
  preserves gpu
  requires maximum |-> before
  ensures maximum |-> (if lt (FStar.Ghost.reveal #f32 before) value
    then value else FStar.Ghost.reveal #f32 before)
{
  let old = !maximum;
  if (lt old value) { maximum := value; } else { () }
}

inline_for_extraction noextract
fn scaled_score (value:bf16) (#xr:erased real)
  preserves gpu
  requires pure (v_approximates value (FStar.Ghost.reveal #real xr))
  returns result:f32
  ensures pure (v_approximates result (FStar.Ghost.reveal #real xr *. 0.0625R))
{
  C.fcast_approx #bf16 #f32 value (FStar.Ghost.reveal #real xr);
  let promoted = C.fcast #bf16 #f32 value;
  let sixteen : f32 = of_int 16L;
  of_int_approx #f32 16L;
  let coefficient = div (one #f32) sixteen;
  div_approx (one #f32) sixteen 1.0R 16.0R;
  let scaled = rounded_mul #xr #0.0625R promoted coefficient;
  C.fcast_approx #f32 #bf16 scaled (FStar.Ghost.reveal #real xr *. 0.0625R);
  let stored = C.fcast #f32 #bf16 scaled;
  C.fcast_approx #bf16 #f32 stored (FStar.Ghost.reveal #real xr *. 0.0625R);
  let promoted_again = C.fcast #bf16 #f32 stored;
  rounded_add #(FStar.Ghost.reveal #real xr *. 0.0625R) #0.0R promoted_again (zero #f32)
}

inline_for_extraction noextract
fn scale_and_max
  (n:szp{n <= 647}) (input:A.larray bf16 n) (values:A.larray f32 n)
  (#fraction:perm) (#sx:erased (lseq bf16 n)) (#before:erased (lseq f32 n))
  (#x:erased (S.vector n))
  preserves gpu ** input |-> Frac fraction sx
  requires values |-> before
  requires pure (bf_rep n sx x)
  returns maximum:f32
  ensures exists* (sv:lseq f32 n).
    values |-> sv ** pure (fp_rep n sv (scaled_scores n x))
{
  let negative_infinity : f32 = sub (zero #f32) (infinity #f32);
  let mut maximum : f32 = negative_infinity;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (m:f32) (sv:lseq f32 n).
      i |-> j ** maximum |-> m ** values |-> sv **
      pure (forall (k:nat{k < j}).
        v_approximates (Seq.index sv k) (scaled_scores n x (k <: nat)))
    decreases (n - !i)
  {
    let j = !i;
    let raw = input.(j);
    let value = scaled_score raw #(vector_at #n x j);
    values.(j) <- value;
    maximum_step maximum value;
    i := j +^ 1sz;
  };
  !maximum
}

inline_for_extraction noextract
fn exponentiate
  (n:szp{n <= 647}) (values:A.larray f32 n) (maximum:f32)
  (#before:erased (lseq f32 n))
  (#scores:erased (S.vector n)) (#shift:erased real)
  preserves gpu
  requires values |-> before
  requires pure (fp_rep n before scores /\
    v_approximates maximum (FStar.Ghost.reveal #real shift))
  ensures exists* (after:lseq f32 n). values |-> after **
    pure (fp_rep n after (exponentials n scores (FStar.Ghost.reveal #real shift)))
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (sv:lseq f32 n).
      i |-> j ** values |-> sv **
      pure (forall (k:nat{k < n}).
        if k < j then
          v_approximates (Seq.index sv k)
            (E.exp (vector_at #n scores k -. FStar.Ghost.reveal #real shift))
        else v_approximates (Seq.index sv k) (vector_at #n scores k))
    decreases (n - !i)
  {
    let j = !i;
    let value = values.(j);
    let difference = rounded_sub value maximum #(vector_at #n scores j) #shift;
    let ex = fexp difference;
    exp_approx difference (vector_at #n scores j -. FStar.Ghost.reveal #real shift);
    values.(j) <- ex;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn softmax_width (n:szp{n <= 647})
  preserves gpu
  returns width:szp
  ensures pure (tree_width width /\ width <= 32)
{
  let mut width : szp = 1sz;
  while (!width <^ n && !width <^ 32sz)
    invariant exists* (w:szp{tree_width w /\ w <= 32}). width |-> w
    decreases (32 - !width)
  {
    width := !width *^ 2sz;
  };
  !width
}

inline_for_extraction noextract
let stripe_tail_zero
  (n width:pos) (group:nat{group < width})
  (q:nat{q <= slots n width}) (terms:S.vector n)
  : Lemma
      (requires q == slots n width \/ n <= q * width + group)
      (ensures forall (i:nat{q <= i /\ i < slots n width}).
        stripe_terms n width group terms (i <: nat) == 0.0R) = ()

inline_for_extraction noextract
fn stripe_sum
  (n:szp{n <= 647}) (width:szp{width <= 32}) (group:sz{group < width})
  (values:A.larray f32 n)
  (#sv:erased (lseq f32 n)) (#terms:erased (S.vector n))
  preserves gpu ** values |-> sv
  requires pure (fp_rep n sv terms)
  returns result:f32
  ensures pure (v_approximates result (S.sum (slots n width) (stripe_terms n width group terms)))
{
  slots_bounds n width;
  let count = (n +^ width -^ 1sz) /^ width;
  assert pure (SZ.v count == slots n width);
  let stripe : erased (S.vector (slots n width)) = hide (stripe_terms n width group terms);
  let mut q = 0sz;
  let mut acc : f32 = zero;
  while (!q <^ count && group +^ !q *^ width <^ n)
    invariant exists* (k:sz{k <= count}) (current:f32).
      q |-> k ** acc |-> current **
      pure (v_approximates current (bounded_sum #(slots n width) k stripe))
    decreases (count - !q)
  {
    let k = !q;
    let index = group +^ k *^ width;
    let value = values.(index);
    let previous = !acc;
    let next = rounded_add
      #(bounded_sum #(slots n width) k stripe)
      #(vector_at #(slots n width) stripe k) previous value;
    bounded_sum_step #(slots n width) k stripe;
    acc := next;
    q := k +^ 1sz;
  };
  let k = !q;
  stripe_tail_zero n width group k terms;
  zero_suffix_sum (slots n width) k stripe;
  !acc
}

inline_for_extraction noextract
fn reduce_exponentials
  (n:szp{n <= 647}) (width:szp{tree_width width /\ width <= 32})
  (values:A.larray f32 n) (partial:A.larray f32 256)
  (#sv:erased (lseq f32 n)) (#sp:erased (lseq f32 256))
  (#terms:erased (S.vector n))
  preserves gpu ** values |-> sv
  requires partial |-> sp
  requires pure (fp_rep n sv terms)
  returns denominator:f32
  ensures exists* (next:lseq f32 256). partial |-> next **
    pure (v_approximates denominator (S.sum n terms))
{
  let mut i = 0sz;
  while (!i <^ width)
    invariant exists* (j:sz{j <= width}) (current:lseq f32 256).
      i |-> j ** partial |-> current **
      pure (partial_rep j current (group_sums n width terms))
    decreases (width - !i)
  {
    let j = !i;
    let value = stripe_sum n width j values #sv #terms;
    partial.(j) <- value;
    i := j +^ 1sz;
  };
  let result = tree_reduce partial width #_ #(group_sums n width terms);
  stripes_sum n width terms;
  result
}

inline_for_extraction noextract
fn softmax_finish
  (n:szp{n <= 647}) (values:A.larray f32 n) (denominator:f32)
  (output:A.larray bf16 n)
  (#sv:erased (lseq f32 n))
  (#scores:erased (S.vector n)) (#shift:erased real)
  preserves gpu ** values |-> sv
  requires A.pts_to_uninit output n
  requires pure (
    fp_rep n sv (exponentials n scores (FStar.Ghost.reveal #real shift)) /\
    v_approximates denominator (exponential_sum n scores (FStar.Ghost.reveal #real shift)))
  ensures exists* (result:lseq bf16 n).
    output |-> result ** pure (bf_rep n result (S.softmax n scores))
{
  unfold A.pts_to_uninit output n;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (masked:Seq.seq (option bf16)).
      i |-> j ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == n /\
        (forall (k:nat{k < j}).
          exists (v:bf16). Seq.index masked k == Some v /\
            v_approximates v (S.softmax n scores k)))
    decreases (n - !i)
  {
    let j = !i;
    let value = values.(j);
    let quotient = div value denominator;
    div_approx value denominator
      (E.exp (vector_at #n scores j -. FStar.Ghost.reveal #real shift))
      (exponential_sum n scores (FStar.Ghost.reveal #real shift));
    softmax_shift_value n scores (FStar.Ghost.reveal #real shift) j;
    C.fcast_approx #f32 #bf16 quotient (S.softmax n scores j);
    A.mask_write output j (C.fcast #f32 #bf16 quotient);
    i := j +^ 1sz;
  };
  A.from_mask output;
}

inline_for_extraction noextract
fn softmax_kernel
  (n:szp{n <= 647}) (input:A.larray bf16 n) (output:A.larray bf16 n)
  (values:A.larray f32 n) (partial:A.larray f32 256)
  (#fraction:perm) (#sx:erased (lseq bf16 n))
  (#sv:erased (lseq f32 n)) (#sp:erased (lseq f32 256))
  (#x:erased (S.vector n))
  preserves gpu ** input |-> Frac fraction sx
  requires values |-> sv ** partial |-> sp ** A.pts_to_uninit output n
  requires pure (bf_rep n sx x)
  ensures exists* (result:lseq bf16 n) (vnext:lseq f32 n) (pnext:lseq f32 256).
    output |-> result ** values |-> vnext ** partial |-> pnext **
    pure (bf_rep n result (S.softmax n (scaled_scores n x)))
{
  let width = softmax_width n;
  let maximum = scale_and_max n input values #fraction #sx #sv #x;
  (* This is the actual reduction result; arbitrary-shift cancellation needs
     neither native finiteness nor real/native comparison preservation. *)
  to_real_ok maximum;
  let shift : erased real = hide (to_real maximum);
  exponentiate n values maximum #_ #(scaled_scores n x) #shift;
  let denominator = reduce_exponentials n width values partial #_ #sp
    #(exponentials n (scaled_scores n x) (FStar.Ghost.reveal #real shift));
  exponential_sum_bridge n (scaled_scores n x) (FStar.Ghost.reveal #real shift);
  softmax_finish n values denominator output #_ #(scaled_scores n x) #shift;
}

fn softmax
  (n:szp{n <= 647})
  (input:A.larray bf16 n{GA.is_global_array input})
  (output:A.larray bf16 n{GA.is_global_array output})
  (#fraction:perm) (#sx:erased (lseq bf16 n)) (#x:erased (S.vector n))
  preserves cpu ** on gpu_loc (input |-> Frac fraction sx)
  requires on gpu_loc (A.pts_to_uninit output n)
  requires pure (bf_rep n sx x)
  ensures exists* (result:lseq bf16 n).
    on gpu_loc (output |-> result) ** pure (bf_rep n result (S.softmax n (scaled_scores n x)))
{
  let va = GA.gpu_array_alloc #f32 n;
  let values : (p:A.larray f32 n{GA.is_global_array p}) = va;
  assert rewrites_to values va;
  with sv. assert on gpu_loc (values |-> sv);
  let initial_values : erased (lseq f32 n) = hide (sv <: lseq f32 n);
  let pa = GA.gpu_array_alloc #f32 256sz;
  let partial : (p:A.larray f32 256{GA.is_global_array p}) = pa;
  assert rewrites_to partial pa;
  with sp. assert on gpu_loc (partial |-> sp);
  let initial_partial : erased (lseq f32 256) = hide (sp <: lseq f32 256);
  launch_kernel_1 (fun () -> softmax_kernel n input output values partial
    #fraction #sx #initial_values #initial_partial #x);
  GA.gpu_array_free values;
  GA.gpu_array_free partial;
}

inline_for_extraction noextract
let chunk_work (n groups chunk:pos) : pos = slots n (groups * chunk) * chunk

inline_for_extraction noextract
let chunk_index (groups chunk:pos) (group:nat{group < groups}) (q:nat) : nat =
  (q / chunk) * (groups * chunk) + group * chunk + q % chunk

inline_for_extraction noextract
let chunk_terms (n groups chunk:pos) (group:nat{group < groups}) (terms:S.vector n)
  : S.vector (chunk_work n groups chunk) =
  fun q ->
    let i = chunk_index groups chunk group q in
    if i < n then terms i else 0.0R

inline_for_extraction noextract
let chunk_group_sums
  (n:pos) (groups:pos{groups <= 32}) (chunk:pos) (terms:S.vector n)
  : S.vector 256 =
  fun group ->
    if group < groups then S.sum (chunk_work n groups chunk) (chunk_terms n groups chunk group terms)
    else 0.0R

inline_for_extraction noextract
let chunk_coordinate
  (blocks groups chunk:pos) (group:nat{group < groups})
  (q:nat{q < blocks}) (o:nat{o < chunk})
  : Lemma (
      chunk_index groups chunk group (q * chunk + o) ==
      q * (groups * chunk) + group * chunk + o /\
      chunk_index groups chunk group (q * chunk + o) < blocks * (groups * chunk)) =
  row_index blocks chunk q o

inline_for_extraction noextract
let grid_index (rows cols:pos) (r:nat{r < rows}) (c:nat{c < cols})
  : i:nat{i < rows * cols /\ i == r * cols + c} =
  row_index rows cols r c;
  r * cols + c

inline_for_extraction noextract
let block_values (blocks width:pos) (padded:S.vector (blocks * width))
  (q:nat{q < blocks}) : S.vector width =
  fun i -> padded (q * width + i)

inline_for_extraction noextract
let block_values_sum (blocks width:pos) (padded:S.vector (blocks * width))
  (q:nat{q < blocks})
  : Lemma (
      S.sum width (block_values blocks width padded q) ==
      S.sum width (fun i -> padded (q * width + i))) =
  sum_ext width (block_values blocks width padded q) (fun i -> padded (q * width + i))

inline_for_extraction noextract
let block_rowsum (blocks groups chunk:pos)
  (padded:S.vector (blocks * (groups * chunk)))
  (q:nat{q < blocks}) (g:nat{g < groups}) : real =
  S.sum chunk (fun o ->
    padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o)))

inline_for_extraction noextract
let block_rowsums_total (blocks groups chunk:pos)
  (padded:S.vector (blocks * (groups * chunk))) (q:nat{q < blocks})
  : Lemma (
      S.sum groups (fun g -> block_rowsum blocks groups chunk padded q g) ==
      S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q)) =
  sum_grid groups chunk (block_values blocks (groups * chunk) padded q);
  let same (g:S.index groups)
    : Lemma (
        block_rowsum blocks groups chunk padded q g ==
        S.sum chunk (fun o -> block_values blocks (groups * chunk) padded q (g * chunk + o))) =
    sum_ext chunk
      (fun o -> padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o)))
      (fun o -> block_values blocks (groups * chunk) padded q (g * chunk + o))
  in
  FStar.Classical.forall_intro #(S.index groups) #(fun g ->
    block_rowsum blocks groups chunk padded q g ==
    S.sum chunk (fun o -> block_values blocks (groups * chunk) padded q (g * chunk + o))) same;
  sum_ext groups
    (fun g -> block_rowsum blocks groups chunk padded q g)
    (fun g -> S.sum chunk (fun o -> block_values blocks (groups * chunk) padded q (g * chunk + o)))

inline_for_extraction noextract
let chunk_groups_sum
  (n:pos) (groups:pos{groups <= 32}) (chunk:pos) (terms:S.vector n)
  : Lemma (partial_sum groups (chunk_group_sums n groups chunk terms) == S.sum n terms) =
  slots_bounds n (groups * chunk);
  let blocks = slots n (groups * chunk) in
  let padded = padded_terms n (blocks * (groups * chunk)) terms in
  let rowsums (q:S.index blocks) (g:S.index groups) : real =
    block_rowsum blocks groups chunk padded q g
  in
  let one_group (g:S.index groups)
    : Lemma (
        chunk_group_sums n groups chunk terms (g <: nat) ==
        S.sum blocks (fun q -> rowsums q g)) =
    let expanded : S.vector (blocks * chunk) = chunk_terms n groups chunk g terms in
    sum_grid blocks chunk expanded;
    let one_block (q:S.index blocks)
      : Lemma (
          S.sum chunk (fun o -> expanded (grid_index blocks chunk q o)) == rowsums q g) =
      let point (o:S.index chunk)
        : Lemma (
            expanded (grid_index blocks chunk q o) ==
            padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o))) =
        chunk_coordinate blocks groups chunk g q o
      in
      FStar.Classical.forall_intro #(S.index chunk) #(fun o ->
        expanded (grid_index blocks chunk q o) ==
        padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o))) point;
      sum_ext chunk (fun o -> expanded (grid_index blocks chunk q o))
        (fun o -> padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o)))
    in
    FStar.Classical.forall_intro #(S.index blocks) #(fun q ->
      S.sum chunk (fun o -> expanded (grid_index blocks chunk q o)) == rowsums q g) one_block;
    let grid_block (q:S.index blocks)
      : Lemma (
          S.sum chunk (fun o -> expanded (q * chunk + o)) ==
          S.sum chunk (fun o -> expanded (grid_index blocks chunk q o))) =
      sum_ext chunk (fun o -> expanded (q * chunk + o))
        (fun o -> expanded (grid_index blocks chunk q o))
    in
    FStar.Classical.forall_intro #(S.index blocks) #(fun q ->
      S.sum chunk (fun o -> expanded (q * chunk + o)) ==
      S.sum chunk (fun o -> expanded (grid_index blocks chunk q o))) grid_block;
    sum_ext blocks
      (fun q -> S.sum chunk (fun o -> expanded (q * chunk + o)))
      (fun q -> rowsums q g)
  in
  FStar.Classical.forall_intro #(S.index groups) #(fun g ->
    chunk_group_sums n groups chunk terms (g <: nat) ==
    S.sum blocks (fun q -> rowsums q g)) one_group;
  norm_spec [delta_once ["KPlay.Functional.partial_sum"]]
    (partial_sum groups (chunk_group_sums n groups chunk terms));
  sum_ext groups
    (fun g -> chunk_group_sums n groups chunk terms (g <: nat))
    (fun g -> S.sum blocks (fun q -> rowsums q g));
  sum_swap groups blocks (fun g q -> rowsums q g);
  let block_sum (q:S.index blocks)
    : Lemma (
        S.sum groups (fun g -> rowsums q g) ==
        S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q)) =
    block_rowsums_total blocks groups chunk padded q;
    sum_ext groups (fun g -> rowsums q g) (fun g -> block_rowsum blocks groups chunk padded q g)
  in
  FStar.Classical.forall_intro #(S.index blocks) #(fun q ->
    S.sum groups (fun g -> rowsums q g) ==
    S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q)) block_sum;
  sum_ext blocks
    (fun q -> S.sum groups (fun g -> rowsums q g))
    (fun q -> S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q));
  sum_grid blocks (groups * chunk) padded;
  let flat_block (q:S.index blocks)
    : Lemma (
        S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q) ==
        S.sum (groups * chunk) (fun i -> padded (q * (groups * chunk) + i))) =
    block_values_sum blocks (groups * chunk) padded q
  in
  FStar.Classical.forall_intro #(S.index blocks) #(fun q ->
    S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q) ==
    S.sum (groups * chunk) (fun i -> padded (q * (groups * chunk) + i))) flat_block;
  sum_ext blocks
    (fun q -> S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q))
    (fun q -> S.sum (groups * chunk) (fun i -> padded (q * (groups * chunk) + i)));
  zero_suffix_sum (blocks * (groups * chunk)) n padded;
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]] (bounded_sum n padded);
  sum_ext n (fun i -> padded (i <: nat)) terms

inline_for_extraction noextract
let linear_terms (rows cols:pos) (w:S.matrix rows cols) (x:S.vector cols)
  (row:nat{row < rows}) : S.vector cols =
  fun c -> w row c *. x c

inline_for_extraction noextract
let matrix_at (#rows #cols:nat) (w:erased (S.matrix rows cols))
  (r:nat{r < rows}) (c:nat{c < cols}) : GTot real =
  (FStar.Ghost.reveal #(S.matrix rows cols) w) r c

inline_for_extraction noextract
let chunk_runtime_bounds
  (n:pos{n <= 647}) (groups:pos{groups = 16 \/ groups = 32})
  (chunk:pos{chunk <= 64}) (group:nat{group < groups})
  (q:nat{q <= chunk_work n groups chunk})
  : Lemma (
      groups * chunk <= 2048 /\
      chunk_work n groups chunk <= 2695 /\
      q / chunk * (groups * chunk) < 10000 /\
      q / chunk * (groups * chunk) + group * chunk < 10000 /\
      chunk_index groups chunk group q < 10000) =
  slots_bounds n (groups * chunk)

inline_for_extraction noextract
let partial_bounded (n:nat{n <= 256}) (x:S.vector 256)
  : Lemma (partial_sum n x == bounded_sum #256 n x) = ()

inline_for_extraction noextract
fn rounded_fma (x y z:f32) (#xr #yr #zr:erased real)
  preserves gpu
  requires pure (
    v_approximates x (FStar.Ghost.reveal #real xr) /\
    v_approximates y (FStar.Ghost.reveal #real yr) /\
    v_approximates z (FStar.Ghost.reveal #real zr))
  returns result:f32
  ensures pure (v_approximates result
    (FStar.Ghost.reveal #real xr *. FStar.Ghost.reveal #real yr +. FStar.Ghost.reveal #real zr))
{
  let result = Fast.fma_rn x y z;
  instantiate_fma x y z result
    (FStar.Ghost.reveal #real xr) (FStar.Ghost.reveal #real yr) (FStar.Ghost.reveal #real zr);
  result
}

inline_for_extraction noextract
fn simt_step
  (rows:szp{rows <= 647}) (cols:szp{cols <= 647}) (count:szp)
  (source:A.larray bf16 count) (offset rs:sz)
  (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (input:A.larray bf16 cols) (row:sz{row < rows})
  (groups:szp{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:szp{chunk <= 64}) (group:sz{group < groups})
  (q:sz{q < chunk_work cols groups chunk})
  (acc:Pulse.Lib.Reference.ref f32)
  (#fw #fx:perm) (#sw:erased (lseq bf16 count)) (#sx:erased (lseq bf16 cols))
  (#w:erased (S.matrix rows cols)) (#x:erased (S.vector cols)) (#before:erased f32)
  preserves gpu ** source |-> Frac fw sw ** input |-> Frac fx sx
  requires acc |-> before
  requires pure (
    matrix_rep rows cols count offset rs cs sw w /\ bf_rep cols sx x /\
    v_approximates (FStar.Ghost.reveal #f32 before) (bounded_sum q
      (chunk_terms cols groups chunk group (linear_terms rows cols w x row))))
  ensures exists* (value:f32). acc |-> value **
    pure (v_approximates value (bounded_sum (q + 1)
      (chunk_terms cols groups chunk group (linear_terms rows cols w x row))))
{
  chunk_runtime_bounds cols groups chunk group q;
  let index = (q /^ chunk) *^ (groups *^ chunk) +^ group *^ chunk +^ (q %^ chunk);
  assert pure (SZ.v index == chunk_index groups chunk group q);
  bounded_sum_step q (chunk_terms cols groups chunk group (linear_terms rows cols w x row));
  if (index <^ cols) {
    matrix_index_bounds rows cols count offset rs cs row index;
    let av = source.(offset +^ row *^ rs +^ index *^ cs);
    let bv = input.(index);
    C.fcast_approx #bf16 #f32 av (matrix_at #rows #cols w row index);
    C.fcast_approx #bf16 #f32 bv (vector_at #cols x index);
    let old = !acc;
    let value = rounded_fma (C.fcast #bf16 #f32 av) (C.fcast #bf16 #f32 bv) old
      #(matrix_at #rows #cols w row index) #(vector_at #cols x index)
      #(bounded_sum q (chunk_terms cols groups chunk group (linear_terms rows cols w x row)));
    acc := value;
  } else {
    ()
  }
}

inline_for_extraction noextract
fn simt_group
  (rows:szp{rows <= 647}) (cols:szp{cols <= 647}) (count:szp)
  (source:A.larray bf16 count) (offset rs:sz)
  (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (input:A.larray bf16 cols) (row:sz{row < rows})
  (groups:szp{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:szp{chunk <= 64}) (group:sz{group < groups})
  (#fw #fx:perm) (#sw:erased (lseq bf16 count)) (#sx:erased (lseq bf16 cols))
  (#w:erased (S.matrix rows cols)) (#x:erased (S.vector cols))
  preserves gpu ** source |-> Frac fw sw ** input |-> Frac fx sx
  requires pure (matrix_rep rows cols count offset rs cs sw w /\ bf_rep cols sx x)
  returns result:f32
  ensures pure (v_approximates result
    (S.sum (chunk_work cols groups chunk)
      (chunk_terms cols groups chunk group (linear_terms rows cols w x row))))
{
  chunk_runtime_bounds cols groups chunk group 0;
  let span = groups *^ chunk;
  let work = ((cols +^ span -^ 1sz) /^ span) *^ chunk;
  assert pure (SZ.v work == chunk_work cols groups chunk);
  let mut q = 0sz;
  let mut acc : f32 = zero;
  while (!q <^ work)
    invariant exists* (k:sz{k <= work}) (value:f32).
      q |-> k ** acc |-> value **
      pure (v_approximates value (bounded_sum k
        (chunk_terms cols groups chunk group (linear_terms rows cols w x row))))
    decreases (work - !q)
  {
    let k = !q;
    simt_step rows cols count source offset rs cs input row groups chunk group k acc
      #fw #fx #sw #sx #w #x;
    q := k +^ 1sz;
  };
  bounded_sum_full (chunk_terms cols groups chunk group (linear_terms rows cols w x row));
  !acc
}

inline_for_extraction noextract
fn linear_reduce
  (partial:A.larray f32 256) (groups:szp{groups <= 32})
  (#sp:erased (lseq f32 256)) (#logical:erased (S.vector 256))
  preserves gpu ** partial |-> sp
  requires pure (partial_rep groups sp logical)
  returns result:f32
  ensures pure (v_approximates result (partial_sum groups logical))
{
  let first = partial.(0sz);
  bounded_sum_step #256 0 logical;
  let mut acc : f32 = first;
  let mut i = 1sz;
  while (!i <^ groups)
    invariant exists* (j:sz{1 <= j /\ j <= groups}) (value:f32).
      i |-> j ** acc |-> value **
      pure (v_approximates value (bounded_sum #256 j logical))
    decreases (groups - !i)
  {
    let j = !i;
    let value = partial.(j);
    let old = !acc;
    let combined = rounded_add #(bounded_sum #256 j logical)
      #(vector_at #256 logical j) old value;
    bounded_sum_step #256 j logical;
    acc := combined;
    i := j +^ 1sz;
  };
  partial_bounded groups logical;
  !acc
}

inline_for_extraction noextract
fn store_reduction
  (partial:A.larray f32 256) (groups:szp{tree_width groups /\ groups <= 32})
  (linear:bool) (out:Pulse.Lib.Reference.ref f32)
  (#sp:erased (lseq f32 256)) (#logical:erased (S.vector 256)) (#before:erased f32)
  preserves gpu
  requires partial |-> sp ** out |-> before
  requires pure (partial_rep groups sp logical)
  ensures exists* (next:lseq f32 256) (value:f32).
    partial |-> next ** out |-> value ** pure (v_approximates value (partial_sum groups logical))
{
  if linear {
    let value = linear_reduce partial groups #sp #logical;
    out := value;
  } else {
    let value = tree_reduce partial groups #sp #logical;
    out := value;
  }
}

inline_for_extraction noextract
fn simt_row
  (rows:szp{rows <= 647}) (cols:szp{cols <= 647}) (count:szp)
  (source:A.larray bf16 count) (offset rs:sz)
  (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (input:A.larray bf16 cols) (row:sz{row < rows})
  (groups:szp{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:szp{chunk <= 64}) (linear:bool)
  (partial:A.larray f32 256)
  (#fw #fx:perm) (#sw:erased (lseq bf16 count)) (#sx:erased (lseq bf16 cols))
  (#w:erased (S.matrix rows cols)) (#x:erased (S.vector cols))
  (#sp:erased (lseq f32 256))
  preserves gpu ** source |-> Frac fw sw ** input |-> Frac fx sx
  requires partial |-> sp
  requires pure (matrix_rep rows cols count offset rs cs sw w /\ bf_rep cols sx x)
  returns result:bf16
  ensures exists* (next:lseq f32 256). partial |-> next **
    pure (v_approximates result (S.linear rows cols w x row))
{
  let mut i = 0sz;
  while (!i <^ groups)
    invariant exists* (j:sz{j <= groups}) (values:lseq f32 256).
      i |-> j ** partial |-> values **
      pure (partial_rep j values
        (chunk_group_sums cols groups chunk (linear_terms rows cols w x row)))
    decreases (groups - !i)
  {
    let j = !i;
    let value = simt_group rows cols count source offset rs cs input row groups chunk j #fw #fx #sw #sx #w #x;
    partial.(j) <- value;
    i := j +^ 1sz;
  };
  let mut reduced : f32 = zero;
  store_reduction partial groups linear reduced #_
    #(chunk_group_sums cols groups chunk (linear_terms rows cols w x row));
  let value = !reduced;
  chunk_groups_sum cols groups chunk (linear_terms rows cols w x row);
  norm_spec [delta_once ["KPlay.SharedSpec.linear"]] (S.linear rows cols w x row);
  sum_ext cols (linear_terms rows cols w x row)
    (fun c -> matrix_at #rows #cols w row (c <: nat) *. vector_at #cols x (c <: nat));
  C.fcast_approx #f32 #bf16 value (S.linear rows cols w x row);
  C.fcast #f32 #bf16 value
}

inline_for_extraction noextract
fn simt_kernel
  (rows:szp{rows <= 647}) (cols:szp{cols <= 647}) (count:szp)
  (source:A.larray bf16 count) (offset rs:sz)
  (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (input:A.larray bf16 cols) (output:A.larray bf16 rows)
  (groups:szp{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:szp{chunk <= 64}) (linear:bool)
  (partial:A.larray f32 256)
  (#fw #fx:perm) (#sw:erased (lseq bf16 count)) (#sx:erased (lseq bf16 cols))
  (#w:erased (S.matrix rows cols)) (#x:erased (S.vector cols))
  (#sp:erased (lseq f32 256))
  preserves gpu ** source |-> Frac fw sw ** input |-> Frac fx sx
  requires partial |-> sp ** A.pts_to_uninit output rows
  requires pure (matrix_rep rows cols count offset rs cs sw w /\ bf_rep cols sx x)
  ensures exists* (result:lseq bf16 rows) (next:lseq f32 256).
    output |-> result ** partial |-> next **
    pure (bf_rep rows result (S.linear rows cols w x))
{
  unfold A.pts_to_uninit output rows;
  let mut i = 0sz;
  while (!i <^ rows)
    invariant exists* (j:sz{j <= rows}) (masked:Seq.seq (option bf16)) (temp:lseq f32 256).
      i |-> j ** partial |-> temp ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == rows /\
        (forall (k:nat{k < j}).
          exists (v:bf16). Seq.index masked k == Some v /\
            v_approximates v (S.linear rows cols w x k)))
    decreases (rows - !i)
  {
    let j = !i;
    let value = simt_row rows cols count source offset rs cs input j groups chunk linear partial #fw #fx #sw #sx #w #x;
    A.mask_write output j value;
    i := j +^ 1sz;
  };
  A.from_mask output;
}

fn simt
  (rows:szp{rows <= 647}) (cols:szp{cols <= 647}) (count:szp)
  (source:A.larray bf16 count{GA.is_global_array source}) (offset rs:sz)
  (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (input:A.larray bf16 cols{GA.is_global_array input})
  (output:A.larray bf16 rows{GA.is_global_array output})
  (groups:szp{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:szp{chunk <= 64}) (linear:bool)
  (#fw #fx:perm) (#sw:erased (lseq bf16 count)) (#sx:erased (lseq bf16 cols))
  (#w:erased (S.matrix rows cols)) (#x:erased (S.vector cols))
  preserves cpu ** on gpu_loc (source |-> Frac fw sw ** input |-> Frac fx sx)
  requires on gpu_loc (A.pts_to_uninit output rows)
  requires pure (matrix_rep rows cols count offset rs cs sw w /\ bf_rep cols sx x)
  ensures exists* (result:lseq bf16 rows).
    on gpu_loc (output |-> result) ** pure (bf_rep rows result (S.linear rows cols w x))
{
  let allocation = GA.gpu_array_alloc #f32 256sz;
  let partial : (p:A.larray f32 256{GA.is_global_array p}) = allocation;
  assert rewrites_to partial allocation;
  with sp. assert on gpu_loc (partial |-> sp);
  let initial : erased (lseq f32 256) = hide (sp <: lseq f32 256);
  launch_kernel_1 (fun () -> simt_kernel rows cols count source offset rs cs input output
    groups chunk linear partial #fw #fx #sw #sx #w #x #initial);
  GA.gpu_array_free partial;
}

inline_for_extraction noextract
fn allocate_bf (n:szp)
  preserves cpu
  returns result:(p:A.larray bf16 n{GA.is_global_array p /\ A.is_full_array p})
  ensures on gpu_loc (A.pts_to_uninit result n)
{
  let a = GA.gpu_array_alloc #bf16 n;
  with values. assert on gpu_loc (a |-> values);
  map_loc gpu_loc #(a |-> values) #(A.pts_to_uninit a n)
    fn _ {
      A.to_mask a;
      fold A.pts_to_uninit a n;
    };
  a
}

inline_for_extraction noextract
let gather_index
  (rows cols:pos) (count:nat)
  (offset rs:nat) (cs:nat{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (i:nat{i < rows * cols})
  : index:nat{index < count /\ index == offset + (i / cols) * rs + (i % cols) * cs} =
  matrix_index_bounds rows cols count offset rs cs (i / cols) (i % cols);
  offset + (i / cols) * rs + (i % cols) * cs

inline_for_extraction noextract
let gathered
  (rows cols:pos) (count:nat)
  (offset rs:nat) (cs:nat{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (source:lseq bf16 count) (result:lseq bf16 (rows * cols)) : prop =
  forall (i:nat{i < rows * cols}).
    Seq.index result i == Seq.index source (gather_index rows cols count offset rs cs i)

inline_for_extraction noextract
fn gather_kernel
  (rows:szp{rows <= 647}) (cols:szp{cols <= 4096}) (count:szp)
  (source:A.larray bf16 count) (offset rs:sz)
  (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (output:A.larray bf16 (rows * cols))
  (#fraction:perm) (#values:erased (lseq bf16 count))
  preserves gpu ** source |-> Frac fraction values
  requires A.pts_to_uninit output (rows * cols)
  ensures exists* (result:lseq bf16 (rows * cols)).
    output |-> result ** pure (gathered rows cols count offset rs cs values result)
{
  unfold A.pts_to_uninit output (rows * cols);
  let mut i = 0sz;
  while (!i <^ rows *^ cols)
    invariant exists* (j:sz{j <= rows * cols}) (masked:Seq.seq (option bf16)).
      i |-> j ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == rows * cols /\
        (forall (k:nat{k < j}). Seq.index masked k ==
          Some (Seq.index values (gather_index rows cols count offset rs cs k))))
    decreases (rows * cols - !i)
  {
    let j = !i;
    let r = j /^ cols;
    let c = j %^ cols;
    matrix_index_bounds rows cols count offset rs cs r c;
    let value = source.(offset +^ r *^ rs +^ c *^ cs);
    A.mask_write output j value;
    i := j +^ 1sz;
  };
  A.from_mask output;
}

fn gather
  (rows:szp{rows <= 647}) (cols:szp{cols <= 4096}) (count:szp)
  (source:A.larray bf16 count{GA.is_global_array source}) (offset rs:sz)
  (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (output:A.larray bf16 (rows * cols){GA.is_global_array output})
  (#fraction:perm) (#values:erased (lseq bf16 count))
  preserves cpu ** on gpu_loc (source |-> Frac fraction values)
  requires on gpu_loc (A.pts_to_uninit output (rows * cols))
  ensures exists* (result:lseq bf16 (rows * cols)).
    on gpu_loc (output |-> result) ** pure (gathered rows cols count offset rs cs values result)
{
  launch_kernel_1 (fun () -> gather_kernel rows cols count source offset rs cs output #fraction #values);
}

inline_for_extraction noextract
let inserted
  (count n:nat) (offset:nat{offset + n <= count})
  (source:lseq bf16 n) (before after:lseq bf16 count) : prop =
  forall (i:nat{i < count}).
    Seq.index after i ==
      (if offset <= i && i < offset + n then Seq.index source (i - offset) else Seq.index before i)

inline_for_extraction noextract
fn insert_kernel
  (count n:szp) (offset:sz{offset + n <= count})
  (source:A.larray bf16 n) (output:A.larray bf16 count)
  (#fraction:perm) (#values:erased (lseq bf16 n)) (#before:erased (lseq bf16 count))
  preserves gpu ** source |-> Frac fraction values
  requires output |-> before
  ensures exists* (after:lseq bf16 count).
    output |-> after ** pure (inserted count n offset values before after)
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (current:lseq bf16 count).
      i |-> j ** output |-> current **
      pure (forall (k:nat{k < count}).
        Seq.index current k ==
          (if offset <= k && k < offset + j then Seq.index values (k - offset) else Seq.index before k))
    decreases (n - !i)
  {
    let j = !i;
    let value = source.(j);
    output.(offset +^ j) <- value;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn insert
  (count n:szp) (offset:sz{offset + n <= count})
  (source:A.larray bf16 n{GA.is_global_array source})
  (output:A.larray bf16 count{GA.is_global_array output})
  (#fraction:perm) (#values:erased (lseq bf16 n)) (#before:erased (lseq bf16 count))
  preserves cpu ** on gpu_loc (source |-> Frac fraction values)
  requires on gpu_loc (output |-> before)
  ensures exists* (after:lseq bf16 count).
    on gpu_loc (output |-> after) ** pure (inserted count n offset values before after)
{
  launch_kernel_1 (fun () -> insert_kernel count n offset source output #fraction #values #before);
}

inline_for_extraction noextract
let appended (n:nat) (prefix:lseq bf16 n) (current:lseq bf16 512)
  (result:lseq bf16 (n + 512)) : prop =
  (forall (i:nat{i < n}). Seq.index result i == Seq.index prefix i) /\
  (forall (i:nat{i < 512}). Seq.index result (n + i) == Seq.index current i)

inline_for_extraction noextract
fn append_element
  (n:szp{n <= 646 * 512})
  (prefix:A.larray bf16 n) (current:A.larray bf16 512)
  (index:sz{index < n + 512})
  (#fp #fc:perm) (#sp:erased (lseq bf16 n)) (#sc:erased (lseq bf16 512))
  preserves gpu ** prefix |-> Frac fp sp ** current |-> Frac fc sc
  returns result:bf16
  ensures pure (result == (if index < n then Seq.index sp index else Seq.index sc (index - n)))
{
  if (index <^ n) { prefix.(index) } else { current.(index -^ n) }
}

inline_for_extraction noextract
fn append_kernel
  (n:szp{n <= 646 * 512})
  (prefix:A.larray bf16 n) (current:A.larray bf16 512)
  (output:A.larray bf16 (n + 512))
  (#fp #fc:perm) (#sp:erased (lseq bf16 n)) (#sc:erased (lseq bf16 512))
  preserves gpu ** prefix |-> Frac fp sp ** current |-> Frac fc sc
  requires A.pts_to_uninit output (n + 512)
  ensures exists* (result:lseq bf16 (n + 512)).
    output |-> result ** pure (appended n sp sc result)
{
  unfold A.pts_to_uninit output (n + 512);
  let mut i = 0sz;
  while (!i <^ n +^ 512sz)
    invariant exists* (j:sz{j <= n + 512}) (masked:Seq.seq (option bf16)).
      i |-> j ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == n + 512 /\
        (forall (k:nat{k < j}). Seq.index masked k ==
          Some (if k < n then Seq.index sp k else Seq.index sc (k - n))))
    decreases (n + 512 - !i)
  {
    let j = !i;
    let value = append_element n prefix current j #fp #fc #sp #sc;
    A.mask_write output j value;
    i := j +^ 1sz;
  };
  A.from_mask output;
}

fn append
  (n:szp{n <= 646 * 512})
  (prefix:A.larray bf16 n{GA.is_global_array prefix})
  (current:A.larray bf16 512{GA.is_global_array current})
  (output:A.larray bf16 (n + 512){GA.is_global_array output})
  (#fp #fc:perm) (#sp:erased (lseq bf16 n)) (#sc:erased (lseq bf16 512))
  preserves cpu ** on gpu_loc (prefix |-> Frac fp sp ** current |-> Frac fc sc)
  requires on gpu_loc (A.pts_to_uninit output (n + 512))
  ensures exists* (result:lseq bf16 (n + 512)).
    on gpu_loc (output |-> result) ** pure (appended n sp sc result)
{
  launch_kernel_1 (fun () -> append_kernel n prefix current output #fp #fc #sp #sc);
}

inline_for_extraction noextract
let activation_value (silu:bool) (x:real) : real =
  if silu then S.silu x else S.sigmoid x

inline_for_extraction noextract
fn activate (silu:bool) (value:bf16) (#xr:erased real)
  preserves gpu
  requires pure (v_approximates value (FStar.Ghost.reveal #real xr))
  returns result:bf16
  ensures pure (v_approximates result (activation_value silu (FStar.Ghost.reveal #real xr)))
{
  C.fcast_approx #bf16 #f32 value (FStar.Ghost.reveal #real xr);
  let x = C.fcast #bf16 #f32 value;
  let minus_one : f32 = of_int (-1L);
  of_int_approx #f32 (-1L);
  let negative = mul x minus_one;
  a_mul x minus_one (FStar.Ghost.reveal #real xr) (0.0R -. 1.0R);
  let ex = fexp negative;
  exp_approx negative (0.0R -. FStar.Ghost.reveal #real xr);
  E.exp_positive (0.0R -. FStar.Ghost.reveal #real xr);
  let denominator = rounded_add #1.0R #(E.exp (0.0R -. FStar.Ghost.reveal #real xr)) (one #f32) ex;
  let numerator : f32 = (if silu then x else (one #f32));
  let quotient = div numerator denominator;
  div_approx numerator denominator
    (if silu then FStar.Ghost.reveal #real xr else 1.0R)
    (1.0R +. E.exp (0.0R -. FStar.Ghost.reveal #real xr));
  multiply_reciprocal (FStar.Ghost.reveal #real xr) (1.0R +. E.exp (0.0R -. FStar.Ghost.reveal #real xr));
  norm_spec [delta_once ["KPlay.SharedSpec.sigmoid"]] (S.sigmoid (FStar.Ghost.reveal #real xr));
  norm_spec [delta_once ["KPlay.SharedSpec.silu"]] (S.silu (FStar.Ghost.reveal #real xr));
  C.fcast_approx #f32 #bf16 quotient (activation_value silu (FStar.Ghost.reveal #real xr));
  C.fcast #f32 #bf16 quotient
}

inline_for_extraction noextract
let activated_product (silu:bool) (gate other:real) : real =
  if silu then S.silu gate *. other else other *. S.sigmoid gate

inline_for_extraction noextract
fn multiply_activation (silu:bool) (gate other:bf16) (#gr #xr:erased real)
  preserves gpu
  requires pure (
    v_approximates gate (FStar.Ghost.reveal #real gr) /\
    v_approximates other (FStar.Ghost.reveal #real xr))
  returns result:bf16
  ensures pure (v_approximates result
    (activated_product silu (FStar.Ghost.reveal #real gr) (FStar.Ghost.reveal #real xr)))
{
  let activated = activate silu gate #gr;
  C.fcast_approx #bf16 #f32 activated (activation_value silu (FStar.Ghost.reveal #real gr));
  C.fcast_approx #bf16 #f32 other (FStar.Ghost.reveal #real xr);
  let af = C.fcast #bf16 #f32 activated;
  let xf = C.fcast #bf16 #f32 other;
  let left : f32 = (if silu then af else xf);
  let right : f32 = (if silu then xf else af);
  let value = rounded_mul
    #(if silu then activation_value silu (FStar.Ghost.reveal #real gr) else FStar.Ghost.reveal #real xr)
    #(if silu then FStar.Ghost.reveal #real xr else activation_value silu (FStar.Ghost.reveal #real gr))
    left right;
  C.fcast_approx #f32 #bf16 value
    (activated_product silu (FStar.Ghost.reveal #real gr) (FStar.Ghost.reveal #real xr));
  C.fcast #f32 #bf16 value
}

inline_for_extraction noextract
let activation_vector (n:nat) (silu:bool) (gate other:S.vector n) : S.vector n =
  fun i -> activated_product silu (gate i) (other i)

inline_for_extraction noextract
fn activation_kernel
  (n:szp{n <= 3584}) (silu:bool)
  (gate other output:A.larray bf16 n)
  (#fg #fx:perm) (#sg #sx:erased (lseq bf16 n))
  (#gr #xr:erased (S.vector n))
  preserves gpu ** gate |-> Frac fg sg ** other |-> Frac fx sx
  requires A.pts_to_uninit output n
  requires pure (bf_rep n sg gr /\ bf_rep n sx xr)
  ensures exists* (result:lseq bf16 n).
    output |-> result ** pure (bf_rep n result (activation_vector n silu gr xr))
{
  unfold A.pts_to_uninit output n;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (masked:Seq.seq (option bf16)).
      i |-> j ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == n /\
        (forall (k:nat{k < j}). exists (v:bf16).
          Seq.index masked k == Some v /\ v_approximates v (activation_vector n silu gr xr k)))
    decreases (n - !i)
  {
    let j = !i;
    let a = gate.(j);
    let b = other.(j);
    let value = multiply_activation silu a b #(vector_at #n gr j) #(vector_at #n xr j);
    A.mask_write output j value;
    i := j +^ 1sz;
  };
  A.from_mask output;
}

fn activation
  (n:szp{n <= 3584}) (silu:bool)
  (gate:A.larray bf16 n{GA.is_global_array gate})
  (other:A.larray bf16 n{GA.is_global_array other})
  (output:A.larray bf16 n{GA.is_global_array output})
  (#fg #fx:perm) (#sg #sx:erased (lseq bf16 n))
  (#gr #xr:erased (S.vector n))
  preserves cpu ** on gpu_loc (gate |-> Frac fg sg ** other |-> Frac fx sx)
  requires on gpu_loc (A.pts_to_uninit output n)
  requires pure (bf_rep n sg gr /\ bf_rep n sx xr)
  ensures exists* (result:lseq bf16 n).
    on gpu_loc (output |-> result) ** pure (bf_rep n result (activation_vector n silu gr xr))
{
  launch_kernel_1 (fun () -> activation_kernel n silu gate other output #fg #fx #sg #sx #gr #xr);
}

inline_for_extraction noextract
fn residual_kernel
  (left right output:A.larray bf16 1024)
  (#fl #fr:perm) (#sl #sr:erased (lseq bf16 1024))
  (#xl #xr:erased (S.vector 1024))
  preserves gpu ** left |-> Frac fl sl ** right |-> Frac fr sr
  requires A.pts_to_uninit output 1024
  requires pure (bf_rep 1024 sl xl /\ bf_rep 1024 sr xr)
  ensures exists* (result:lseq bf16 1024).
    output |-> result ** pure (bf_rep 1024 result (S.vector_add 1024 xl xr))
{
  unfold A.pts_to_uninit output 1024;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (j:sz{j <= 1024}) (masked:Seq.seq (option bf16)).
      i |-> j ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == 1024 /\
        (forall (k:nat{k < j}). exists (v:bf16).
          Seq.index masked k == Some v /\ v_approximates v (S.vector_add 1024 xl xr k)))
    decreases (1024 - !i)
  {
    let j = !i;
    let a = left.(j);
    let b = right.(j);
    C.fcast_approx #bf16 #f32 a (vector_at #1024 xl j);
    C.fcast_approx #bf16 #f32 b (vector_at #1024 xr j);
    let value = rounded_add #(vector_at #1024 xl j) #(vector_at #1024 xr j)
      (C.fcast #bf16 #f32 a) (C.fcast #bf16 #f32 b);
    C.fcast_approx #f32 #bf16 value (S.vector_add 1024 xl xr j);
    A.mask_write output j (C.fcast #f32 #bf16 value);
    i := j +^ 1sz;
  };
  A.from_mask output;
}

fn residual
  (left:A.larray bf16 1024{GA.is_global_array left})
  (right:A.larray bf16 1024{GA.is_global_array right})
  (output:A.larray bf16 1024{GA.is_global_array output})
  (#fl #fr:perm) (#sl #sr:erased (lseq bf16 1024))
  (#xl #xr:erased (S.vector 1024))
  preserves cpu ** on gpu_loc (left |-> Frac fl sl ** right |-> Frac fr sr)
  requires on gpu_loc (A.pts_to_uninit output 1024)
  requires pure (bf_rep 1024 sl xl /\ bf_rep 1024 sr xr)
  ensures exists* (result:lseq bf16 1024).
    on gpu_loc (output |-> result) ** pure (bf_rep 1024 result (S.vector_add 1024 xl xr))
{
  launch_kernel_1 (fun () -> residual_kernel left right output #fl #fr #sl #sr #xl #xr);
}

inline_for_extraction noextract
let head_vector (heads:pos) (values:S.matrix heads 256) (h:nat{h < heads}) : S.vector 256 =
  values h

inline_for_extraction noextract
let normalized_heads (heads:pos) (weight:S.vector 256) (values:S.matrix heads 256)
  : S.vector (heads * 256) =
  fun i -> S.rms_norm 256 weight (head_vector heads values (i / 256)) (i % 256)

inline_for_extraction noextract
let gathered_head_rep
  (heads:pos) (count:nat) (offset:nat)
  (stride:nat{offset + (heads - 1) * stride + 255 < count})
  (source:lseq bf16 count) (logical:S.matrix heads 256)
  (h:nat{h < heads}) (copied:lseq bf16 256)
  : Lemma
      (requires
        matrix_rep heads 256 count offset stride 1 source logical /\
        gathered 1 256 count (offset + h * stride) 0 1 source copied)
      (ensures bf_rep 256 copied (head_vector heads logical h)) = ()

inline_for_extraction noextract
let head_slot (h:nat) (k:nat{256 * h <= k /\ k < 256 * (h + 1)})
  : Lemma (k / 256 == h /\ k % 256 == k - 256 * h) = ()

inline_for_extraction noextract
let norm_head_insert
  (heads:pos) (weight:S.vector 256) (logical:S.matrix heads 256)
  (h:nat{h < heads})
  (before after:lseq bf16 (heads * 256)) (head:lseq bf16 256)
  : Lemma
      (requires
        (forall (i:nat{i < h * 256}).
          v_approximates (Seq.index before i) (normalized_heads heads weight logical i)) /\
        bf_rep 256 head (S.rms_norm 256 weight (head_vector heads logical h)) /\
        inserted (heads * 256) 256 (h * 256) head before after)
      (ensures forall (i:nat{i < (h + 1) * 256}).
        v_approximates (Seq.index after i) (normalized_heads heads weight logical i)) =
  let point (i:nat{i < (h + 1) * 256})
    : Lemma (v_approximates (Seq.index after i) (normalized_heads heads weight logical i)) =
    if i < h * 256 then ()
    else head_slot h i
  in
  FStar.Classical.forall_intro #(i:nat{i < (h + 1) * 256})
    #(fun i -> v_approximates (Seq.index after i) (normalized_heads heads weight logical i)) point

fn norm_heads
  (heads:szp{heads <= 8}) (count:szp)
  (source:A.larray bf16 count{GA.is_global_array source})
  (offset:sz) (stride:sz{offset + (heads - 1) * stride + 255 < count})
  (weight:A.larray bf16 256{GA.is_global_array weight})
  (output:A.larray bf16 (heads * 256){GA.is_global_array output})
  (#fs #fw:perm) (#ss:erased (lseq bf16 count)) (#sw:erased (lseq bf16 256))
  (#before:erased (lseq bf16 (heads * 256)))
  (#logical:erased (S.matrix heads 256)) (#wr:erased (S.vector 256))
  preserves cpu ** on gpu_loc (source |-> Frac fs ss ** weight |-> Frac fw sw)
  requires on gpu_loc (output |-> before)
  requires pure (matrix_rep heads 256 count offset stride 1 ss logical /\ bf_rep 256 sw wr)
  ensures exists* (result:lseq bf16 (heads * 256)).
    on gpu_loc (output |-> result) ** pure (bf_rep (heads * 256) result (normalized_heads heads wr logical))
{
  let mut h = 0sz;
  while (!h <^ heads)
    invariant exists* (j:sz{j <= heads}) (values:lseq bf16 (heads * 256)).
      h |-> j ** on gpu_loc (output |-> values) **
      pure (forall (i:nat{i < j * 256}).
        v_approximates (Seq.index values i) (normalized_heads heads wr logical i))
    decreases (heads - !h)
  {
    let j = !h;
    matrix_index_bounds heads 256 count offset stride 1 j 255;
    let copied = allocate_bf 256sz;
    let normalized = allocate_bf 256sz;
    gather 1sz 256sz count source (offset +^ j *^ stride) 0sz 1sz copied #fs #ss;
    with sx. assert on gpu_loc (copied |-> sx);
    gathered_head_rep heads count offset stride ss logical j sx;
    execute_rms 256sz copied weight normalized #1.0R #fw #sx #sw #(head_vector heads logical j) #wr;
    with sy. assert on gpu_loc (normalized |-> sy);
    with current. assert on gpu_loc (output |-> current);
    insert (heads *^ 256sz) 256sz (j *^ 256sz) normalized output #1.0R #sy #current;
    with next. assert on gpu_loc (output |-> next);
    norm_head_insert heads wr logical j current next sy;
    GA.gpu_array_free copied;
    GA.gpu_array_free normalized;
    h := j +^ 1sz;
  }
}

inline_for_extraction noextract
let rope_values (position:nat) : S.vector 64 =
  fun i -> if i < 32 then S.rope_cos position i else S.rope_sin position (i - 32)

inline_for_extraction noextract
let rotated_heads (heads:pos) (position:nat) (values:S.vector (heads * 256))
  : S.vector (heads * 256) =
  fun i -> S.rotate_partial position
    (fun d -> values (256 * (i / 256) + d)) (i % 256)

inline_for_extraction noextract
let rotate_rounded_value (position:nat) (x y:real) (j:S.index 32) (negative:bool) : real =
  x *. S.rope_cos position j +.
    (if negative then 0.0R -. y else y) *. S.rope_sin position j

inline_for_extraction noextract
fn rotate_pair_value
  (negative:bool) (x y cosine sine:bf16)
  (#xr #yr #cr #sr:erased real)
  preserves gpu
  requires pure (
    v_approximates x (FStar.Ghost.reveal #real xr) /\
    v_approximates y (FStar.Ghost.reveal #real yr) /\
    v_approximates cosine (FStar.Ghost.reveal #real cr) /\
    v_approximates sine (FStar.Ghost.reveal #real sr))
  returns result:bf16
  ensures pure (v_approximates result
    (FStar.Ghost.reveal #real xr *. FStar.Ghost.reveal #real cr +.
      (if negative then 0.0R -. FStar.Ghost.reveal #real yr else FStar.Ghost.reveal #real yr) *.
      FStar.Ghost.reveal #real sr))
{
  C.fcast_approx #bf16 #f32 x (FStar.Ghost.reveal #real xr);
  C.fcast_approx #bf16 #f32 y (FStar.Ghost.reveal #real yr);
  C.fcast_approx #bf16 #f32 cosine (FStar.Ghost.reveal #real cr);
  C.fcast_approx #bf16 #f32 sine (FStar.Ghost.reveal #real sr);
  let xf = C.fcast #bf16 #f32 x;
  let yf = C.fcast #bf16 #f32 y;
  let cf = C.fcast #bf16 #f32 cosine;
  let sf = C.fcast #bf16 #f32 sine;
  let minus_one : f32 = of_int (-1L);
  of_int_approx #f32 (-1L);
  let signed_y : f32 = (if negative then mul yf minus_one else yf);
  a_mul yf minus_one (FStar.Ghost.reveal #real yr) (0.0R -. 1.0R);
  let first = rounded_mul #xr #cr xf cf;
  let second = rounded_mul
    #(if negative then 0.0R -. FStar.Ghost.reveal #real yr else FStar.Ghost.reveal #real yr) #sr signed_y sf;
  C.fcast_approx #f32 #bf16 first (FStar.Ghost.reveal #real xr *. FStar.Ghost.reveal #real cr);
  C.fcast_approx #f32 #bf16 second
    ((if negative then 0.0R -. FStar.Ghost.reveal #real yr else FStar.Ghost.reveal #real yr) *.
      FStar.Ghost.reveal #real sr);
  let first16 = C.fcast #f32 #bf16 first;
  let second16 = C.fcast #f32 #bf16 second;
  C.fcast_approx #bf16 #f32 first16 (FStar.Ghost.reveal #real xr *. FStar.Ghost.reveal #real cr);
  C.fcast_approx #bf16 #f32 second16
    ((if negative then 0.0R -. FStar.Ghost.reveal #real yr else FStar.Ghost.reveal #real yr) *.
      FStar.Ghost.reveal #real sr);
  let combined = rounded_add
    #(FStar.Ghost.reveal #real xr *. FStar.Ghost.reveal #real cr)
    #((if negative then 0.0R -. FStar.Ghost.reveal #real yr else FStar.Ghost.reveal #real yr) *.
      FStar.Ghost.reveal #real sr)
    (C.fcast #bf16 #f32 first16) (C.fcast #bf16 #f32 second16);
  C.fcast_approx #f32 #bf16 combined
    (FStar.Ghost.reveal #real xr *. FStar.Ghost.reveal #real cr +.
      (if negative then 0.0R -. FStar.Ghost.reveal #real yr else FStar.Ghost.reveal #real yr) *.
      FStar.Ghost.reveal #real sr);
  C.fcast #f32 #bf16 combined
}

inline_for_extraction noextract
fn rope_element
  (heads:szp{heads <= 8})
  (input:A.larray bf16 (heads * 256)) (rope:A.larray bf16 64)
  (index:sz{index < heads * 256})
  (#fi #fr:perm) (#si:erased (lseq bf16 (heads * 256))) (#sr:erased (lseq bf16 64))
  (#position:erased nat) (#logical:erased (S.vector (heads * 256)))
  preserves gpu ** input |-> Frac fi si ** rope |-> Frac fr sr
  requires pure (bf_rep (heads * 256) si logical /\ bf_rep 64 sr (rope_values position))
  returns result:bf16
  ensures pure (v_approximates result (rotated_heads heads position logical index))
{
  let d = index %^ 256sz;
  let value = input.(index);
  if (d <^ 64sz) {
    let j = d %^ 32sz;
    let paired : sz = (if d <^ 32sz then index +^ 32sz else index -^ 32sz);
    let y = input.(paired);
    let cosine = rope.(j);
    let sine = rope.(32sz +^ j);
    let result = rotate_pair_value (d <^ 32sz) value y cosine sine
      #(vector_at #(heads * 256) logical index)
      #(vector_at #(heads * 256) logical paired)
      #(S.rope_cos position j) #(S.rope_sin position j);
    norm_spec [delta_once ["KPlay.SharedSpec.rotate_partial"]]
      (S.rotate_partial position (fun k -> vector_at #(heads * 256) logical (256 * (index / 256) + k)) d);
    result
  } else {
    value
  }
}

inline_for_extraction noextract
fn rope_kernel
  (heads:szp{heads <= 8})
  (input output:A.larray bf16 (heads * 256)) (rope:A.larray bf16 64)
  (#fi #fr:perm) (#si:erased (lseq bf16 (heads * 256))) (#sr:erased (lseq bf16 64))
  (#position:erased nat) (#logical:erased (S.vector (heads * 256)))
  preserves gpu ** input |-> Frac fi si ** rope |-> Frac fr sr
  requires A.pts_to_uninit output (heads * 256)
  requires pure (bf_rep (heads * 256) si logical /\ bf_rep 64 sr (rope_values position))
  ensures exists* (result:lseq bf16 (heads * 256)).
    output |-> result ** pure (bf_rep (heads * 256) result (rotated_heads heads position logical))
{
  unfold A.pts_to_uninit output (heads * 256);
  let mut i = 0sz;
  while (!i <^ heads *^ 256sz)
    invariant exists* (j:sz{j <= heads * 256}) (masked:Seq.seq (option bf16)).
      i |-> j ** A.pts_to_mask output masked (fun _ -> True) **
      pure (Seq.length masked == heads * 256 /\
        (forall (k:nat{k < j}). exists (v:bf16). Seq.index masked k == Some v /\
          v_approximates v (rotated_heads heads position logical k)))
    decreases (heads * 256 - !i)
  {
    let j = !i;
    let value = rope_element heads input rope j #fi #fr #si #sr #position #logical;
    A.mask_write output j value;
    i := j +^ 1sz;
  };
  A.from_mask output;
}

fn rotate
  (heads:szp{heads <= 8})
  (input:A.larray bf16 (heads * 256){GA.is_global_array input})
  (output:A.larray bf16 (heads * 256){GA.is_global_array output})
  (rope:A.larray bf16 64{GA.is_global_array rope})
  (#fi #fr:perm) (#si:erased (lseq bf16 (heads * 256))) (#sr:erased (lseq bf16 64))
  (#position:erased nat) (#logical:erased (S.vector (heads * 256)))
  preserves cpu ** on gpu_loc (input |-> Frac fi si ** rope |-> Frac fr sr)
  requires on gpu_loc (A.pts_to_uninit output (heads * 256))
  requires pure (bf_rep (heads * 256) si logical /\ bf_rep 64 sr (rope_values position))
  ensures exists* (result:lseq bf16 (heads * 256)).
    on gpu_loc (output |-> result) ** pure (bf_rep (heads * 256) result (rotated_heads heads position logical))
{
  launch_kernel_1 (fun () -> rope_kernel heads input output rope #fi #fr #si #sr #position #logical);
}

inline_for_extraction noextract
let flat_heads (heads:pos) (values:S.matrix heads 256) : S.vector (heads * 256) =
  fun i -> values (i / 256) (i % 256)

inline_for_extraction noextract
let history_rep (n:pos)
  (keys values:lseq bf16 (n * 512)) (history:S.kv_history{history.length == n}) : prop =
  (forall (h:S.index 2) (t:S.index n) (d:S.index 256).
    v_approximates (Seq.index keys (512 * t + 256 * h + d)) (history.keys h t d)) /\
  (forall (h:S.index 2) (t:S.index n) (d:S.index 256).
    v_approximates (Seq.index values (512 * t + 256 * h + d)) (history.values h t d))

inline_for_extraction noextract
let score_matrix (n:pos) (history:S.kv_history{history.length == n}) (head:S.index 8)
  : S.matrix n 256 =
  fun t d -> history.keys (S.kv_head head) t d

inline_for_extraction noextract
let value_matrix (n:pos) (history:S.kv_history{history.length == n}) (head:S.index 8)
  : S.matrix 256 n =
  fun d t -> history.values (S.kv_head head) t d

inline_for_extraction noextract
let score_dots (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8)
  : S.vector (position + 1) =
  fun t -> S.sum 256 (fun d -> queries head d *. history.keys (S.kv_head head) t d)

inline_for_extraction noextract
let attention_scores (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8)
  : S.vector (position + 1) =
  fun t -> score_dots position queries history head t *. (1.0R /. 16.0R) +. S.causal_bias position t

inline_for_extraction noextract
let attention_probabilities (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8)
  : S.vector (position + 1) =
  S.softmax (position + 1) (attention_scores position queries history head)

inline_for_extraction noextract
let bf_rep_congr (n:nat) (storage:lseq bf16 n) (f g:S.vector n)
  : Lemma (requires bf_rep n storage f /\ (forall (i:S.index n). f i == g i))
          (ensures bf_rep n storage g) = ()

inline_for_extraction noextract
let softmax_congr (n:pos) (f g:S.vector n) (i:S.index n)
  : Lemma
      (requires forall (j:S.index n). f j == g j)
      (ensures S.softmax n f i == S.softmax n g i) =
  sum_ext n (exponentials n f 0.0R) (exponentials n g 0.0R);
  exponential_sum_bridge n f 0.0R;
  exponential_sum_bridge n g 0.0R;
  softmax_denominator n f i;
  softmax_denominator n g i

inline_for_extraction noextract
let history_matrices
  (n:pos) (keys values:lseq bf16 (n * 512))
  (history:S.kv_history{history.length == n}) (head:S.index 8)
  : Lemma
      (requires history_rep n keys values history)
      (ensures
        matrix_rep n 256 (n * 512) (S.kv_head head * 256) 512 1 keys (score_matrix n history head) /\
        matrix_rep 256 n (n * 512) (S.kv_head head * 256) 1 512 values (value_matrix n history head)) = ()

inline_for_extraction noextract
let score_linear (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8) (t:S.index (position + 1))
  : Lemma (
      S.linear (position + 1) 256 (score_matrix (position + 1) history head) (queries head) t ==
      score_dots position queries history head t) =
  sum_ext 256
    (fun d -> score_matrix (position + 1) history head t d *. queries head d)
    (fun d -> queries head d *. history.keys (S.kv_head head) t d);
  norm_spec [delta_once ["KPlay.SharedSpec.linear"]]
    (S.linear (position + 1) 256 (score_matrix (position + 1) history head) (queries head) t)

inline_for_extraction noextract
let score_representation (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8)
  (raw:lseq bf16 (position + 1))
  : Lemma
      (requires bf_rep (position + 1) raw
        (S.linear (position + 1) 256 (score_matrix (position + 1) history head) (queries head)))
      (ensures bf_rep (position + 1) raw (score_dots position queries history head)) =
  let point (t:S.index (position + 1))
    : Lemma (
        S.linear (position + 1) 256 (score_matrix (position + 1) history head) (queries head) t ==
        score_dots position queries history head t) =
    score_linear position queries history head t
  in
  FStar.Classical.forall_intro #(S.index (position + 1))
    #(fun t ->
      S.linear (position + 1) 256 (score_matrix (position + 1) history head) (queries head) t ==
      score_dots position queries history head t) point;
  bf_rep_congr (position + 1) raw
    (S.linear (position + 1) 256 (score_matrix (position + 1) history head) (queries head))
    (score_dots position queries history head)

inline_for_extraction noextract
let probabilities_representation (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8)
  (raw:lseq bf16 (position + 1))
  : Lemma
      (requires bf_rep (position + 1) raw
        (S.softmax (position + 1) (scaled_scores (position + 1) (score_dots position queries history head))))
      (ensures bf_rep (position + 1) raw (attention_probabilities position queries history head)) =
  let f = scaled_scores (position + 1) (score_dots position queries history head) in
  let g = attention_scores position queries history head in
  let point (i:S.index (position + 1)) : Lemma (S.softmax (position + 1) f i == S.softmax (position + 1) g i) =
    softmax_congr (position + 1) f g i
  in
  FStar.Classical.forall_intro #(S.index (position + 1))
    #(fun i -> S.softmax (position + 1) f i == S.softmax (position + 1) g i) point;
  bf_rep_congr (position + 1) raw (S.softmax (position + 1) f) (attention_probabilities position queries history head)

inline_for_extraction noextract
let attention_unfold (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8) (d:S.index 256)
  : Lemma (
      S.attention position queries history head d ==
      S.sum (position + 1) (fun t ->
        attention_probabilities position queries history head t *. history.values (S.kv_head head) t d)) =
  assert (
      S.attention position queries history head d ==
      S.sum (position + 1) (fun t ->
        attention_probabilities position queries history head t *. history.values (S.kv_head head) t d))
    by (
      FStar.Tactics.V2.norm [delta_only [
        "KPlay.SharedSpec.attention";
        "KPlay.Functional.attention_probabilities";
        "KPlay.Functional.attention_scores";
        "KPlay.Functional.score_dots"
      ]];
      FStar.Tactics.V2.trefl ()
    )

inline_for_extraction noextract
let context_linear (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8) (d:S.index 256)
  : Lemma (
      S.linear 256 (position + 1) (value_matrix (position + 1) history head)
        (attention_probabilities position queries history head) d ==
      S.attention position queries history head d) =
  attention_unfold position queries history head d;
  sum_ext (position + 1)
    (fun t -> value_matrix (position + 1) history head d t *. attention_probabilities position queries history head t)
    (fun t -> attention_probabilities position queries history head t *. history.values (S.kv_head head) t d);
  norm_spec [delta_once ["KPlay.SharedSpec.linear"]]
    (S.linear 256 (position + 1) (value_matrix (position + 1) history head)
      (attention_probabilities position queries history head) d)

inline_for_extraction noextract
let context_representation (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8) (raw:lseq bf16 256)
  : Lemma
      (requires bf_rep 256 raw
        (S.linear 256 (position + 1) (value_matrix (position + 1) history head)
          (attention_probabilities position queries history head)))
      (ensures bf_rep 256 raw (S.attention position queries history head)) =
  let point (d:S.index 256)
    : Lemma (
        S.linear 256 (position + 1) (value_matrix (position + 1) history head)
          (attention_probabilities position queries history head) d ==
        S.attention position queries history head d) =
    context_linear position queries history head d
  in
  FStar.Classical.forall_intro #(S.index 256)
    #(fun d ->
      S.linear 256 (position + 1) (value_matrix (position + 1) history head)
        (attention_probabilities position queries history head) d ==
      S.attention position queries history head d) point;
  bf_rep_congr 256 raw
    (S.linear 256 (position + 1) (value_matrix (position + 1) history head)
      (attention_probabilities position queries history head))
    (S.attention position queries history head)

inline_for_extraction noextract
let gathered_query
  (raw:lseq bf16 2048) (queries:S.matrix 8 256) (h:S.index 8) (copied:lseq bf16 256)
  : Lemma
      (requires bf_rep 2048 raw (flat_heads 8 queries) /\ gathered 1 256 2048 (h * 256) 0 1 raw copied)
      (ensures bf_rep 256 copied (queries h)) = ()

inline_for_extraction noextract
let context_insert
  (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (h:S.index 8)
  (before after:lseq bf16 2048) (head:lseq bf16 256)
  : Lemma
      (requires
        (forall (i:nat{i < h * 256}).
          v_approximates (Seq.index before i) (flat_heads 8 (S.attention position queries history) i)) /\
        bf_rep 256 head (S.attention position queries history h) /\
        inserted 2048 256 (h * 256) head before after)
      (ensures forall (i:nat{i < (h + 1) * 256}).
        v_approximates (Seq.index after i) (flat_heads 8 (S.attention position queries history) i)) =
  let point (i:nat{i < (h + 1) * 256})
    : Lemma (v_approximates (Seq.index after i) (flat_heads 8 (S.attention position queries history) i)) =
    if i < h * 256 then () else head_slot h i
  in
  FStar.Classical.forall_intro #(i:nat{i < (h + 1) * 256})
    #(fun i -> v_approximates (Seq.index after i) (flat_heads 8 (S.attention position queries history) i)) point

fn score_projection
  (position:sz{position <= 646}) (head:sz{head < 8})
  (keys:A.larray bf16 ((position + 1) * 512){GA.is_global_array keys})
  (query:A.larray bf16 256{GA.is_global_array query})
  (output:A.larray bf16 (position + 1){GA.is_global_array output})
  (#fk #fq:perm)
  (#sk:erased (lseq bf16 ((position + 1) * 512))) (#sq:erased (lseq bf16 256))
  (#history:(h:erased S.kv_history{(FStar.Ghost.reveal #S.kv_history h).length == position + 1}))
  (#queries:erased (S.matrix 8 256))
  preserves cpu ** on gpu_loc (keys |-> Frac fk sk ** query |-> Frac fq sq)
  requires on gpu_loc (A.pts_to_uninit output (position + 1))
  requires pure (
    matrix_rep (position + 1) 256 ((position + 1) * 512) (S.kv_head head * 256) 512 1 sk
      (score_matrix (position + 1) history head) /\
    bf_rep 256 sq (head_vector 8 queries head))
  ensures exists* (result:lseq bf16 (position + 1)).
    on gpu_loc (output |-> result) **
    pure (bf_rep (position + 1) result
      (S.linear (position + 1) 256 (score_matrix (position + 1) history head) (head_vector 8 queries head)))
{
  let n = position +^ 1sz;
  let offset = (head /^ 4sz) *^ 256sz;
  if (n %^ 8sz = 0sz) {
    matvec n 256sz (n *^ 512sz) keys offset 512sz 1sz query output false
      #fk #fq #sk #sq #(score_matrix (position + 1) history head) #(head_vector 8 queries head);
  } else {
    let linear : bool = ((n >=^ 388sz && n <=^ 415sz) || n >=^ 444sz);
    let interleaved : bool = (n >=^ 417sz && n <=^ 443sz);
    let groups : szp = (if n >=^ 57sz && n <=^ 139sz then 32sz else 16sz);
    let chunk : szp = (if linear || interleaved then 1sz else 256sz /^ groups);
    simt n 256sz (n *^ 512sz) keys offset 512sz 1sz query output groups chunk linear
      #fk #fq #sk #sq #(score_matrix (position + 1) history head) #(head_vector 8 queries head);
  }
}

fn value_projection
  (position:sz{position <= 646}) (head:sz{head < 8})
  (values:A.larray bf16 ((position + 1) * 512){GA.is_global_array values})
  (probabilities:A.larray bf16 (position + 1){GA.is_global_array probabilities})
  (output:A.larray bf16 256{GA.is_global_array output})
  (#fv #fp:perm)
  (#sv:erased (lseq bf16 ((position + 1) * 512))) (#sp:erased (lseq bf16 (position + 1)))
  (#history:(h:erased S.kv_history{(FStar.Ghost.reveal #S.kv_history h).length == position + 1}))
  (#queries:erased (S.matrix 8 256))
  preserves cpu ** on gpu_loc (values |-> Frac fv sv ** probabilities |-> Frac fp sp)
  requires on gpu_loc (A.pts_to_uninit output 256)
  requires pure (
    matrix_rep 256 (position + 1) ((position + 1) * 512) (S.kv_head head * 256) 1 512 sv
      (value_matrix (position + 1) history head) /\
    bf_rep (position + 1) sp (attention_probabilities position queries history head))
  ensures exists* (result:lseq bf16 256).
    on gpu_loc (output |-> result) **
    pure (bf_rep 256 result
      (S.linear 256 (position + 1) (value_matrix (position + 1) history head)
        (attention_probabilities position queries history head)))
{
  let n = position +^ 1sz;
  let offset = (head /^ 4sz) *^ 256sz;
  if (n %^ 8sz = 0sz) {
    matvec 256sz n (n *^ 512sz) values offset 1sz 512sz probabilities output false
      #fv #fp #sv #sp #(value_matrix (position + 1) history head) #(attention_probabilities position queries history head);
  } else {
    let groups : szp = (if n <^ 426sz then 32sz else 16sz);
    let chunk : szp = (if n <^ 426sz then 4sz else (n +^ 15sz) /^ 16sz);
    simt 256sz n (n *^ 512sz) values offset 1sz 512sz probabilities output groups chunk false
      #fv #fp #sv #sp #(value_matrix (position + 1) history head) #(attention_probabilities position queries history head);
  }
}

fn execute_attention
  (position:sz{position <= 646})
  (query:A.larray bf16 2048{GA.is_global_array query})
  (keys:A.larray bf16 ((position + 1) * 512){GA.is_global_array keys})
  (values:A.larray bf16 ((position + 1) * 512){GA.is_global_array values})
  (output:A.larray bf16 2048{GA.is_global_array output})
  (#fq #fk #fv:perm) (#sq:erased (lseq bf16 2048))
  (#sk #sv:erased (lseq bf16 ((position + 1) * 512))) (#before:erased (lseq bf16 2048))
  (#queries:erased (S.matrix 8 256))
  (#history:(h:erased S.kv_history{(FStar.Ghost.reveal #S.kv_history h).length == position + 1}))
  preserves cpu ** on gpu_loc (query |-> Frac fq sq ** keys |-> Frac fk sk ** values |-> Frac fv sv)
  requires on gpu_loc (output |-> before)
  requires pure (bf_rep 2048 sq (flat_heads 8 queries) /\ history_rep (position + 1) sk sv history)
  ensures exists* (result:lseq bf16 2048).
    on gpu_loc (output |-> result) ** pure (bf_rep 2048 result (flat_heads 8 (S.attention position queries history)))
{
  let n = position +^ 1sz;
  let mut head = 0sz;
  while (!head <^ 8sz)
    invariant exists* (h:sz{h <= 8}) (current:lseq bf16 2048).
      head |-> h ** on gpu_loc (output |-> current) **
      pure (forall (i:nat{i < h * 256}).
        v_approximates (Seq.index current i) (flat_heads 8 (S.attention position queries history) i))
    decreases (8 - !head)
  {
    let h = !head;
    history_matrices n sk sv history h;
    let q = allocate_bf 256sz;
    let scores = allocate_bf n;
    let probabilities = allocate_bf n;
    let context = allocate_bf 256sz;
    gather 1sz 256sz 2048sz query (h *^ 256sz) 0sz 1sz q #fq #sq;
    with qdata. assert on gpu_loc (q |-> qdata);
    gathered_query sq queries h qdata;
    score_projection position h keys q scores #fk #1.0R #sk #qdata #history #queries;
    with sdata. assert on gpu_loc (scores |-> sdata);
    score_representation position queries history h sdata;
    softmax n scores probabilities #1.0R #sdata #(score_dots position queries history h);
    with pdata. assert on gpu_loc (probabilities |-> pdata);
    probabilities_representation position queries history h pdata;
    value_projection position h values probabilities context #fv #1.0R #sv #pdata #history #queries;
    with cdata. assert on gpu_loc (context |-> cdata);
    context_representation position queries history h cdata;
    with old. assert on gpu_loc (output |-> old);
    insert 2048sz 256sz (h *^ 256sz) context output #1.0R #cdata #old;
    with next. assert on gpu_loc (output |-> next);
    context_insert position queries history h old next cdata;
    GA.gpu_array_free q;
    GA.gpu_array_free scores;
    GA.gpu_array_free probabilities;
    GA.gpu_array_free context;
    head := h +^ 1sz;
  }
}

#push-options "--fuel 4 --ifuel 2 --z3rlimit 20"

inline_for_extraction noextract
let complete_history_pointwise
  (p:nat) (old_keys old_values:S.vector (prefix_count p))
  (new_key new_value:S.matrix 2 256)
  (h:S.index 2) (t:S.index (p + 1)) (d:S.index 256)
  : Lemma (
      let next = S.append_kv p (old_history_view p old_keys old_values) new_key new_value in
      let mapped = history_view (p + 1)
        (flatten_kv (p + 1) next.keys) (flatten_kv (p + 1) next.values) in
      mapped.keys h t d == next.keys h t d /\
      mapped.values h t d == next.values h t d) =
  let next = S.append_kv p (old_history_view p old_keys old_values) new_key new_value in
  kv_logical_roundtrip (p + 1) next.keys h t d;
  kv_logical_roundtrip (p + 1) next.values h t d

inline_for_extraction noextract
let decoded_head (h:S.index 2) (t:nat) (d:S.index 256)
  : Lemma (((512 * t + 256 * h + d) % 512) / 256 == h) = ()

inline_for_extraction noextract
let decoded_time (h:S.index 2) (t:nat) (d:S.index 256)
  : Lemma ((512 * t + 256 * h + d) / 512 == t) = ()

inline_for_extraction noextract
let decoded_channel (h:S.index 2) (t:nat) (d:S.index 256)
  : Lemma ((512 * t + 256 * h + d) % 256 == d) = ()

inline_for_extraction noextract
let initial_history_equality
  (old_keys old_values:S.vector 0)
  (new_key new_value:S.matrix 2 256)
  : Lemma (
      let next = S.append_kv 0 (old_history_view 0 old_keys old_values) new_key new_value in
      history_view 1
        (flatten_kv 1 next.keys) (flatten_kv 1 next.values) == next) =
  assert (
      let next = S.append_kv 0 (old_history_view 0 old_keys old_values) new_key new_value in
      history_view 1
        (flatten_kv 1 next.keys) (flatten_kv 1 next.values) == next)
    by (
      Proof.norm [delta_only [
        "KPlay.SharedSpec.append_kv";
        "KPlay.Functional.history_view";
        "KPlay.Functional.old_history_view";
        "KPlay.Functional.flatten_kv";
        "KPlay.Functional.kv_view"
      ]];
      Proof.l_to_r [`decoded_head; `decoded_time; `decoded_channel];
      Proof.trefl ()
    )

inline_for_extraction noextract
let positive_old_history
  (p:pos) (old_keys old_values:S.vector (prefix_count p))
  : Lemma (
      old_history_view p old_keys old_values ==
      Some (history_view p old_keys old_values)) = ()

inline_for_extraction noextract
let positive_history_equality
  (p:pos) (old_keys old_values:S.vector (prefix_count p))
  (new_key new_value:S.matrix 2 256)
  : Lemma (
      let next = S.append_kv p (old_history_view p old_keys old_values) new_key new_value in
      history_view (p + 1)
        (flatten_kv (p + 1) next.keys) (flatten_kv (p + 1) next.values) == next) =
  assert (
      let next = S.append_kv p (old_history_view p old_keys old_values) new_key new_value in
      history_view (p + 1)
        (flatten_kv (p + 1) next.keys) (flatten_kv (p + 1) next.values) == next)
    by (
      Proof.l_to_r [`positive_old_history];
      Proof.norm [delta_only [
        "KPlay.SharedSpec.append_kv";
        "KPlay.Functional.history_view";
        "KPlay.Functional.flatten_kv";
        "KPlay.Functional.kv_view"
      ]];
      Proof.l_to_r [`decoded_head; `decoded_time; `decoded_channel];
      Proof.trefl ()
    )

inline_for_extraction noextract
let complete_history_equality
  (p:nat) (old_keys old_values:S.vector (prefix_count p))
  (new_key new_value:S.matrix 2 256)
  : Lemma (
      let next = S.append_kv p (old_history_view p old_keys old_values) new_key new_value in
      history_view (p + 1)
        (flatten_kv (p + 1) next.keys) (flatten_kv (p + 1) next.values) == next) =
  if p = 0 then initial_history_equality old_keys old_values new_key new_value
  else positive_history_equality p old_keys old_values new_key new_value

inline_for_extraction noextract
let absorb_selection (selected:bool) (next old frame:option S.kv_history)
  : Lemma (
      (if selected then next else if selected then old else frame) ==
      (if selected then next else frame)) = ()

inline_for_extraction noextract
let cache_slot
  (call:S.invocation) (frame:S.hybrid_cache)
  (old_keys old_values:S.vector (prefix_count call.position))
  (l:S.layer_index)
  : Lemma (
      (cache_view call frame old_keys old_values).kv l ==
      (if l = call.layer
       then old_history_view call.position old_keys old_values
       else frame.kv l)) = ()

inline_for_extraction noextract
let cache_update_slot
  (call:S.invocation) (frame:S.hybrid_cache)
  (old_keys old_values:S.vector (prefix_count call.position))
  (next:S.kv_history) (l:S.layer_index)
  : Lemma (
      (if l = call.layer then Some next
       else (cache_view call frame old_keys old_values).kv l) ==
      (if l = call.layer then Some next else frame.kv l)) =
  cache_slot call frame old_keys old_values l

inline_for_extraction noextract
let unselected_cache_slot
  (call:S.invocation) (frame:S.hybrid_cache)
  (old_keys old_values:S.vector (prefix_count call.position))
  (l:S.layer_index{l <> call.layer})
  : Lemma ((cache_view call frame old_keys old_values).kv l == frame.kv l) =
  cache_slot call frame old_keys old_values l

inline_for_extraction noextract
let choose_kv (selected:bool) (next frame:option S.kv_history) : option S.kv_history =
  if selected then next else frame

inline_for_extraction noextract
let choose_twice (selected:bool) (next old frame:option S.kv_history)
  : Lemma (
      choose_kv selected next (choose_kv selected old frame) ==
      choose_kv selected next frame) = ()

inline_for_extraction noextract
let choose_equal_conditions
  (selected also_selected:bool{selected == also_selected})
  (next old frame:option S.kv_history)
  : Lemma (
      choose_kv selected next (choose_kv also_selected old frame) ==
      choose_kv selected next frame) = ()

inline_for_extraction noextract
let composed_cache
  (call:S.invocation) (frame:S.hybrid_cache)
  (old_keys old_values:S.vector (prefix_count call.position))
  (next:S.kv_history) : S.hybrid_cache = {
  kv = (fun l -> choose_kv (Prims.op_Equals #S.layer_index l call.layer) (Some next)
    (choose_kv (l = call.layer)
      (old_history_view call.position old_keys old_values) (frame.kv l)));
  convolution = frame.convolution;
  recurrent = frame.recurrent
}

inline_for_extraction noextract
let single_cache
  (call:S.invocation) (frame:S.hybrid_cache) (next:S.kv_history) : S.hybrid_cache = {
  kv = (fun l -> choose_kv (Prims.op_Equals #S.layer_index l call.layer) (Some next) (frame.kv l));
  convolution = frame.convolution;
  recurrent = frame.recurrent
}

inline_for_extraction noextract
let replace_cache_frame
  (call:S.invocation) (frame:S.hybrid_cache)
  (old_keys old_values:S.vector (prefix_count call.position))
  (next:S.kv_history)
  : Lemma (
      S.replace_kv (cache_view call frame old_keys old_values) call.layer next ==
      S.replace_kv frame call.layer next) =
  assert (
      S.replace_kv (cache_view call frame old_keys old_values) call.layer next ==
      composed_cache call frame old_keys old_values next)
    by (
      Proof.norm [delta_only [
        "KPlay.SharedSpec.replace_kv"; "KPlay.Functional.cache_view";
        "KPlay.Functional.composed_cache"; "KPlay.Functional.choose_kv"
      ]];
      Proof.l_to_r [`cache_slot];
      Proof.trefl ()
    );
  assert (composed_cache call frame old_keys old_values next == single_cache call frame next)
    by (
      Proof.norm [delta_only [
        "KPlay.Functional.composed_cache"; "KPlay.Functional.single_cache"
      ]];
      Proof.l_to_r [`choose_equal_conditions];
      Proof.trefl ()
    );
  assert (single_cache call frame next == S.replace_kv frame call.layer next)
    by (
      Proof.norm [delta_only [
        "KPlay.SharedSpec.replace_kv";
        "KPlay.Functional.single_cache"; "KPlay.Functional.choose_kv"
      ]];
      Proof.trefl ()
    )

inline_for_extraction noextract
let result_cache_refinement
  (call:S.invocation) (frame:S.hybrid_cache)
  (old_keys old_values:S.vector (prefix_count call.position))
  (new_key new_value:S.matrix 2 256)
  (hidden:S.vector 1024)
  : Lemma (
      let next = S.append_kv call.position
        (old_history_view call.position old_keys old_values) new_key new_value in
      let mapped = result_view call frame hidden
        (flatten_kv (call.position + 1) next.keys)
        (flatten_kv (call.position + 1) next.values) in
      mapped.cache ==
        S.replace_kv (cache_view call frame old_keys old_values) call.layer next) =
  let next = S.append_kv call.position
    (old_history_view call.position old_keys old_values) new_key new_value in
  complete_history_equality call.position old_keys old_values new_key new_value;
  replace_cache_frame call frame old_keys old_values next

#pop-options

inline_for_extraction noextract
let physical_rep (n:nat) (storage:bfseq n) (logical:S.vector n)
  : Lemma (represents n storage logical <==> bf_rep n storage logical) = ()

inline_for_extraction noextract
let physical_counts (position:nat)
  : Lemma (
      kv_width == 512 /\ hidden_width == 1024 /\ rope_width == 64 /\
      prefix_count position == position * 512 /\
      next_count position == prefix_count position + 512 /\
      next_count position == (position + 1) * 512 /\
      (position + 1) * kv_width == prefix_count position + 512) = ()

inline_for_extraction noextract
let rope_input_rep (position:nat) (storage:bfseq 64)
  : Lemma (requires rope_represents position storage)
          (ensures bf_rep 64 storage (rope_values position)) = ()

inline_for_extraction noextract
let all_weight_matrices (storage:bfseq weight_count) (packed:S.vector weight_count)
  : Lemma
      (requires represents weight_count storage packed)
      (ensures
        matrix_rep 4096 1024 weight_count 11012096 1024 1 storage (weights_view packed).query_gate_proj /\
        matrix_rep 512 1024 weight_count 15206400 1024 1 storage (weights_view packed).key_proj /\
        matrix_rep 512 1024 weight_count 15730688 1024 1 storage (weights_view packed).value_proj /\
        matrix_rep 1024 2048 weight_count 16255488 2048 1 storage (weights_view packed).output_proj /\
        matrix_rep 3584 1024 weight_count 2048 1024 1 storage (weights_view packed).mlp_gate_proj /\
        matrix_rep 3584 1024 weight_count 3672064 1024 1 storage (weights_view packed).mlp_up_proj /\
        matrix_rep 1024 3584 weight_count 7342080 3584 1 storage (weights_view packed).mlp_down_proj) = ()

fn load_weight_vector
  (n:szp{n <= 1024}) (offset:sz{offset + n <= weight_count})
  (weights:A.larray bf16 weight_count{GA.is_global_array weights})
  (output:A.larray bf16 n{GA.is_global_array output})
  (#fraction:perm) (#storage:erased (bfseq weight_count)) (#packed:erased (S.vector weight_count))
  preserves cpu ** on gpu_loc (weights |-> Frac fraction storage)
  requires on gpu_loc (A.pts_to_uninit output n)
  requires pure (represents weight_count storage packed)
  ensures exists* (result:lseq bf16 n). on gpu_loc (output |-> result) **
    pure (bf_rep n result (fun i -> vector_at #weight_count packed (offset + i)))
{
  let count = SZ.uint32_to_sizet 18352640ul;
  gather 1sz n count weights offset 0sz 1sz output #fraction #storage;
}

inline_for_extraction noextract
let rotated_matrix (heads:pos) (position:nat) (weight:S.vector 256)
  (inputs:S.matrix heads 256) : S.matrix heads 256 =
  fun h -> S.rotate_partial position (S.rms_norm 256 weight (head_vector heads inputs h))

inline_for_extraction noextract
let rotation_congr (position:nat) (f g:S.vector 256) (d:S.index 256)
  : Lemma (requires forall (i:S.index 256). f i == g i)
          (ensures S.rotate_partial position f d == S.rotate_partial position g d) = ()

inline_for_extraction noextract
let rotated_normalized_point
  (heads:pos) (position:nat) (weight:S.vector 256) (inputs:S.matrix heads 256)
  (i:nat{i < heads * 256})
  : Lemma (
      rotated_heads heads position (normalized_heads heads weight inputs) i ==
      flat_heads heads (rotated_matrix heads position weight inputs) i) =
  let h = i / 256 in
  let f:S.vector 256 = fun d -> normalized_heads heads weight inputs (256 * h + d) in
  let g = S.rms_norm 256 weight (head_vector heads inputs h) in
  rotation_congr position f g (i % 256)

inline_for_extraction noextract
let rotated_normalized_rep
  (heads:pos) (position:nat) (weight:S.vector 256) (inputs:S.matrix heads 256)
  (storage:lseq bf16 (heads * 256))
  : Lemma
      (requires bf_rep (heads * 256) storage (rotated_heads heads position (normalized_heads heads weight inputs)))
      (ensures bf_rep (heads * 256) storage (flat_heads heads (rotated_matrix heads position weight inputs))) =
  let point (i:S.index (heads * 256))
    : Lemma (
        rotated_heads heads position (normalized_heads heads weight inputs) i ==
        flat_heads heads (rotated_matrix heads position weight inputs) i) =
    rotated_normalized_point heads position weight inputs i
  in
  FStar.Classical.forall_intro #(S.index (heads * 256)) #(fun i ->
    rotated_heads heads position (normalized_heads heads weight inputs) i ==
    flat_heads heads (rotated_matrix heads position weight inputs) i) point;
  bf_rep_congr (heads * 256) storage
    (rotated_heads heads position (normalized_heads heads weight inputs))
    (flat_heads heads (rotated_matrix heads position weight inputs))

inline_for_extraction noextract
let reshape_value_rep (storage:lseq bf16 512) (logical:S.vector 512)
  : Lemma (requires bf_rep 512 storage logical)
          (ensures bf_rep 512 storage (flat_heads 2 (S.reshape_kv logical))) = ()

inline_for_extraction noextract
let query_projection_rep (storage:lseq bf16 4096) (logical:S.vector 4096)
  : Lemma (requires bf_rep 4096 storage logical)
          (ensures matrix_rep 8 256 4096 0 512 1 storage (S.split_queries logical)) = ()

inline_for_extraction noextract
let key_projection_rep (storage:lseq bf16 512) (logical:S.vector 512)
  : Lemma (requires bf_rep 512 storage logical)
          (ensures matrix_rep 2 256 512 0 256 1 storage (S.reshape_kv logical)) = ()

inline_for_extraction noextract
let gate_projection_rep (source:lseq bf16 4096) (logical:S.vector 4096) (copied:lseq bf16 2048)
  : Lemma
      (requires bf_rep 4096 source logical /\ gathered 8 256 4096 256 512 1 source copied)
      (ensures bf_rep 2048 copied (flat_heads 8 (S.split_gates logical))) = ()

fn append_state
  (#logical_position:erased nat)
  (position:sz{position <= 646 /\ SZ.v position == FStar.Ghost.reveal #nat logical_position})
  (old:A.array bf16)
  (current:A.larray bf16 512{GA.is_global_array current})
  (output:A.larray bf16 (next_count logical_position){GA.is_global_array output})
  (#fo #fc:perm)
  (#so:erased (bfseq (prefix_count logical_position))) (#sc:erased (bfseq 512))
  preserves cpu ** on gpu_loc (current |-> Frac fc sc)
  preserves (if FStar.Ghost.reveal #nat logical_position = 0 then pure (old == A.null)
    else pure (GA.is_global_array old /\ A.length old == prefix_count logical_position) **
      on gpu_loc (old |-> Frac fo (FStar.Ghost.reveal #(bfseq (prefix_count logical_position)) so)))
  requires on gpu_loc (A.pts_to_uninit output (next_count logical_position))
  ensures exists* (result:bfseq (next_count logical_position)).
    on gpu_loc (output |-> result) ** pure (appended (prefix_count logical_position) so sc result)
{
  if (position = 0sz) {
    rewrite each (FStar.Ghost.reveal #nat logical_position = 0) as true;
    gather 1sz 512sz 512sz current 0sz 0sz 1sz output #fc #sc;
    rewrite (pure (old == A.null)) as
      (if FStar.Ghost.reveal #nat logical_position = 0 then pure (old == A.null)
       else pure (GA.is_global_array old /\ A.length old == prefix_count logical_position) **
         on gpu_loc (old |-> Frac fo (FStar.Ghost.reveal #(bfseq (prefix_count logical_position)) so)));
  } else {
    assert pure (SZ.v position > 0);
    rewrite each (FStar.Ghost.reveal #nat logical_position = 0) as false;
    assert pure (GA.is_global_array old /\ A.length old == prefix_count logical_position);
    let active : (p:A.larray bf16 (prefix_count logical_position){GA.is_global_array p}) = old;
    assert rewrites_to active old;
    append (position *^ 512sz) active current output #fo #fc #so #sc;
    rewrite (pure (GA.is_global_array old /\ A.length old == prefix_count logical_position) **
      on gpu_loc (old |-> Frac fo (FStar.Ghost.reveal #(bfseq (prefix_count logical_position)) so))) as
      (if FStar.Ghost.reveal #nat logical_position = 0 then pure (old == A.null)
       else pure (GA.is_global_array old /\ A.length old == prefix_count logical_position) **
         on gpu_loc (old |-> Frac fo (FStar.Ghost.reveal #(bfseq (prefix_count logical_position)) so)));
  }
}

inline_for_extraction noextract
let appended_history_rep
  (position:nat) (old_keys old_values:S.vector (prefix_count position))
  (sk sv:bfseq (prefix_count position))
  (key value:S.matrix 2 256) (ck cv:bfseq 512)
  (nk nv:bfseq (next_count position))
  : Lemma
      (requires
        represents (prefix_count position) sk old_keys /\
        represents (prefix_count position) sv old_values /\
        bf_rep 512 ck (flat_heads 2 key) /\ bf_rep 512 cv (flat_heads 2 value) /\
        appended (prefix_count position) sk ck nk /\ appended (prefix_count position) sv cv nv)
      (ensures history_rep (position + 1) nk nv
        (S.append_kv position (old_history_view position old_keys old_values) key value)) =
  let next = S.append_kv position (old_history_view position old_keys old_values) key value in
  let point (h:S.index 2) (t:S.index (position + 1)) (d:S.index 256)
    : Lemma (
        v_approximates (Seq.index nk (512 * t + 256 * h + d)) (next.keys h t d) /\
        v_approximates (Seq.index nv (512 * t + 256 * h + d)) (next.values h t d)) =
    if t < position then (
      assert (position > 0);
      let old = history_view position old_keys old_values in
      S.append_preserves_prefix old key value h t d;
      assert (Seq.index nk (512 * t + 256 * h + d) == Seq.index sk (512 * t + 256 * h + d));
      assert (Seq.index nv (512 * t + 256 * h + d) == Seq.index sv (512 * t + 256 * h + d))
    ) else (
      assert (t == position);
      row_index 2 256 h d;
      S.append_includes_current position (old_history_view position old_keys old_values) key value h d;
      assert (Seq.index nk (512 * t + 256 * h + d) == Seq.index ck (256 * h + d));
      assert (Seq.index nv (512 * t + 256 * h + d) == Seq.index cv (256 * h + d))
    )
  in
  FStar.Classical.forall_intro_3 #(S.index 2) #(fun _ -> S.index (position + 1))
    #(fun _ _ -> S.index 256)
    #(fun h t d ->
      v_approximates (Seq.index nk (512 * t + 256 * h + d)) (next.keys h t d) /\
      v_approximates (Seq.index nv (512 * t + 256 * h + d)) (next.values h t d)) point

inline_for_extraction noextract
let kv_flat_index (n:pos) (i:nat{i < n * 512})
  : Lemma (
      i / 512 < n /\ (i % 512) / 256 < 2 /\ i % 256 < 256 /\
      512 * (i / 512) + 256 * ((i % 512) / 256) + i % 256 == i) = ()

inline_for_extraction noextract
let history_flat_rep
  (n:pos) (keys values:bfseq (prefix_count n)) (history:S.kv_history{history.length == n})
  : Lemma
      (requires history_rep n keys values history)
      (ensures represents (prefix_count n) keys (flatten_kv n history.keys) /\
               represents (prefix_count n) values (flatten_kv n history.values)) =
  let point (i:S.index (prefix_count n))
    : Lemma (
        v_approximates (Seq.index keys i) (flatten_kv n history.keys i) /\
        v_approximates (Seq.index values i) (flatten_kv n history.values i)) =
    kv_flat_index n i
  in
  FStar.Classical.forall_intro #(S.index (prefix_count n))
    #(fun i -> v_approximates (Seq.index keys i) (flatten_kv n history.keys i) /\
               v_approximates (Seq.index values i) (flatten_kv n history.values i)) point

noextract
let shared_decoder_cache
  (call:S.invocation) (w:S.layer_weights) (x:S.vector 1024)
  (before:S.hybrid_cache{S.domain call before})
  : Lemma (
      (S.full_attention_decoder call w x before).cache ==
      S.replace_kv before call.layer
        (S.append_kv call.position (before.kv call.layer)
          (rotated_matrix 2 call.position w.key_norm
            (S.reshape_kv (S.linear 512 1024 w.key_proj (S.rms_norm 1024 w.input_norm x))))
          (S.reshape_kv (S.linear 512 1024 w.value_proj (S.rms_norm 1024 w.input_norm x))))) =
  assert (
      (S.full_attention_decoder call w x before).cache ==
      S.replace_kv before call.layer
        (S.append_kv call.position (before.kv call.layer)
          (rotated_matrix 2 call.position w.key_norm
            (S.reshape_kv (S.linear 512 1024 w.key_proj (S.rms_norm 1024 w.input_norm x))))
          (S.reshape_kv (S.linear 512 1024 w.value_proj (S.rms_norm 1024 w.input_norm x)))))
    by (
      Proof.norm [delta_only [
        "KPlay.SharedSpec.full_attention_decoder";
        "KPlay.Functional.rotated_matrix"; "KPlay.Functional.head_vector"
      ]];
      Proof.trefl ()
    )

noextract
let shared_decoder_hidden
  (call:S.invocation) (w:S.layer_weights) (x:S.vector 1024)
  (before:S.hybrid_cache{S.domain call before})
  : Lemma (
      (let u = S.rms_norm 1024 w.input_norm x in
       let qg = S.linear 4096 1024 w.query_gate_proj u in
       let q = rotated_matrix 8 call.position w.query_norm (S.split_queries qg) in
       let k = rotated_matrix 2 call.position w.key_norm (S.reshape_kv (S.linear 512 1024 w.key_proj u)) in
       let v = S.reshape_kv (S.linear 512 1024 w.value_proj u) in
       let next = S.append_kv call.position (before.kv call.layer) k v in
       let gated = activation_vector 2048 false (flat_heads 8 (S.split_gates qg))
         (flat_heads 8 (S.attention call.position q next)) in
       let mixed = S.linear 1024 2048 w.output_proj gated in
       let residual = S.vector_add 1024 x mixed in
       let mlp_input = S.rms_norm 1024 w.post_attention_norm residual in
       let gate = S.linear 3584 1024 w.mlp_gate_proj mlp_input in
       let up = S.linear 3584 1024 w.mlp_up_proj mlp_input in
       let product = activation_vector 3584 true gate up in
       S.vector_add 1024 residual (S.linear 1024 3584 w.mlp_down_proj product))
      == (S.full_attention_decoder call w x before).hidden) =
  assert (
      (let u = S.rms_norm 1024 w.input_norm x in
       let qg = S.linear 4096 1024 w.query_gate_proj u in
       let q = rotated_matrix 8 call.position w.query_norm (S.split_queries qg) in
       let k = rotated_matrix 2 call.position w.key_norm (S.reshape_kv (S.linear 512 1024 w.key_proj u)) in
       let v = S.reshape_kv (S.linear 512 1024 w.value_proj u) in
       let next = S.append_kv call.position (before.kv call.layer) k v in
       let gated = activation_vector 2048 false (flat_heads 8 (S.split_gates qg))
         (flat_heads 8 (S.attention call.position q next)) in
       let mixed = S.linear 1024 2048 w.output_proj gated in
       let residual = S.vector_add 1024 x mixed in
       let mlp_input = S.rms_norm 1024 w.post_attention_norm residual in
       let gate = S.linear 3584 1024 w.mlp_gate_proj mlp_input in
       let up = S.linear 3584 1024 w.mlp_up_proj mlp_input in
       let product = activation_vector 3584 true gate up in
       S.vector_add 1024 residual (S.linear 1024 3584 w.mlp_down_proj product))
      == (S.full_attention_decoder call w x before).hidden)
    by (
      Proof.norm [nbe; unmeta; unascribe; delta_only [
        "KPlay.SharedSpec.full_attention_decoder";
        "KPlay.Functional.rotated_matrix"; "KPlay.Functional.head_vector";
        "KPlay.Functional.activation_vector"; "KPlay.Functional.activated_product";
        "KPlay.Functional.flat_heads"
      ]];
      Proof.trefl ()
    )

noextract
let final_refinement
  (input:logical_inputs)
  (sk sv:bfseq (prefix_count input.call.position))
  (ck cv:bfseq 512)
  (sy:bfseq 1024) (nk nv:bfseq (next_count input.call.position))
  : Lemma
      (requires (
        let w = weights_view input.packed_weights in
        let u = S.rms_norm 1024 w.input_norm input.input_hidden in
        let qg = S.linear 4096 1024 w.query_gate_proj u in
        let key = rotated_matrix 2 input.call.position w.key_norm (S.reshape_kv (S.linear 512 1024 w.key_proj u)) in
        let value = S.reshape_kv (S.linear 512 1024 w.value_proj u) in
        represents (prefix_count input.call.position) sk input.old_keys /\
        represents (prefix_count input.call.position) sv input.old_values /\
        bf_rep 512 ck (flat_heads 2 key) /\ bf_rep 512 cv (flat_heads 2 value) /\
        appended (prefix_count input.call.position) sk ck nk /\
        appended (prefix_count input.call.position) sv cv nv /\
        bf_rep 1024 sy (S.full_attention_decoder input.call w input.input_hidden (logical_before input)).hidden))
      (ensures semantic_postcondition input sk sv sy nk nv) =
  let w = weights_view input.packed_weights in
  let u = S.rms_norm 1024 w.input_norm input.input_hidden in
  let key = rotated_matrix 2 input.call.position w.key_norm (S.reshape_kv (S.linear 512 1024 w.key_proj u)) in
  let value = S.reshape_kv (S.linear 512 1024 w.value_proj u) in
  let next = S.append_kv input.call.position (old_history_view input.call.position input.old_keys input.old_values) key value in
  let expected = S.full_attention_decoder input.call w input.input_hidden (logical_before input) in
  appended_history_rep input.call.position input.old_keys input.old_values sk sv key value ck cv nk nv;
  history_flat_rep (input.call.position + 1) nk nv next;
  result_cache_refinement input.call input.frame input.old_keys input.old_values key value expected.hidden;
  shared_decoder_cache input.call w input.input_hidden (logical_before input);
  assert ((logical_before input).kv input.call.layer ==
    old_history_view input.call.position input.old_keys input.old_values);
  assert (expected.cache == S.replace_kv (logical_before input) input.call.layer next);
  assert (S.postcondition input.call w input.input_hidden (logical_before input)
    (result_view input.call input.frame expected.hidden
      (flatten_kv (input.call.position + 1) next.keys)
      (flatten_kv (input.call.position + 1) next.values)));
  assert (exists (logical_hidden:S.vector 1024)
                 (logical_keys:S.vector (next_count input.call.position))
                 (logical_values:S.vector (next_count input.call.position)).
    represents 1024 sy logical_hidden /\
    represents (next_count input.call.position) nk logical_keys /\
    represents (next_count input.call.position) nv logical_values /\
    S.postcondition input.call w input.input_hidden (logical_before input)
      (result_view input.call input.frame logical_hidden logical_keys logical_values))

fn kplay_run
  (#input:erased logical_inputs)
  (#sw:erased (bfseq weight_count))
  (#sx:erased (bfseq hidden_width))
  (#sr:erased (bfseq rope_width))
  (#sk:erased (bfseq (prefix_count input.call.position)))
  (#sv:erased (bfseq (prefix_count input.call.position)))
  (#fw #fx #fr #fk #fv:perm)
  (weights:A.larray BF.t weight_count{GA.is_global_array weights})
  (hidden:A.larray BF.t hidden_width{GA.is_global_array hidden})
  (rope:A.larray BF.t rope_width{GA.is_global_array rope})
  (old_keys:A.array BF.t)
  (old_values:A.array BF.t)
  (position:I64.t{I64.v position == input.call.position})
  (output:A.larray BF.t hidden_width{GA.is_global_array output})
  (new_keys:A.larray BF.t (next_count input.call.position){GA.is_global_array new_keys})
  (new_values:A.larray BF.t (next_count input.call.position){GA.is_global_array new_values})
  preserves cpu **
    on gpu_loc (weights |-> Frac fw sw) **
    on gpu_loc (hidden |-> Frac fx sx) **
    on gpu_loc (rope |-> Frac fr sr) **
    prefix_ownership input.call.position old_keys fk sk **
    prefix_ownership input.call.position old_values fv sv
  requires pure (input_representation input sw sx sr sk sv)
  requires on gpu_loc (
    A.pts_to_uninit output hidden_width **
    A.pts_to_uninit new_keys (next_count input.call.position) **
    A.pts_to_uninit new_values (next_count input.call.position))
  ensures exists* (sy:bfseq hidden_width)
                  (nk:bfseq (next_count input.call.position))
                  (nv:bfseq (next_count input.call.position)).
    on gpu_loc (output |-> sy ** new_keys |-> nk ** new_values |-> nv) **
    pure (semantic_postcondition input sk sv sy nk nv)
{
  open KPlay.SharedSpec;
  S.full_profile_position_bound input.call;
  physical_counts input.call.position;
  let p = SZ.uint32_to_sizet (IC.int64_to_uint32 position);
  assert pure (SZ.v p == input.call.position /\ p <= 646);
  all_weight_matrices sw input.packed_weights;
  physical_rep 1024 sx input.input_hidden;
  rope_input_rep input.call.position sr;

  let w : erased S.layer_weights = hide (weights_view input.packed_weights);
  let u : erased (S.vector 1024) = hide (S.rms_norm 1024 w.input_norm input.input_hidden);
  let qg_model : erased (S.vector 4096) = hide (S.linear 4096 1024 w.query_gate_proj u);
  let key_model : erased (S.vector 512) = hide (S.linear 512 1024 w.key_proj u);
  let value_model : erased (S.vector 512) = hide (S.linear 512 1024 w.value_proj u);
  let q_model : erased (S.matrix 8 256) = hide (rotated_matrix 8 input.call.position w.query_norm (S.split_queries qg_model));
  let k_model : erased (S.matrix 2 256) = hide (rotated_matrix 2 input.call.position w.key_norm (S.reshape_kv key_model));
  let v_model : erased (S.matrix 2 256) = hide (S.reshape_kv value_model);
  let history : (h:erased S.kv_history{(FStar.Ghost.reveal #S.kv_history h).length == input.call.position + 1}) =
    hide (S.append_kv input.call.position
      (old_history_view input.call.position input.old_keys input.old_values) k_model v_model);
  let context_model : erased (S.vector 2048) = hide (flat_heads 8 (S.attention input.call.position q_model history));
  let gates_model : erased (S.vector 2048) = hide (flat_heads 8 (S.split_gates qg_model));
  let gated_model : erased (S.vector 2048) = hide (activation_vector 2048 false gates_model context_model);
  let mixed_model : erased (S.vector 1024) = hide (S.linear 1024 2048 w.output_proj gated_model);
  let residual_model : erased (S.vector 1024) = hide (S.vector_add 1024 input.input_hidden mixed_model);
  let mlp_input_model : erased (S.vector 1024) = hide (S.rms_norm 1024 w.post_attention_norm residual_model);
  let gate_model : erased (S.vector 3584) = hide (S.linear 3584 1024 w.mlp_gate_proj mlp_input_model);
  let up_model : erased (S.vector 3584) = hide (S.linear 3584 1024 w.mlp_up_proj mlp_input_model);
  let product_model : erased (S.vector 3584) = hide (activation_vector 3584 true gate_model up_model);
  let down_model : erased (S.vector 1024) = hide (S.linear 1024 3584 w.mlp_down_proj product_model);

  let count = SZ.uint32_to_sizet 18352640ul;
  let qg_offset = SZ.uint32_to_sizet 11012096ul;
  let key_offset = SZ.uint32_to_sizet 15206400ul;
  let value_offset = SZ.uint32_to_sizet 15730688ul;
  let query_norm_offset = SZ.uint32_to_sizet 16254976ul;
  let key_norm_offset = SZ.uint32_to_sizet 16255232ul;
  let output_offset = SZ.uint32_to_sizet 16255488ul;
  let up_offset = SZ.uint32_to_sizet 3672064ul;
  let down_offset = SZ.uint32_to_sizet 7342080ul;

  let input_weight = allocate_bf 1024sz;
  let post_weight = allocate_bf 1024sz;
  let query_weight = allocate_bf 256sz;
  let key_weight = allocate_bf 256sz;
  load_weight_vector 1024sz 0sz weights input_weight #fw #sw #input.packed_weights;
  load_weight_vector 1024sz 1024sz weights post_weight #fw #sw #input.packed_weights;
  load_weight_vector 256sz query_norm_offset weights query_weight #fw #sw #input.packed_weights;
  load_weight_vector 256sz key_norm_offset weights key_weight #fw #sw #input.packed_weights;
  with siw. assert on gpu_loc (input_weight |-> siw);
  with spw. assert on gpu_loc (post_weight |-> spw);
  with sqw. assert on gpu_loc (query_weight |-> sqw);
  with skw. assert on gpu_loc (key_weight |-> skw);
  assert pure (
    bf_rep 1024 siw w.input_norm /\ bf_rep 1024 spw w.post_attention_norm /\
    bf_rep 256 sqw w.query_norm /\ bf_rep 256 skw w.key_norm);

  let normalized = allocate_bf 1024sz;
  execute_rms 1024sz hidden input_weight normalized #fx #1.0R #sx #siw #input.input_hidden #w.input_norm;
  with sn. assert on gpu_loc (normalized |-> sn);
  let projected = allocate_bf 4096sz;
  let projected_key = allocate_bf 512sz;
  let projected_value = allocate_bf 512sz;
  matvec 4096sz 1024sz count weights qg_offset 1024sz 1sz normalized projected false
    #fw #1.0R #sw #sn #w.query_gate_proj #u;
  matvec 512sz 1024sz count weights key_offset 1024sz 1sz normalized projected_key false
    #fw #1.0R #sw #sn #w.key_proj #u;
  matvec 512sz 1024sz count weights value_offset 1024sz 1sz normalized projected_value false
    #fw #1.0R #sw #sn #w.value_proj #u;
  with sqg. assert on gpu_loc (projected |-> sqg);
  with skp. assert on gpu_loc (projected_key |-> skp);
  with scv. assert on gpu_loc (projected_value |-> scv);
  query_projection_rep sqg qg_model;
  key_projection_rep skp key_model;
  reshape_value_rep scv value_model;

  let normalized_query = GA.gpu_array_alloc #bf16 2048sz;
  let normalized_key = GA.gpu_array_alloc #bf16 512sz;
  with nq0. assert on gpu_loc (normalized_query |-> nq0);
  with nk0. assert on gpu_loc (normalized_key |-> nk0);
  let initial_query : erased (lseq bf16 2048) = hide (nq0 <: lseq bf16 2048);
  let initial_key : erased (lseq bf16 512) = hide (nk0 <: lseq bf16 512);
  norm_heads 8sz 4096sz projected 0sz 512sz query_weight normalized_query
    #1.0R #1.0R #sqg #sqw #initial_query #(S.split_queries qg_model) #w.query_norm;
  norm_heads 2sz 512sz projected_key 0sz 256sz key_weight normalized_key
    #1.0R #1.0R #skp #skw #initial_key #(S.reshape_kv key_model) #w.key_norm;
  with snq. assert on gpu_loc (normalized_query |-> snq);
  with snk. assert on gpu_loc (normalized_key |-> snk);
  let query = allocate_bf 2048sz;
  let current_key = allocate_bf 512sz;
  rotate 8sz normalized_query query rope #1.0R #fr #snq #sr #input.call.position
    #(normalized_heads 8 w.query_norm (S.split_queries qg_model));
  rotate 2sz normalized_key current_key rope #1.0R #fr #snk #sr #input.call.position
    #(normalized_heads 2 w.key_norm (S.reshape_kv key_model));
  with sq. assert on gpu_loc (query |-> sq);
  with sck. assert on gpu_loc (current_key |-> sck);
  rotated_normalized_rep 8 input.call.position w.query_norm (S.split_queries qg_model) sq;
  rotated_normalized_rep 2 input.call.position w.key_norm (S.reshape_kv key_model) sck;

  unfold prefix_ownership input.call.position old_keys fk sk;
  unfold prefix_ownership input.call.position old_values fv sv;
  append_state #input.call.position p old_keys current_key new_keys #fk #1.0R #sk #sck;
  append_state #input.call.position p old_values projected_value new_values #fv #1.0R #sv #scv;
  fold prefix_ownership input.call.position old_keys fk sk;
  fold prefix_ownership input.call.position old_values fv sv;
  with nk. assert on gpu_loc (new_keys |-> nk);
  with nv. assert on gpu_loc (new_values |-> nv);
  appended_history_rep input.call.position input.old_keys input.old_values sk sv
    k_model v_model sck scv nk nv;
  assert pure (history_rep (input.call.position + 1) nk nv history);
  assert pure (history_rep (SZ.v p + 1) nk nv history);

  let context = GA.gpu_array_alloc #bf16 2048sz;
  with context0. assert on gpu_loc (context |-> context0);
  let initial_context : erased (lseq bf16 2048) = hide (context0 <: lseq bf16 2048);
  execute_attention p query new_keys new_values context #1.0R #1.0R #1.0R #sq #nk #nv #initial_context #q_model #history;
  with context_values. assert on gpu_loc (context |-> context_values);
  let gates = allocate_bf 2048sz;
  gather 8sz 256sz 4096sz projected 256sz 512sz 1sz gates #1.0R #sqg;
  with gate_values. assert on gpu_loc (gates |-> gate_values);
  gate_projection_rep sqg qg_model gate_values;
  let gated = allocate_bf 2048sz;
  activation 2048sz false gates context gated #1.0R #1.0R #gate_values #context_values #gates_model #context_model;
  with sga. assert on gpu_loc (gated |-> sga);
  let mixed = allocate_bf 1024sz;
  matvec 1024sz 2048sz count weights output_offset 2048sz 1sz gated mixed false
    #fw #1.0R #sw #sga #w.output_proj #gated_model;
  with sm. assert on gpu_loc (mixed |-> sm);
  let after_attention = allocate_bf 1024sz;
  residual hidden mixed after_attention #fx #1.0R #sx #sm #input.input_hidden #mixed_model;
  with sa. assert on gpu_loc (after_attention |-> sa);

  let mlp_input = allocate_bf 1024sz;
  execute_rms 1024sz after_attention post_weight mlp_input #1.0R #1.0R #sa #spw #residual_model #w.post_attention_norm;
  with sxm. assert on gpu_loc (mlp_input |-> sxm);
  let mlp_gate = allocate_bf 3584sz;
  let mlp_up = allocate_bf 3584sz;
  matvec 3584sz 1024sz count weights 2048sz 1024sz 1sz mlp_input mlp_gate false
    #fw #1.0R #sw #sxm #w.mlp_gate_proj #mlp_input_model;
  matvec 3584sz 1024sz count weights up_offset 1024sz 1sz mlp_input mlp_up false
    #fw #1.0R #sw #sxm #w.mlp_up_proj #mlp_input_model;
  with sg. assert on gpu_loc (mlp_gate |-> sg);
  with su. assert on gpu_loc (mlp_up |-> su);
  let mlp_product = allocate_bf 3584sz;
  activation 3584sz true mlp_gate mlp_up mlp_product #1.0R #1.0R #sg #su #gate_model #up_model;
  with sp. assert on gpu_loc (mlp_product |-> sp);
  let mlp_down = allocate_bf 1024sz;
  matvec 1024sz 3584sz count weights down_offset 3584sz 1sz mlp_product mlp_down true
    #fw #1.0R #sw #sp #w.mlp_down_proj #product_model;
  with sd. assert on gpu_loc (mlp_down |-> sd);
  residual after_attention mlp_down output #1.0R #1.0R #sa #sd #residual_model #down_model;
  with sy. assert on gpu_loc (output |-> sy);

  shared_decoder_hidden input.call w input.input_hidden (logical_before input);
  assert pure ((logical_before input).kv input.call.layer ==
    old_history_view input.call.position input.old_keys input.old_values);
  assert pure (bf_rep 1024 sy
    (S.full_attention_decoder input.call w input.input_hidden (logical_before input)).hidden);
  final_refinement input sk sv sck scv sy nk nv;

  GA.gpu_array_free input_weight;
  GA.gpu_array_free post_weight;
  GA.gpu_array_free query_weight;
  GA.gpu_array_free key_weight;
  GA.gpu_array_free normalized;
  GA.gpu_array_free projected;
  GA.gpu_array_free projected_key;
  GA.gpu_array_free projected_value;
  GA.gpu_array_free normalized_query;
  GA.gpu_array_free normalized_key;
  GA.gpu_array_free query;
  GA.gpu_array_free current_key;
  GA.gpu_array_free context;
  GA.gpu_array_free gates;
  GA.gpu_array_free gated;
  GA.gpu_array_free mixed;
  GA.gpu_array_free after_attention;
  GA.gpu_array_free mlp_input;
  GA.gpu_array_free mlp_gate;
  GA.gpu_array_free mlp_up;
  GA.gpu_array_free mlp_product;
  GA.gpu_array_free mlp_down;
}
