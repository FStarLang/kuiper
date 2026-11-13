module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Float.Casts
module S = FStar.Seq
module R = FStar.Real
module Sqrt = FStar.Math.Sqrt
module U32 = FStar.UInt32
module U64 = FStar.UInt64
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module Shared = KPlay.SharedSpec
module Sum = Kuiper.Sum

#set-options "--z3rlimit 80"

[@@CPrologue "__device__"]
let bounded_advance (i limit amount : sz)
  : Pure sz
      (requires i <= limit)
      (ensures fun result -> SZ.v result == min (SZ.v limit) (i + amount))
  = if limit -^ i <=^ amount then limit else i +^ amount

inline_for_extraction noextract
let rec masked_squares
  (x : seq real) (start : nat)
  (count : nat{start + count <= S.length x})
  (p : nat -> bool)
  : GTot real (decreases count)
  = if count = 0 then 0.0R
    else masked_squares x start (count - 1) p +.
      (if p (count - 1)
       then S.index x (start + count - 1) *. S.index x (start + count - 1)
       else 0.0R)

inline_for_extraction noextract
let rec masked_none
  (x : seq real) (start : nat)
  (count : nat{start + count <= S.length x}) (p : nat -> bool)
  : Lemma
      (requires forall i. i < count ==> not (p i))
      (ensures masked_squares x start count p == 0.0R)
      (decreases count)
  = if count > 0 then masked_none x start (count - 1) p

inline_for_extraction noextract
let rec masked_extend
  (x : seq real) (start : nat)
  (lo hi : nat{lo <= hi /\ start + hi <= S.length x})
  (p : nat -> bool)
  : Lemma
      (requires forall i. lo <= i /\ i < hi ==> not (p i))
      (ensures masked_squares x start hi p == masked_squares x start lo p)
      (decreases hi - lo)
  = if hi > lo then masked_extend x start lo (hi - 1) p

inline_for_extraction noextract
let rec masked_partition
  (x : seq real) (start : nat)
  (count : nat{start + count <= S.length x})
  (p q r : nat -> bool)
  : Lemma
      (requires forall i. i < count ==>
        (p i == (q i || r i)) /\ not (q i && r i))
      (ensures masked_squares x start count p ==
        masked_squares x start count q +. masked_squares x start count r)
      (decreases count)
  = if count > 0 then masked_partition x start (count - 1) p q r

inline_for_extraction noextract
let rec masked_equal
  (x : seq real) (start : nat)
  (count : nat{start + count <= S.length x})
  (p q : nat -> bool)
  : Lemma
      (requires forall i. i < count ==> p i == q i)
      (ensures masked_squares x start count p == masked_squares x start count q)
      (decreases count)
  = if count > 0 then masked_equal x start (count - 1) p q

inline_for_extraction noextract
let rec masked_full
  (x : seq real) (start : nat)
  (count : nat{start + count <= S.length x})
  : Lemma
      (ensures masked_squares x start count (fun _ -> true) ==
        sum_squares x start count)
      (decreases count)
  = if count > 0 then masked_full x start (count - 1)

inline_for_extraction noextract
let thread_mask (t i : nat) : bool = i % 256 = t

inline_for_extraction noextract
let leaf_zero
  (#n : nat) (x : lseq bf16 n)
  (start width : nat{start + width <= n})
  (t : nat{t < 256}) (j : nat{j == min t width})
  : Lemma (forall (rx : lseq real n). x %~ rx ==>
      (zero #f32) %~ masked_squares rx start j (thread_mask t))
  = introduce forall (rx : lseq real n).
      x %~ rx ==> (zero #f32) %~ masked_squares rx start j (thread_mask t)
    with (if x %~ rx then masked_none rx start j (thread_mask t))

inline_for_extraction noextract
let leaf_step
  (#n : nat) (x : lseq bf16 n)
  (start width : nat{start + width <= n})
  (t : nat{t < 256})
  (j : nat{j < width /\ j % 256 == t})
  (next : nat{next == min width (j + 256)})
  (old result : f32)
  : Lemma
      (requires
        (forall (rx : lseq real n). x %~ rx ==>
          old %~ masked_squares rx start j (thread_mask t)) /\
        (forall (a b c : real).
          (fcast (S.index x (start + j)) <: f32) %~ a /\
          (fcast (S.index x (start + j)) <: f32) %~ b /\ old %~ c ==>
          result %~ (a *. b +. c)))
      (ensures forall (rx : lseq real n). x %~ rx ==>
          result %~ masked_squares rx start next (thread_mask t))
  = introduce forall (rx : lseq real n).
      x %~ rx ==> result %~ masked_squares rx start next (thread_mask t)
    with (
      if x %~ rx then (
        masked_extend rx start (j + 1) next (thread_mask t);
        assert (old %~ masked_squares rx start j (thread_mask t));
        let v : f32 = fcast (S.index x (start + j)) in
        assert (S.index x (start + j) %~ S.index rx (start + j));
        fcast_approx #bf16 #f32 (S.index x (start + j)) (S.index rx (start + j));
        assert (v %~ S.index rx (start + j));
        assert (result %~ (S.index rx (start + j) *. S.index rx (start + j) +.
          masked_squares rx start j (thread_mask t)));
        assert (masked_squares rx start (j + 1) (thread_mask t) ==
          masked_squares rx start j (thread_mask t) +.
            S.index rx (start + j) *. S.index rx (start + j));
        assert (result %~ masked_squares rx start next (thread_mask t))
      ))

[@@CPrologue "__device__"]
fn rms_leaf
  (#n : erased nat)
  (input : larray bf16 n)
  (start : sz) (width : szp{start + width <= n})
  (t : szlt 256)
  (#x : erased (lseq bf16 n)) (#f : perm)
  preserves gpu ** input |-> Frac f x
  returns result : f32
  ensures pure (forall (rx : lseq real n). x %~ rx ==>
    result %~ masked_squares rx start width (thread_mask t))
{
  let initial = SZ.smin t width;
  leaf_zero x start width t initial;
  let mut j = initial;
  let mut accumulator : f32 = zero;
  while (!j <^ width)
    invariant exists* (k : sz{k <= width /\ (k == width \/ k % 256 == t)})
                      (a : f32).
      j |-> k ** accumulator |-> a **
      pure (forall (rx : lseq real n). x %~ rx ==>
        a %~ masked_squares rx start k (thread_mask t))
    decreases (width - !j)
  {
    let k = !j;
    let v : f32 = fcast input.(start +^ k);
    let old = !accumulator;
    let result = F32.fma_rn_ftz v v old;
    let next = if (width -^ k <=^ 256sz) {
      (width <: FStar.SizeT.t)
    } else {
      (k +^ 256sz <: FStar.SizeT.t)
    };
    leaf_step x start width t k next old result;
    accumulator := result;
    j := bounded_advance !j width 256sz;
  };
  !accumulator
}

inline_for_extraction noextract
let small_pow2 (s : nat) : prop =
  s == 1 \/ s == 2 \/ s == 4 \/ s == 8 \/ s == 16 \/ s == 32

inline_for_extraction noextract
let lane_mask (warp lane : nat) (step : pos) (i : nat) : bool =
  i % 256 / 32 = warp && i % 32 % step = lane

inline_for_extraction noextract
let warp_mask (warp : nat) (step : pos) (i : nat) : bool =
  i % 256 / 32 % step = warp

inline_for_extraction noextract
let lane_partition
  (warp : nat{warp < 8})
  (step : pos{small_pow2 step /\ step < 32})
  (lane : nat{lane < step}) (i : nat)
  : Lemma (
    lane_mask warp lane step i ==
      (lane_mask warp lane (2 * step) i ||
       lane_mask warp (lane + step) (2 * step) i) /\
    not (lane_mask warp lane (2 * step) i &&
         lane_mask warp (lane + step) (2 * step) i))
  = if step = 1 then ()
    else if step = 2 then ()
    else if step = 4 then ()
    else if step = 8 then ()
    else ()

inline_for_extraction noextract
let warp_partition
  (step : pos{small_pow2 step /\ step < 8})
  (warp : nat{warp < step}) (i : nat)
  : Lemma (
    warp_mask warp step i ==
      (warp_mask warp (2 * step) i || warp_mask (warp + step) (2 * step) i) /\
    not (warp_mask warp (2 * step) i && warp_mask (warp + step) (2 * step) i))
  = if step = 1 then () else if step = 2 then () else ()

inline_for_extraction noextract
let approx_mask_add
  (#n : nat) (x : lseq bf16 n)
  (start width : nat{start + width <= n})
  (p q r : nat -> bool)
  (a b c : f32)
  : Lemma
      (requires
        (forall i. i < width ==> (p i == (q i || r i)) /\ not (q i && r i)) /\
        (forall (rx : lseq real n). x %~ rx ==>
          a %~ masked_squares rx start width q /\ b %~ masked_squares rx start width r) /\
        (forall (ra rb : real). a %~ ra /\ b %~ rb ==> c %~ (ra +. rb)))
      (ensures forall (rx : lseq real n). x %~ rx ==>
        c %~ masked_squares rx start width p)
  = introduce forall (rx : lseq real n).
      x %~ rx ==> c %~ masked_squares rx start width p
    with (if x %~ rx then masked_partition rx start width p q r)

inline_for_extraction noextract
let approx_mask_equal
  (#n : nat) (x : lseq bf16 n)
  (start width : nat{start + width <= n})
  (p q : nat -> bool) (a : f32)
  : Lemma
      (requires
        (forall i. i < width ==> p i == q i) /\
        (forall (rx : lseq real n). x %~ rx ==> a %~ masked_squares rx start width p))
      (ensures forall (rx : lseq real n). x %~ rx ==> a %~ masked_squares rx start width q)
  = introduce forall (rx : lseq real n).
      x %~ rx ==> a %~ masked_squares rx start width q
    with (if x %~ rx then masked_equal rx start width p q)

[@@CPrologue "__device__"]
fn rec rms_warp
  (#n : erased nat) (input : larray bf16 n)
  (start : sz) (width : szp{start + width <= n})
  (warp : szlt 8)
  (step : szp{small_pow2 step})
  (lane : sz)
  (#x : erased (lseq bf16 n)) (#f : perm)
  preserves gpu ** input |-> Frac f x
  requires pure (lane < step)
  returns result : f32
  ensures pure (forall (rx : lseq real n). x %~ rx ==>
    result %~ masked_squares rx start width (lane_mask warp lane step))
  decreases (32 - step)
{
  if (step = 32sz) {
    let result = rms_leaf input start width (warp *^ 32sz +^ lane);
    approx_mask_equal x start width
      (thread_mask (warp * 32 + lane)) (lane_mask warp lane step) result;
    result
  } else {
    let next_step = step *^ 2sz;
    let next_lane = lane +^ step;
    let a = rms_warp input start width warp next_step lane;
    let b = rms_warp input start width warp next_step next_lane;
    let c = F32.add_rn_ftz a b;
    let _ = introduce forall (i : nat).
      lane_mask warp lane step i ==
        (lane_mask warp lane (2 * step) i || lane_mask warp (lane + step) (2 * step) i) /\
      not (lane_mask warp lane (2 * step) i && lane_mask warp (lane + step) (2 * step) i)
      with lane_partition warp step lane i;
    approx_mask_add x start width (lane_mask warp lane step)
      (lane_mask warp lane (2 * step)) (lane_mask warp (lane + step) (2 * step)) a b c;
    c
  }
}

[@@CPrologue "__device__"]
fn rec rms_tree
  (#n : erased nat) (input : larray bf16 n)
  (start : sz) (width : szp{start + width <= n})
  (step : szp{small_pow2 step /\ step <= 8})
  (warp : sz)
  (#x : erased (lseq bf16 n)) (#f : perm)
  preserves gpu ** input |-> Frac f x
  requires pure (warp < step)
  returns result : f32
  ensures pure (forall (rx : lseq real n). x %~ rx ==>
    result %~ masked_squares rx start width (warp_mask warp step))
  decreases (8 - step)
{
  if (step = 8sz) {
    let result = rms_warp input start width warp 1sz 0sz;
    approx_mask_equal x start width (lane_mask warp 0 1) (warp_mask warp step) result;
    result
  } else {
    let next_step = step *^ 2sz;
    let next_warp = warp +^ step;
    let a = rms_tree input start width next_step warp;
    let b = rms_tree input start width next_step next_warp;
    let c = F32.add_rn_ftz a b;
    let _ = introduce forall (i : nat).
      warp_mask warp step i ==
        (warp_mask warp (2 * step) i || warp_mask (warp + step) (2 * step) i) /\
      not (warp_mask warp (2 * step) i && warp_mask (warp + step) (2 * step) i)
      with warp_partition step warp i;
    approx_mask_add x start width (warp_mask warp step)
      (warp_mask warp (2 * step)) (warp_mask (warp + step) (2 * step)) a b c;
    c
  }
}

inline_for_extraction noextract
let rms_tree_full
  (#n : nat) (x : lseq bf16 n)
  (start width : nat{start + width <= n}) (a : f32)
  : Lemma
      (requires forall (rx : lseq real n). x %~ rx ==>
        a %~ masked_squares rx start width (warp_mask 0 1))
      (ensures forall (rx : lseq real n). x %~ rx ==>
        a %~ (sum_squares rx start width <: real))
  = introduce forall (rx : lseq real n).
      x %~ rx ==> a %~ (sum_squares rx start width <: real)
    with (if x %~ rx then (
      masked_equal rx start width (warp_mask 0 1) (fun _ -> true);
      masked_full rx start width))

inline_for_extraction noextract
let mean_model
  (#n : nat) (x : lseq bf16 n)
  (start : nat) (width : pos{start + width <= n})
  (a inv eps mean : f32)
  : Lemma
      (requires
        (forall (rx : lseq real n). x %~ rx ==> a %~ (sum_squares rx start width <: real)) /\
        inv %~ (1.0R /. R.of_int width) /\ eps %~ (epsilon <: real) /\
        (forall (ra rb rc : real). a %~ ra /\ inv %~ rb /\ eps %~ rc ==>
          mean %~ (ra *. rb +. rc)))
      (ensures forall (rx : lseq real n). x %~ rx ==>
        mean %~ (sum_squares rx start width /. R.of_int width +. epsilon))
  = introduce forall (rx : lseq real n).
      x %~ rx ==> mean %~ (sum_squares rx start width /. R.of_int width +. epsilon)
    with (if x %~ rx then (
      assert (a %~ (sum_squares rx start width <: real));
      assert (mean %~ (sum_squares rx start width *. (1.0R /. R.of_int width) +. epsilon))))

[@@CPrologue "__device__"]
fn rms_scale_value
  (#n : erased nat) (input : larray bf16 n)
  (start : sz) (width : szp{start + width <= n})
  (#x : erased (lseq bf16 n)) (#f : perm)
  preserves gpu ** input |-> Frac f x
  returns result : f32
  ensures pure (forall (rx : lseq real n). x %~ rx ==>
    result %~ (rms_scale rx start width <: real))
{
  let sum = rms_tree input start width 1sz 0sz;
  rms_tree_full x start width sum;
  let wi = FStar.Int.Cast.uint32_to_int64 (SZ.sizet_to_u32 width);
  let wf : f32 = of_int wi;
  of_int_approx #f32 wi;
  let inv : f32 = div one wf;
  div_approx #f32 one wf 1.0R (R.of_int width);
  let million : f32 = of_int 1000000L;
  of_int_approx #f32 1000000L;
  let eps : f32 = div one million;
  div_approx #f32 one million 1.0R 1000000.0R;
  let mean = F32.fma_rn_ftz sum inv eps;
  mean_model x start width sum inv eps mean;
  F32.rsqrt_approx_ftz mean
}

[@@CPrologue "__device__"]
fn normalized_product (x gamma : bf16) (scale : f32)
  preserves gpu
  returns result : bf16
  ensures pure (forall (rx rgamma rscale : real).
    x %~ rx /\ gamma %~ rgamma /\ scale %~ rscale ==>
    result %~ (rx *. (rgamma *. rscale)))
{
  let xf : f32 = fcast x;
  let gf : f32 = fcast gamma;
  let weighted_scale = F32.mul_rn_ftz scale gf;
  let scaled = F32.mul_rn_ftz xf weighted_scale;
  let result : bf16 = fcast scaled;
  let _ = introduce forall (rx rgamma rscale : real).
    x %~ rx /\ gamma %~ rgamma /\ scale %~ rscale ==>
    result %~ (rx *. (rgamma *. rscale))
    with (if x %~ rx /\ gamma %~ rgamma /\ scale %~ rscale then (
      fcast_approx #bf16 #f32 x rx;
      fcast_approx #bf16 #f32 gamma rgamma;
      assert (weighted_scale %~ (rscale *. rgamma));
      assert (scaled %~ (rx *. (rscale *. rgamma)));
      fcast_approx #f32 #bf16 scaled (rx *. (rscale *. rgamma))));
  result
}

inline_for_extraction noextract
let norm_prefix
  (n : nat) (width : pos) (rows : nat{rows * width <= n})
  (x : lseq bf16 n) (gamma : lseq bf16 width)
  (hidden : lseq bf16 (rows * width)) (count : nat)
  : prop
  = forall (rx : lseq real n) (rgamma : lseq real width).
      x %~ rx /\ gamma %~ rgamma ==>
      (forall (r : nat{r < rows}) (c : nat{c < width}).
        r * width + c < count ==>
        S.index hidden (r * width + c) %~
          normalized_cell n width rows rx rgamma r c)

inline_for_extraction noextract
let norm_prefix_write
  (n : nat) (width : pos) (rows : nat{rows * width <= n})
  (x : lseq bf16 n) (gamma : lseq bf16 width)
  (hidden : lseq bf16 (rows * width))
  (row : nat{row < rows}) (column : nat{column < width}) (v : bf16)
  : Lemma
      (requires norm_prefix n width rows x gamma hidden (row * width + column) /\
        (forall (rx : lseq real n) (rgamma : lseq real width).
          x %~ rx /\ gamma %~ rgamma ==>
          v %~ normalized_cell n width rows rx rgamma row column))
      (ensures norm_prefix n width rows x gamma
        (S.upd hidden (row * width + column) v) (row * width + column + 1))
  = ()

inline_for_extraction noextract
fn normalization_kernel
  (n : sz) (width : szp) (rows : sz{rows * width <= n})
  (input : larray bf16 n)
  (gamma_array : larray bf16 width)
  (output : larray bf16 (rows * width))
  (#x : erased (lseq bf16 n)) (#gamma : erased (lseq bf16 width))
  (#old : erased (lseq bf16 (rows * width)))
  (#fx #fgamma : perm)
  preserves gpu ** input |-> Frac fx x ** gamma_array |-> Frac fgamma gamma
  requires output |-> old
  ensures exists* (hidden : lseq bf16 (rows * width)).
    output |-> hidden ** pure (normalization_post n width rows x gamma hidden)
{
  pts_to_len output;
  let mut row = 0sz;
  while (!row <^ rows)
    invariant exists* (r : sz{r <= rows}) (h : lseq bf16 (rows * width)).
      row |-> r ** output |-> h **
      pure (norm_prefix n width rows x gamma h (r * width))
    decreases (rows - !row)
  {
    let r = !row;
    let start = r *^ width;
    let scale = rms_scale_value input start width;
    let mut column = 0sz;
    while (!column <^ width)
      invariant exists* (c : sz{c <= width}) (h : lseq bf16 (rows * width)).
        column |-> c ** output |-> h **
        pure (norm_prefix n width rows x gamma h (r * width + c))
      decreases (width - !column)
    {
      let c = !column;
      let index = start +^ c;
      let xv = input.(index);
      let gv = gamma_array.(c);
      let v = normalized_product xv gv scale;
      with h. assert output |-> h;
      norm_prefix_write n width rows x gamma h r c v;
      output.(index) <- v;
      column := !column +^ 1sz;
    };
    row := !row +^ 1sz;
  }
}

open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }
module MMA = Kuiper.TensorCore.MMA
module GEMM = Kuiper.Spec.GEMM
module BW = Kuiper.Barrier.Warp

inline_for_extraction noextract
instance c8 : concrete_sz 8 = { x = 8sz }

inline_for_extraction noextract
instance layout_a : ctlayout (rm 16 16) = Kuiper.Tensor.Layout.Alg.c_l2_row_major 16 16sz
inline_for_extraction noextract
instance layout_b : ctlayout (cm 16 8) = Kuiper.Tensor.Layout.Alg.c_l2_col_major 16sz 8
inline_for_extraction noextract
instance layout_c : ctlayout (rm 16 8) = Kuiper.Tensor.Layout.Alg.c_l2_row_major 16 8sz

inline_for_extraction noextract
let padded_cell
  (#et : Type0) {| scalar et |}
  (width rows : nat) (values : lseq et (rows * width))
  (row column : nat)
  : GTot et
  = if row < rows && column < width
    then S.index values (row * width + column)
    else zero

inline_for_extraction noextract
let weight_tile
  (#et : Type0) {| scalar et |}
  (width vocab : nat) (weights : lseq et (vocab * width))
  (cb kb : nat)
  : chest2 et 16 16
  = mk2 (fun i k -> padded_cell width vocab weights (cb + i) (kb + k))

inline_for_extraction noextract
let hidden_tile
  (#et : Type0) {| scalar et |}
  (width rows : nat) (hidden : lseq et (rows * width))
  (rb kb : nat)
  : chest2 et 16 8
  = mk2 (fun k j -> padded_cell width rows hidden (rb + j) (kb + k))

inline_for_extraction noextract
let rec dot_prefix
  (width vocab rows : nat)
  (hidden : lseq real (rows * width))
  (weights : lseq real (vocab * width))
  (row column count : nat)
  : GTot real (decreases count)
  = if count = 0 then 0.0R
    else dot_prefix width vocab rows hidden weights row column (count - 1) +.
      padded_cell width rows hidden row (count - 1) *.
      padded_cell width vocab weights column (count - 1)

inline_for_extraction noextract
let real_accumulator
  (width vocab rows : nat)
  (hidden : lseq real (rows * width))
  (weights : lseq real (vocab * width))
  (rb cb count : nat)
  : chest2 real 16 8
  = mk2 (fun i j -> dot_prefix width vocab rows hidden weights (rb + j) (cb + i) count)

inline_for_extraction noextract
let rec mma_accumulator
  (width vocab rows : nat)
  (hidden : lseq bf16 (rows * width))
  (weights : lseq bf16 (vocab * width))
  (rb cb steps : nat)
  : GTot (chest2 f32 16 8) (decreases steps)
  = if steps = 0 then const (16 @| 8 @| INil) (zero #f32)
    else MMA.emma
      (mma_accumulator width vocab rows hidden weights rb cb (steps - 1))
      (weight_tile width vocab weights cb ((steps - 1) * 16))
      (hidden_tile width rows hidden rb ((steps - 1) * 16))

inline_for_extraction noextract
let rec matmul_tile_dot
  (width vocab rows : nat)
  (hidden : lseq real (rows * width))
  (weights : lseq real (vocab * width))
  (rb cb kb : nat) (i : natlt 16) (j : natlt 8) (count : natle 16)
  : Lemma
      (ensures
        dot_prefix width vocab rows hidden weights (rb + j) (cb + i) (kb + count) ==
        dot_prefix width vocab rows hidden weights (rb + j) (cb + i) kb +.
        GEMM.__matmul_single
          (weight_tile width vocab weights cb kb)
          (hidden_tile width rows hidden rb kb) i j count)
      (decreases count)
  = if count > 0 then (
      matmul_tile_dot width vocab rows hidden weights rb cb kb i j (count - 1);
      GEMM.matmul_single_lemma
        (weight_tile width vocab weights cb kb)
        (hidden_tile width rows hidden rb kb) i j count)

inline_for_extraction noextract
let real_accumulator_step
  (width vocab rows : nat)
  (hidden : lseq real (rows * width))
  (weights : lseq real (vocab * width))
  (rb cb kb : nat)
  : Lemma (
      GEMM.matplus (real_accumulator width vocab rows hidden weights rb cb kb)
        (GEMM.matmul (weight_tile width vocab weights cb kb)
          (hidden_tile width rows hidden rb kb)) ==
      real_accumulator width vocab rows hidden weights rb cb (kb + 16))
  = let a = weight_tile width vocab weights cb kb in
    let b = hidden_tile width rows hidden rb kb in
    let lhs = GEMM.matplus (real_accumulator width vocab rows hidden weights rb cb kb)
      (GEMM.matmul a b) in
    let rhs = real_accumulator width vocab rows hidden weights rb cb (kb + 16) in
    let _ = introduce forall (idx : abs (16 @| 8 @| INil)).
      acc lhs idx == acc rhs idx
      with (let i, (j, ()) = idx in
        GEMM.lemma_matmul_index a b i j;
        matmul_tile_dot width vocab rows hidden weights rb cb kb i j 16) in
    Kuiper.Chest.lemma_equal_intro lhs rhs;
    Kuiper.Chest.ext lhs rhs

inline_for_extraction noextract
let rec mma_accumulator_approx
  (width vocab rows : nat)
  (hidden : lseq bf16 (rows * width))
  (weights : lseq bf16 (vocab * width))
  (rh : lseq real (rows * width))
  (rw : lseq real (vocab * width))
  (rb cb steps : nat)
  : Lemma
      (requires hidden %~ rh /\ weights %~ rw)
      (ensures mma_accumulator width vocab rows hidden weights rb cb steps %~
        real_accumulator width vocab rows rh rw rb cb (steps * 16))
      (decreases steps)
  = if steps > 0 then (
      mma_accumulator_approx width vocab rows hidden weights rh rw rb cb (steps - 1);
      let kb = (steps - 1) * 16 in
      let mc = mma_accumulator width vocab rows hidden weights rb cb (steps - 1) in
      let ma = weight_tile width vocab weights cb kb in
      let mb = hidden_tile width rows hidden rb kb in
      let rc = real_accumulator width vocab rows rh rw rb cb kb in
      let ra = weight_tile width vocab rw cb kb in
      let rbv = hidden_tile width rows rh rb kb in
      assert (ma %~ ra /\ mb %~ rbv);
      MMA.emma_approx_lemma mc ma mb rc ra rbv;
      real_accumulator_step width vocab rows rh rw rb cb kb)

inline_for_extraction noextract
let rec dot_pad
  (width vocab rows : nat)
  (hidden : lseq real (rows * width))
  (weights : lseq real (vocab * width))
  (row column : nat) (count : nat{width <= count})
  : Lemma
      (ensures dot_prefix width vocab rows hidden weights row column count ==
        dot_prefix width vocab rows hidden weights row column width)
      (decreases count - width)
  = if count > width then dot_pad width vocab rows hidden weights row column (count - 1)

inline_for_extraction noextract
let rec dot_rsum
  (width vocab rows : nat)
  (hidden : lseq real (rows * width))
  (weights : lseq real (vocab * width))
  (row column : nat) (count : natle width)
  : Lemma
      (ensures dot_prefix width vocab rows hidden weights row column count ==
        rsum (S.slice (S.init_ghost width (fun k ->
          padded_cell width rows hidden row k *. padded_cell width vocab weights column k))
          0 count))
      (decreases count)
  = if count > 0 then (
      dot_rsum width vocab rows hidden weights row column (count - 1);
      let terms = S.init_ghost width (fun k ->
        padded_cell width rows hidden row k *. padded_cell width vocab weights column k) in
      Kuiper.Seq.Common.lemma_seq_fold_left_slice 0.0R (+.) terms 0 (count - 1))

inline_for_extraction noextract
let dot_projected
  (width : pos) (vocab rows : nat)
  (hidden : lseq real (rows * width))
  (weights : lseq real (vocab * width))
  (row : natlt rows) (column : nat) (count : nat{width <= count})
  : Lemma
      (ensures dot_prefix width vocab rows hidden weights row column count ==
        projected_cell width vocab rows hidden weights row column)
  = dot_pad width vocab rows hidden weights row column count;
    dot_rsum width vocab rows hidden weights row column width;
    let a = S.init_ghost width (fun k ->
      padded_cell width rows hidden row k *. padded_cell width vocab weights column k) in
    let b = S.init_ghost width (fun k ->
      S.index hidden (row * width + k) *. padded_weight width vocab weights column k) in
    assert (S.equal a b)

inline_for_extraction noextract
let tile_post
  (width : pos) (vocab rows : nat)
  (hidden : lseq bf16 (rows * width))
  (weights : lseq bf16 (vocab * width))
  (rb cb : nat) (tile : chest2 f32 16 8)
  : prop
  = forall (rh : lseq real (rows * width)) (rw : lseq real (vocab * width)).
      hidden %~ rh /\ weights %~ rw ==>
      (forall (i : natlt 16) (j : natlt 8{rb + j < rows}).
        acc2 tile i j %~ projected_cell width vocab rows rh rw (rb + j) (cb + i))

inline_for_extraction noextract
let mma_tile_post
  (width : pos) (vocab rows : nat)
  (hidden : lseq bf16 (rows * width))
  (weights : lseq bf16 (vocab * width))
  (rb cb : nat) (steps : nat{width <= steps * 16})
  : Lemma (tile_post width vocab rows hidden weights rb cb
      (mma_accumulator width vocab rows hidden weights rb cb steps))
  = introduce forall (rh : lseq real (rows * width)) (rw : lseq real (vocab * width)).
      hidden %~ rh /\ weights %~ rw ==>
      (forall (i : natlt 16) (j : natlt 8{rb + j < rows}).
        acc2 (mma_accumulator width vocab rows hidden weights rb cb steps) i j %~
          projected_cell width vocab rows rh rw (rb + j) (cb + i))
    with (if hidden %~ rh /\ weights %~ rw then (
      mma_accumulator_approx width vocab rows hidden weights rh rw rb cb steps;
      introduce forall (i : natlt 16) (j : natlt 8{rb + j < rows}).
        acc2 (mma_accumulator width vocab rows hidden weights rb cb steps) i j %~
          projected_cell width vocab rows rh rw (rb + j) (cb + i)
      with dot_projected width vocab rows rh rw (rb + j) (cb + i) (steps * 16)))

inline_for_extraction noextract
fn read_padded
  (width : szp) (rows : sz)
  (input : larray bf16 (rows * width))
  (rb kb i k : sz)
  (#values : erased (lseq bf16 (rows * width))) (#f : perm)
  preserves gpu ** input |-> Frac f values
  returns result : bf16
  ensures pure (result == padded_cell width rows values (rb + i) (kb + k))
{
  if (rb <^ rows && kb <^ width) {
    let remaining_rows = rows -^ rb;
    let remaining_columns = width -^ kb;
    if (i <^ remaining_rows && k <^ remaining_columns) {
      let r = rb +^ i;
      let c = kb +^ k;
      let index = r *^ width +^ c;
      input.(index)
    } else {
      zero #bf16
    }
  } else {
    zero #bf16
  }
}

inline_for_extraction noextract
fn fill_weight_tile
  (width : szp) (vocab : sz)
  (weights : larray bf16 (vocab * width))
  (a : array2 bf16 (rm 16 16))
  (cb kb : sz)
  (#w : erased (lseq bf16 (vocab * width))) (#fw : perm)
  (#old : chest2 bf16 16 16)
  preserves gpu ** weights |-> Frac fw w
  requires a |-> old
  ensures a |-> weight_tile width vocab w cb kb
{
  let mut t = 0sz;
  while (!t <^ 256sz)
    invariant exists* (done : sz{done <= 256}) (av : chest2 bf16 16 16).
      t |-> done ** a |-> av **
      pure (forall (i k : natlt 16). i * 16 + k < done ==>
        acc2 av i k == acc2 (weight_tile width vocab w cb kb) i k)
    decreases (256 - !t)
  {
    let index = !t;
    let i = index /^ 16sz;
    let k = index %^ 16sz;
    let v = read_padded width vocab weights cb kb i k;
    tensor_write a (cidx2 i k) v;
    t := !t +^ 1sz;
  };
  with av. assert a |-> av;
  Kuiper.EMatrix.lemma_equal_intro av (weight_tile width vocab w cb kb);
  Kuiper.Chest.ext av (weight_tile width vocab w cb kb);
}

inline_for_extraction noextract
fn fill_hidden_tile
  (width : szp) (rows : sz)
  (hidden : larray bf16 (rows * width))
  (b : array2 bf16 (cm 16 8))
  (rb kb : sz)
  (#h : erased (lseq bf16 (rows * width))) (#fh : perm)
  (#old : chest2 bf16 16 8)
  preserves gpu ** hidden |-> Frac fh h
  requires b |-> old
  ensures b |-> hidden_tile width rows h rb kb
{
  let mut t = 0sz;
  while (!t <^ 128sz)
    invariant exists* (done : sz{done <= 128}) (bv : chest2 bf16 16 8).
      t |-> done ** b |-> bv **
      pure (forall (k : natlt 16) (j : natlt 8). j * 16 + k < done ==>
        acc2 bv k j == acc2 (hidden_tile width rows h rb kb) k j)
    decreases (128 - !t)
  {
    let index = !t;
    let j = index /^ 16sz;
    let k = index %^ 16sz;
    let v = read_padded width rows hidden rb kb j k;
    tensor_write b (cidx2 k j) v;
    t := !t +^ 1sz;
  };
  with bv. assert b |-> bv;
  Kuiper.EMatrix.lemma_equal_intro bv (hidden_tile width rows h rb kb);
  Kuiper.Chest.ext bv (hidden_tile width rows h rb kb);
}

inline_for_extraction noextract
unfold let leader (tid : nat) (p : slprop) : slprop =
  if tid = 0 then p else emp

ghost
fn leader_pack (p : slprop)
  norewrite
  requires p
  ensures forall+ (tid : natlt 32). leader tid p
{
  forevery_intro_if #(natlt 32) 0 (fun _ -> p);
  forevery_ext #(natlt 32) _ (fun (tid : natlt 32) -> leader tid p);
}

ghost
fn leader_unpack (p : slprop)
  norewrite
  requires forall+ (tid : natlt 32). leader tid p
  ensures p
{
  forevery_extract_if_eqtype #(natlt 32) 0 (fun tid -> leader tid p);
  forevery_ext #(natlt 32) _ (fun (_ : natlt 32) -> emp);
  forevery_emp_elim (natlt 32);
}

ghost
fn publish_inputs_proof
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (#va : chest2 bf16 16 16) (#vb : chest2 bf16 16 8)
  requires forall+ (i : natlt 32). leader i (a |-> va ** b |-> vb)
  ensures forall+ (_ : natlt 32).
    a |-> Frac (1.0R /. 32) va ** b |-> Frac (1.0R /. 32) vb
{
  leader_unpack (a |-> va ** b |-> vb);
  tensor_share_n a 32;
  tensor_share_n b 32;
  forevery_zip
    (fun (_ : natlt 32) -> a |-> Frac (1.0R /. 32) va)
    (fun (_ : natlt 32) -> b |-> Frac (1.0R /. 32) vb);
}

inline_for_extraction noextract
let publish_inputs
  (tid : szlt 32)
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (#va : chest2 bf16 16 16) (#vb : chest2 bf16 16 8)
  : stt unit
      (requires thread_id 32 tid ** leader tid (a |-> va ** b |-> vb))
      (ensures fun _ -> thread_id 32 tid **
        a |-> Frac (1.0R /. 32) va ** b |-> Frac (1.0R /. 32) vb)
  = BW.warp_barrier_wait ()
    (fun i -> leader i (a |-> va ** b |-> vb))
    (fun _ -> a |-> Frac (1.0R /. 32) va ** b |-> Frac (1.0R /. 32) vb)
    (publish_inputs_proof a b #va #vb) #32 #(SZ.v tid)

ghost
fn reclaim_inputs_proof
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (#va : chest2 bf16 16 16) (#vb : chest2 bf16 16 8)
  requires forall+ (_ : natlt 32).
    a |-> Frac (1.0R /. 32) va ** b |-> Frac (1.0R /. 32) vb
  ensures forall+ (i : natlt 32). leader i (a |-> va ** b |-> vb)
{
  forevery_unzip
    (fun (_ : natlt 32) -> a |-> Frac (1.0R /. 32) va)
    (fun (_ : natlt 32) -> b |-> Frac (1.0R /. 32) vb);
  tensor_gather_n a 32;
  tensor_gather_n b 32;
  leader_pack (a |-> va ** b |-> vb);
}

inline_for_extraction noextract
let reclaim_inputs
  (tid : szlt 32)
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (#va : chest2 bf16 16 16) (#vb : chest2 bf16 16 8)
  : stt unit
      (requires thread_id 32 tid **
        a |-> Frac (1.0R /. 32) va ** b |-> Frac (1.0R /. 32) vb)
      (ensures fun _ -> thread_id 32 tid ** leader tid (a |-> va ** b |-> vb))
  = BW.warp_barrier_wait ()
    (fun _ -> a |-> Frac (1.0R /. 32) va ** b |-> Frac (1.0R /. 32) vb)
    (fun i -> leader i (a |-> va ** b |-> vb))
    (reclaim_inputs_proof a b #va #vb) #32 #(SZ.v tid)

ghost
fn publish_accumulator_proof
  (c : array2 f32 (rm 16 8)) (#v : chest2 f32 16 8)
  requires forall+ (_ : natlt 32). c |-> Frac (1.0R /. 32) v
  ensures forall+ (i : natlt 32). leader i (c |-> v)
{
  tensor_gather_n c 32;
  leader_pack (c |-> v);
}

inline_for_extraction noextract
let publish_accumulator
  (tid : szlt 32) (c : array2 f32 (rm 16 8)) (#v : chest2 f32 16 8)
  : stt unit
      (requires thread_id 32 tid ** c |-> Frac (1.0R /. 32) v)
      (ensures fun _ -> thread_id 32 tid ** leader tid (c |-> v))
  = BW.warp_barrier_wait ()
    (fun _ -> c |-> Frac (1.0R /. 32) v)
    (fun i -> leader i (c |-> v))
    (publish_accumulator_proof c #v) #32 #(SZ.v tid)

ghost
fn release_accumulator_proof
  (c : array2 f32 (rm 16 8)) (#v : chest2 f32 16 8)
  requires forall+ (i : natlt 32). leader i (c |-> v)
  ensures forall+ (_ : natlt 32). c |-> Frac (1.0R /. 32) v
{
  leader_unpack (c |-> v);
  tensor_share_n c 32;
}

inline_for_extraction noextract
let release_accumulator
  (tid : szlt 32) (c : array2 f32 (rm 16 8)) (#v : chest2 f32 16 8)
  : stt unit
      (requires thread_id 32 tid ** leader tid (c |-> v))
      (ensures fun _ -> thread_id 32 tid ** c |-> Frac (1.0R /. 32) v)
  = BW.warp_barrier_wait ()
    (fun i -> leader i (c |-> v))
    (fun _ -> c |-> Frac (1.0R /. 32) v)
    (release_accumulator_proof c #v) #32 #(SZ.v tid)

inline_for_extraction noextract
let score_done (rb cb t r c : nat) : prop =
  r < rb \/
  (r < rb + 8 /\ c < cb) \/
  (rb <= r /\ r < rb + 8 /\ cb <= c /\ c < cb + 16 /\
    (r - rb) * 16 + c - cb < t)

inline_for_extraction noextract
let score_prefix
  (width : pos) (vocab rows : nat) (columns : nat{vocab <= columns})
  (hidden : lseq bf16 (rows * width))
  (weights : lseq bf16 (vocab * width))
  (scores : lseq bf16 (rows * columns)) (rb cb t : nat)
  : prop
  = forall (rh : lseq real (rows * width)) (rw : lseq real (vocab * width)).
      hidden %~ rh /\ weights %~ rw ==>
      (forall (r : natlt rows) (c : natlt columns).
        score_done rb cb t r c ==>
        S.index scores (r * columns + c) %~ projected_cell width vocab rows rh rw r c)

inline_for_extraction noextract
let score_done_step (rb cb t r c : nat)
  : Lemma (score_done rb cb (t + 1) r c ==>
    score_done rb cb t r c \/ (r == rb + t / 16 /\ c == cb + t % 16))
  = if score_done rb cb (t + 1) r c /\ not (score_done rb cb t r c) then (
      assert ((r - rb) * 16 + (c - cb) == t);
      Kuiper.Math.div_mod_of_mul_add 16 (r - rb) (c - cb))

inline_for_extraction noextract
let score_write
  (width : pos) (vocab rows : nat) (columns : nat{vocab <= columns})
  (hidden : lseq bf16 (rows * width)) (weights : lseq bf16 (vocab * width))
  (scores : lseq bf16 (rows * columns))
  (rb cb : nat) (t : natlt 128)
  (v : bf16)
  : Lemma
      (requires rb + t / 16 < rows /\ cb + t % 16 < columns /\
        score_prefix width vocab rows columns hidden weights scores rb cb t /\
        (forall (rh : lseq real (rows * width)) (rw : lseq real (vocab * width)).
          hidden %~ rh /\ weights %~ rw ==>
          v %~ projected_cell width vocab rows rh rw (rb + t / 16) (cb + t % 16)))
      (ensures score_prefix width vocab rows columns hidden weights
        (S.upd scores ((rb + t / 16) * columns + cb + t % 16) v) rb cb (t + 1))
  = introduce forall (rh : lseq real (rows * width)) (rw : lseq real (vocab * width)).
      hidden %~ rh /\ weights %~ rw ==>
      (forall (r : natlt rows) (c : natlt columns).
        score_done rb cb (t + 1) r c ==>
        S.index (S.upd scores ((rb + t / 16) * columns + cb + t % 16) v)
          (r * columns + c) %~ projected_cell width vocab rows rh rw r c)
    with (if hidden %~ rh /\ weights %~ rw then (
      introduce forall (r : natlt rows) (c : natlt columns).
        score_done rb cb (t + 1) r c ==>
        S.index (S.upd scores ((rb + t / 16) * columns + cb + t % 16) v)
          (r * columns + c) %~ projected_cell width vocab rows rh rw r c
      with (if score_done rb cb (t + 1) r c then (
        score_done_step rb cb t r c;
        Kuiper.Math.div_mod_of_mul_add columns r c;
        Kuiper.Math.div_mod_of_mul_add columns (rb + t / 16) (cb + t % 16);
        if r == rb + t / 16 /\ c == cb + t % 16 then ()
        else assert (S.index scores (r * columns + c) %~
          projected_cell width vocab rows rh rw r c)))))

inline_for_extraction noextract
let score_skip
  (width : pos) (vocab rows : nat) (columns : nat{vocab <= columns})
  (hidden : lseq bf16 (rows * width)) (weights : lseq bf16 (vocab * width))
  (scores : lseq bf16 (rows * columns))
  (rb cb : nat) (t : natlt 128)
  : Lemma
      (requires (rb + t / 16 >= rows \/ cb + t % 16 >= columns) /\
        score_prefix width vocab rows columns hidden weights scores rb cb t)
      (ensures score_prefix width vocab rows columns hidden weights scores rb cb (t + 1))
  = let _ = introduce forall (r : natlt rows) (c : natlt columns).
      score_done rb cb (t + 1) r c ==> score_done rb cb t r c
      with score_done_step rb cb t r c in
    ()

inline_for_extraction noextract
fn store_score
  (width : szp) (vocab rows : sz) (columns : sz{vocab <= columns})
  (output : larray bf16 (rows * columns))
  (tile : array2 f32 (rm 16 8))
  (rb : sz{rb < rows}) (cb : sz{cb < columns})
  (index : szlt 128)
  (#hidden : erased (lseq bf16 (rows * width)))
  (#weights : erased (lseq bf16 (vocab * width)))
  (#scores : erased (lseq bf16 (rows * columns)))
  (#v : chest2 f32 16 8)
  preserves gpu ** tile |-> v
  requires output |-> scores
  requires pure (tile_post width vocab rows hidden weights rb cb v /\
    score_prefix width vocab rows columns hidden weights scores rb cb index)
  ensures exists* (after : lseq bf16 (rows * columns)).
    output |-> after **
    pure (score_prefix width vocab rows columns hidden weights after rb cb (index + 1))
{
  let j = index /^ 16sz;
  let i = index %^ 16sz;
  let rleft = rows -^ rb;
  let cleft = columns -^ cb;
  if (j <^ rleft && i <^ cleft) {
    let r = rb +^ j;
    let c = cb +^ i;
    let value = tensor_read tile (cidx2 i j);
    let narrowed : bf16 = fcast value;
    score_write width vocab rows columns hidden weights scores rb cb index narrowed;
    let dst = r *^ columns +^ c;
    open Pulse.Lib.Array;
    output.(dst) <- narrowed;
  } else {
    score_skip width vocab rows columns hidden weights scores rb cb index;
    ();
  }
}

inline_for_extraction noextract
fn store_scores
  (width : szp) (vocab rows : sz) (columns : sz{vocab <= columns})
  (output : larray bf16 (rows * columns))
  (tile : array2 f32 (rm 16 8))
  (rb : sz{rb < rows}) (cb : sz{cb < columns})
  (#hidden : erased (lseq bf16 (rows * width)))
  (#weights : erased (lseq bf16 (vocab * width)))
  (#scores : erased (lseq bf16 (rows * columns)))
  (#v : chest2 f32 16 8)
  preserves gpu ** tile |-> v
  requires output |-> scores
  requires pure (tile_post width vocab rows hidden weights rb cb v /\
    score_prefix width vocab rows columns hidden weights scores rb cb 0)
  ensures exists* (after : lseq bf16 (rows * columns)).
    output |-> after **
    pure (score_prefix width vocab rows columns hidden weights after rb cb 128)
{
  let mut t = 0sz;
  while (!t <^ 128sz)
    invariant exists* (done : sz{done <= 128}) (s : lseq bf16 (rows * columns)).
      t |-> done ** output |-> s **
      pure (score_prefix width vocab rows columns hidden weights s rb cb done)
    decreases (128 - !t)
  {
    let index = !t;
    store_score width vocab rows columns output tile rb cb index #hidden #weights;
    t := !t +^ 1sz;
  }
}

inline_for_extraction noextract
let input_scratch (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8)) : slprop =
  exists* (va : chest2 bf16 16 16) (vb : chest2 bf16 16 8). a |-> va ** b |-> vb

inline_for_extraction noextract
let accumulator_scratch (c : array2 f32 (rm 16 8)) (f : perm) : slprop =
  exists* (v : chest2 f32 16 8). c |-> Frac f v

inline_for_extraction noextract
fn load_tiles
  (width : szp) (vocab rows : sz)
  (hidden : larray bf16 (rows * width)) (weights : larray bf16 (vocab * width))
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (rb cb kb : sz) (tid : szlt 32)
  (#h : erased (lseq bf16 (rows * width))) (#w : erased (lseq bf16 (vocab * width)))
  (#fh #fw : perm)
  preserves gpu ** hidden |-> Frac fh h ** weights |-> Frac fw w
  requires leader tid (input_scratch a b)
  ensures leader tid (a |-> weight_tile width vocab w cb kb **
    b |-> hidden_tile width rows h rb kb)
{
  if (tid = 0sz) {
    rewrite (leader tid (input_scratch a b)) as (input_scratch a b);
    unfold input_scratch a b;
    fill_weight_tile width vocab weights a cb kb;
    fill_hidden_tile width rows hidden b rb kb;
    rewrite (a |-> weight_tile width vocab w cb kb ** b |-> hidden_tile width rows h rb kb)
      as (leader tid (a |-> weight_tile width vocab w cb kb ** b |-> hidden_tile width rows h rb kb));
  } else {
    rewrite (leader tid (input_scratch a b)) as emp;
    rewrite emp as
      (leader tid (a |-> weight_tile width vocab w cb kb ** b |-> hidden_tile width rows h rb kb));
  }
}

ghost
fn forget_tiles
  (tid : natlt 32) (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (#va : chest2 bf16 16 16) (#vb : chest2 bf16 16 8)
  requires leader tid (a |-> va ** b |-> vb)
  ensures leader tid (input_scratch a b)
{
  if (tid = 0) {
    fold input_scratch a b;
    rewrite (input_scratch a b) as (leader tid (input_scratch a b));
  } else {
    rewrite emp as (leader tid (input_scratch a b));
  }
}

inline_for_extraction noextract
fn compute_tile
  (width : szp) (vocab rows : sz)
  (hidden : larray bf16 (rows * width)) (weights : larray bf16 (vocab * width))
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (c : array2 f32 (rm 16 8))
  (rb cb : sz) (tid : szlt 32)
  (#h : erased (lseq bf16 (rows * width))) (#w : erased (lseq bf16 (vocab * width)))
  (#fh #fw : perm) (#old : chest2 f32 16 8)
  preserves gpu ** thread_id 32 tid **
    hidden |-> Frac fh h ** weights |-> Frac fw w
  preserves leader tid (input_scratch a b)
  requires c |-> Frac (1.0R /. 32) old
  ensures c |-> Frac (1.0R /. 32)
    (mma_accumulator width vocab rows h w rb cb ((width - 1) / 16 + 1))
  ensures pure (tile_post width vocab rows h w rb cb
    (mma_accumulator width vocab rows h w rb cb ((width - 1) / 16 + 1)))
{
  let fr = MMA.alloc_fragment ();
  MMA.fill fr zero;
  let last = width -^ 1sz;
  let full = last /^ 16sz;
  let count = full +^ 1sz;
  assert pure (SZ.v count == (width - 1) / 16 + 1);
  let mut step = 0sz;
  while (!step <^ count)
    invariant exists* (i : sz{i <= count}).
      step |-> i ** fr |-> mma_accumulator width vocab rows h w rb cb i
    invariant leader tid (input_scratch a b)
    decreases (count - !step)
  {
    let i = !step;
    let kb = i *^ 16sz;
    load_tiles width vocab rows hidden weights a b rb cb kb tid;
    publish_inputs tid a b
      #(weight_tile width vocab w cb kb) #(hidden_tile width rows h rb kb);
    MMA.mma_sync a b fr;
    reclaim_inputs tid a b;
    forget_tiles tid a b;
    step := !step +^ 1sz;
  };
  MMA.store fr c;
  drop_ (fr |-> mma_accumulator width vocab rows h w rb cb count);
  mma_tile_post width vocab rows h w rb cb count;
  ();
}

inline_for_extraction noextract
let score_state
  (width : pos) (vocab rows : nat) (columns : nat{vocab <= columns})
  (hidden : lseq bf16 (rows * width)) (weights : lseq bf16 (vocab * width))
  (output : larray bf16 (rows * columns)) (rb cb t : nat)
  : slprop
  = exists* (scores : lseq bf16 (rows * columns)).
      output |-> scores **
      pure (score_prefix width vocab rows columns hidden weights scores rb cb t)

inline_for_extraction noextract
fn store_on_leader
  (width : szp) (vocab rows : sz) (columns : sz{vocab <= columns})
  (output : larray bf16 (rows * columns)) (tile : array2 f32 (rm 16 8))
  (rb : sz{rb < rows}) (cb : sz{cb < columns}) (tid : szlt 32)
  (#h : erased (lseq bf16 (rows * width))) (#w : erased (lseq bf16 (vocab * width)))
  (#v : chest2 f32 16 8)
  preserves gpu ** leader tid (tile |-> v)
  requires leader tid (score_state width vocab rows columns h w output rb cb 0)
  requires pure (tile_post width vocab rows h w rb cb v)
  ensures leader tid (score_state width vocab rows columns h w output rb cb 128)
{
  if (tid = 0sz) {
    rewrite (leader tid (tile |-> v)) as (tile |-> v);
    rewrite (leader tid (score_state width vocab rows columns h w output rb cb 0))
      as (score_state width vocab rows columns h w output rb cb 0);
    unfold score_state width vocab rows columns h w output rb cb 0;
    store_scores width vocab rows columns output tile rb cb #h #w;
    fold score_state width vocab rows columns h w output rb cb 128;
    rewrite (tile |-> v) as (leader tid (tile |-> v));
    rewrite (score_state width vocab rows columns h w output rb cb 128)
      as (leader tid (score_state width vocab rows columns h w output rb cb 128));
  } else {
    rewrite (leader tid (score_state width vocab rows columns h w output rb cb 0)) as emp;
    rewrite emp as (leader tid (score_state width vocab rows columns h w output rb cb 128));
  }
}

ghost
fn move_score_state
  (width : pos) (vocab rows : nat) (columns : nat{vocab <= columns})
  (hidden : lseq bf16 (rows * width)) (weights : lseq bf16 (vocab * width))
  (output : larray bf16 (rows * columns))
  (rb cb t rb' cb' t' : nat) (tid : natlt 32)
  requires pure (forall (r : natlt rows) (c : natlt columns).
    score_done rb' cb' t' r c ==> score_done rb cb t r c)
  requires leader tid (score_state width vocab rows columns hidden weights output rb cb t)
  ensures leader tid (score_state width vocab rows columns hidden weights output rb' cb' t')
{
  if (tid = 0) {
    unfold score_state width vocab rows columns hidden weights output rb cb t;
    fold score_state width vocab rows columns hidden weights output rb' cb' t';
    rewrite (score_state width vocab rows columns hidden weights output rb' cb' t')
      as (leader tid (score_state width vocab rows columns hidden weights output rb' cb' t'));
  } else {
    rewrite emp as (leader tid (score_state width vocab rows columns hidden weights output rb' cb' t'));
  }
}

inline_for_extraction noextract
fn projection_thread
  (width : szp) (vocab rows : sz) (columns : sz{vocab <= columns})
  (hidden : larray bf16 (rows * width)) (weights : larray bf16 (vocab * width))
  (output : larray bf16 (rows * columns))
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (c : array2 f32 (rm 16 8))
  (#h : erased (lseq bf16 (rows * width))) (#w : erased (lseq bf16 (vocab * width)))
  (#fh #fw : perm)
  (tid : szlt 32) ()
  preserves gpu ** thread_id 32 tid **
    hidden |-> Frac fh h ** weights |-> Frac fw w
  preserves leader tid (input_scratch a b) ** accumulator_scratch c (1.0R /. 32)
  requires leader tid (score_state width vocab rows columns h w output 0 0 0)
  ensures leader tid (score_state width vocab rows columns h w output rows 0 0)
{
  let mut row = 0sz;
  while (!row <^ rows)
    invariant exists* (rb : sz{rb <= rows}).
      row |-> rb **
      leader tid (score_state width vocab rows columns h w output rb 0 0)
    invariant leader tid (input_scratch a b) ** accumulator_scratch c (1.0R /. 32)
    decreases (rows - !row)
  {
    let rb = !row;
    let mut column = 0sz;
    while (!column <^ columns)
      invariant exists* (cb : sz{cb <= columns}).
        column |-> cb **
        leader tid (score_state width vocab rows columns h w output rb cb 0)
      invariant leader tid (input_scratch a b) ** accumulator_scratch c (1.0R /. 32)
      decreases (columns - !column)
    {
      let cb = !column;
      unfold accumulator_scratch c (1.0R /. 32);
      compute_tile width vocab rows hidden weights a b c rb cb tid;
      publish_accumulator tid c;
      store_on_leader width vocab rows columns output c rb cb tid #h #w;
      release_accumulator tid c;
      fold accumulator_scratch c (1.0R /. 32);
      column := bounded_advance !column columns 16sz;
      let next = !column;
      move_score_state width vocab rows columns h w output rb cb 128 rb next 0 tid;
    };
    let end_column = !column;
    row := bounded_advance !row rows 8sz;
    let next = !row;
    move_score_state width vocab rows columns h w output rb end_column 0 next 0 0 tid;
  };
  let end_row = !row;
  move_score_state width vocab rows columns h w output end_row 0 0 rows 0 0 tid;
}

inline_for_extraction noextract
let warp_state
  (hidden weights : array bf16)
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (c : array2 f32 (rm 16 8))
  (h w : seq bf16) (fh fw : perm) (p : slprop) (tid : natlt 32)
  : slprop
  = (hidden |-> Frac (fh /. 32) h ** weights |-> Frac (fw /. 32) w **
      accumulator_scratch c (1.0R /. 32)) **
    leader tid (input_scratch a b) ** leader tid p

ghost
fn projection_setup
  (hidden weights : array bf16)
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (c : array2 f32 (rm 16 8))
  (#h #w : seq bf16) (#fh #fw : perm) (p : slprop) ()
  norewrite
  requires hidden |-> Frac fh h ** weights |-> Frac fw w **
    input_scratch a b ** accumulator_scratch c 1.0R ** p
  ensures (forall+ (tid : natlt 32). warp_state hidden weights a b c h w fh fw p tid) ** emp
{
  Kuiper.Array.Extra.array_share hidden #h #fh 32;
  Kuiper.Array.Extra.array_share weights #w #fw 32;
  unfold accumulator_scratch c 1.0R;
  with cv. assert c |-> cv;
  tensor_share_n c 32;
  forevery_map #(natlt 32)
    (fun _ -> c |-> Frac (1.0R /. 32) cv)
    (fun _ -> accumulator_scratch c (1.0R /. 32))
    fn _ { fold accumulator_scratch c (1.0R /. 32); };
  leader_pack (input_scratch a b);
  leader_pack p;
  forevery_zip3 #(natlt 32)
    (fun _ -> hidden |-> Frac (fh /. 32) h)
    (fun _ -> weights |-> Frac (fw /. 32) w)
    (fun _ -> accumulator_scratch c (1.0R /. 32));
  forevery_zip3 #(natlt 32)
    (fun _ -> hidden |-> Frac (fh /. 32) h ** weights |-> Frac (fw /. 32) w **
      accumulator_scratch c (1.0R /. 32))
    (fun tid -> leader tid (input_scratch a b))
    (fun tid -> leader tid p);
  forevery_map #(natlt 32)
    (fun tid ->
      (hidden |-> Frac (fh /. 32) h ** weights |-> Frac (fw /. 32) w **
        accumulator_scratch c (1.0R /. 32)) **
      leader tid (input_scratch a b) ** leader tid p)
    (fun tid -> warp_state hidden weights a b c h w fh fw p tid)
    fn tid { fold warp_state hidden weights a b c h w fh fw p tid; };
}

ghost
fn projection_teardown
  (hidden weights : array bf16)
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (c : array2 f32 (rm 16 8))
  (#h #w : seq bf16) (#fh #fw : perm) (p : slprop) ()
  norewrite
  requires (forall+ (tid : natlt 32). warp_state hidden weights a b c h w fh fw p tid) ** emp
  ensures hidden |-> Frac fh h ** weights |-> Frac fw w **
    input_scratch a b ** accumulator_scratch c 1.0R ** p
{
  forevery_map #(natlt 32)
    (fun tid -> warp_state hidden weights a b c h w fh fw p tid)
    (fun tid ->
      (hidden |-> Frac (fh /. 32) h ** weights |-> Frac (fw /. 32) w **
        accumulator_scratch c (1.0R /. 32)) **
      leader tid (input_scratch a b) ** leader tid p)
    fn tid { unfold warp_state hidden weights a b c h w fh fw p tid; };
  forevery_unzip3 #(natlt 32)
    (fun _ -> hidden |-> Frac (fh /. 32) h ** weights |-> Frac (fw /. 32) w **
      accumulator_scratch c (1.0R /. 32))
    (fun tid -> leader tid (input_scratch a b))
    (fun tid -> leader tid p);
  forevery_unzip3 #(natlt 32)
    (fun _ -> hidden |-> Frac (fh /. 32) h)
    (fun _ -> weights |-> Frac (fw /. 32) w)
    (fun _ -> accumulator_scratch c (1.0R /. 32));
  Kuiper.Array.Extra.array_gather hidden #h #fh 32;
  Kuiper.Array.Extra.array_gather weights #w #fw 32;
  leader_unpack (input_scratch a b);
  leader_unpack p;
  forevery_map #(natlt 32)
    (fun _ -> accumulator_scratch c (1.0R /. 32))
    (fun _ -> exists* cv. c |-> Frac (1.0R /. 32) cv)
    fn _ { unfold accumulator_scratch c (1.0R /. 32); };
  tensor_gather_n_underspec c 32;
  fold accumulator_scratch c 1.0R;
}

inline_for_extraction noextract
fn projection_kernel
  (width : szp) (vocab rows : sz) (columns : sz{vocab <= columns})
  (hidden : larray bf16 (rows * width)) (weights : larray bf16 (vocab * width))
  (output : larray bf16 (rows * columns))
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (c : array2 f32 (rm 16 8))
  (#h : erased (lseq bf16 (rows * width))) (#w : erased (lseq bf16 (vocab * width)))
  (#fh #fw : perm)
  (tid : szlt 32) ()
  requires gpu **
    warp_state hidden weights a b c h w fh fw
      (score_state width vocab rows columns h w output 0 0 0) tid **
    thread_id 32 tid
  ensures gpu **
    warp_state hidden weights a b c h w fh fw
      (score_state width vocab rows columns h w output rows 0 0) tid **
    thread_id 32 tid
{
  unfold warp_state hidden weights a b c h w fh fw
    (score_state width vocab rows columns h w output 0 0 0) tid;
  projection_thread width vocab rows columns hidden weights output a b c #h #w
    #(fh /. 32) #(fw /. 32) tid ();
  fold warp_state hidden weights a b c h w fh fw
    (score_state width vocab rows columns h w output rows 0 0) tid;
}

inline_for_extraction noextract
instance send_global_array_contents
  (#et : Type0) (arr : array et{is_global_array arr}) (#f : perm) (contents : seq et)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

inline_for_extraction noextract
instance send_leader
  (tid : nat) (p : slprop) (sp : is_send_across gpu_of p)
  : is_send_across gpu_of (leader tid p)
  = fun l0 l1 -> if tid = 0 then sp l0 l1
    else let s : is_send_across gpu_of emp = solve in s l0 l1

inline_for_extraction noextract
let projection_desc
  (width : szp) (vocab rows : sz) (columns : sz{vocab <= columns})
  (hidden : larray bf16 (rows * width){is_global_array hidden})
  (weights : larray bf16 (vocab * width){is_global_array weights})
  (output : larray bf16 (rows * columns){is_global_array output})
  (a : array2 bf16 (rm 16 16){is_global a})
  (b : array2 bf16 (cm 16 8){is_global b})
  (c : array2 f32 (rm 16 8){is_global c})
  (#h : erased (lseq bf16 (rows * width))) (#w : erased (lseq bf16 (vocab * width)))
  (#fh #fw : perm)
  : kernel_desc_1_n
      (hidden |-> Frac fh h ** weights |-> Frac fw w **
        input_scratch a b ** accumulator_scratch c 1.0R **
        score_state width vocab rows columns h w output 0 0 0)
      (hidden |-> Frac fh h ** weights |-> Frac fw w **
        input_scratch a b ** accumulator_scratch c 1.0R **
        score_state width vocab rows columns h w output rows 0 0)
  = {
    nthr = 32sz;
    frame = emp;
    kpre = warp_state hidden weights a b c h w fh fw
      (score_state width vocab rows columns h w output 0 0 0);
    kpost = warp_state hidden weights a b c h w fh fw
      (score_state width vocab rows columns h w output rows 0 0);
    block_setup = projection_setup hidden weights a b c #h #w #fh #fw
      (score_state width vocab rows columns h w output 0 0 0);
    block_teardown = projection_teardown hidden weights a b c #h #w #fh #fw
      (score_state width vocab rows columns h w output rows 0 0);
    f = projection_kernel width vocab rows columns hidden weights output a b c #h #w #fh #fw;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve;
  }

ghost
fn prepare_projection_buffers
  (width : pos) (vocab rows : nat) (columns : nat{vocab <= columns})
  (output : larray bf16 (rows * columns))
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (c : array2 f32 (rm 16 8))
  (#h : lseq bf16 (rows * width)) (#w : lseq bf16 (vocab * width))
  (#old : lseq bf16 (rows * columns))
  (#va : chest2 bf16 16 16) (#vb : chest2 bf16 16 8) (#vc : chest2 f32 16 8)
  requires output |-> old ** a |-> va ** b |-> vb ** c |-> vc
  ensures input_scratch a b ** accumulator_scratch c 1.0R **
    score_state width vocab rows columns h w output 0 0 0
{
  fold input_scratch a b;
  fold accumulator_scratch c 1.0R;
  fold score_state width vocab rows columns h w output 0 0 0;
}

ghost
fn finish_projection_buffers
  (width : pos) (vocab rows : nat) (columns : nat{vocab <= columns})
  (output : larray bf16 (rows * columns))
  (a : array2 bf16 (rm 16 16)) (b : array2 bf16 (cm 16 8))
  (c : array2 f32 (rm 16 8))
  (#h : lseq bf16 (rows * width)) (#w : lseq bf16 (vocab * width))
  requires input_scratch a b ** accumulator_scratch c 1.0R **
    score_state width vocab rows columns h w output rows 0 0
  ensures exists* (va : chest2 bf16 16 16) (vb : chest2 bf16 16 8)
                  (vc : chest2 f32 16 8) (scores : lseq bf16 (rows * columns)).
    a |-> va ** b |-> vb ** c |-> vc ** output |-> scores **
    pure (projection_post width vocab rows columns h w scores)
{
  unfold input_scratch a b;
  unfold accumulator_scratch c 1.0R;
  unfold score_state width vocab rows columns h w output rows 0 0;
}

#push-options "--fuel 4 --ifuel 2 --z3rlimit 100"

inline_for_extraction noextract
let real_sum_op (a b:real)
  : Lemma (Sum.op #real #Sum.real_add_semigroup a b == a +. b) =
  assert_norm (Sum.op #real #Sum.real_add_semigroup a b == a +. b)

inline_for_extraction noextract
let rec squares_refine_shared
  (rx:lseq real input_elements)
  (p:Shared.prompt_index)
  (row:Shared.hidden_row)
  (count:nat{count <= Shared.hidden_size})
  : Lemma
      (requires forall (d:Shared.hidden_index).
        S.index rx (p * Shared.hidden_size + d) == row d)
      (ensures sum_squares rx (p * Shared.hidden_size) count ==
       Sum.sum #real #Sum.real_add_semigroup 0 count
         (fun (d:Shared.hidden_index) -> row d *. row d))
      (decreases count) =
  if count > 0 then (
    squares_refine_shared rx p row (count - 1);
    let f = fun (d:Shared.hidden_index) -> row d *. row d in
    assert (S.index rx (p * Shared.hidden_size + count - 1) == row (count - 1));
    assert (sum_squares rx (p * Shared.hidden_size) count ==
      sum_squares rx (p * Shared.hidden_size) (count - 1) +. f (count - 1));
    assert (sum_squares rx (p * Shared.hidden_size) (count - 1) ==
      Sum.sum #real #Sum.real_add_semigroup 0 (count - 1) f);
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 count f;
    real_sum_op (Sum.sum #real #Sum.real_add_semigroup 0 (count - 1) f)
      (f (count - 1));
    assert (Sum.sum #real #Sum.real_add_semigroup 0 count f ==
      sum_squares rx (p * Shared.hidden_size) count)
  )

inline_for_extraction noextract
let shared_mean_definition (row:Shared.hidden_row)
  : Lemma (
      Shared.mean_square row ==
        Sum.sum #real #Sum.real_add_semigroup 0 1024
          (fun (d:Shared.hidden_index) -> row d *. row d) /. 1024.0R) =
  assert (Shared.mean_square row ==
    Sum.sum #real #Sum.real_add_semigroup 0 1024
      (fun (d:Shared.hidden_index) -> row d *. row d) /. 1024.0R) by
    (FStar.Tactics.V2.norm
      [delta_only [`%Shared.mean_square; `%Shared.sum_hidden; `%Shared.hidden_size]];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let normalization_product_equality (a b c left right:real)
  : Lemma
      (requires left == a *. (b *. c) /\ right == b *. (a *. c))
      (ensures left == right)
  = ()

inline_for_extraction noextract
let normalized_cell_refines_shared
  (rx:lseq real input_elements)
  (rgamma:lseq real Shared.hidden_size)
  (rweights:lseq real embedding_elements)
  (p:Shared.prompt_index) (d:Shared.hidden_index)
  : Lemma (
      normalized_cell input_elements Shared.hidden_size Shared.prompt_length
        rx rgamma p d ==
      Shared.final_rms_norm (map_inputs rx rgamma rweights) p d) =
  let logical = map_inputs rx rgamma rweights in
  let row = logical.hidden_in p in
  squares_refine_shared rx p row Shared.hidden_size;
  shared_mean_definition row;
  assert (Shared.mean_square row ==
    sum_squares rx (p * Shared.hidden_size) Shared.hidden_size /.
      R.of_int Shared.hidden_size);
  assert (rms_scale rx (p * Shared.hidden_size) Shared.hidden_size ==
    Shared.normalization_scale row);
  assert (Shared.final_rms_norm logical p d ==
    logical.norm_weight d *.
      (logical.hidden_in p d *. Shared.normalization_scale row)) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.final_rms_norm]];
     FStar.Tactics.V2.smt ());
  assert (logical.norm_weight d == S.index rgamma d);
  assert (logical.hidden_in p d == S.index rx (p * Shared.hidden_size + d));
  assert (normalized_cell input_elements Shared.hidden_size Shared.prompt_length
      rx rgamma p d ==
    S.index rx (p * Shared.hidden_size + d) *.
      (S.index rgamma d *. rms_scale rx (p * Shared.hidden_size) Shared.hidden_size));
  normalization_product_equality
    (S.index rx (p * Shared.hidden_size + d)) (S.index rgamma d)
    (rms_scale rx (p * Shared.hidden_size) Shared.hidden_size)
    (normalized_cell input_elements Shared.hidden_size Shared.prompt_length rx rgamma p d)
    (Shared.final_rms_norm logical p d)

inline_for_extraction noextract
let normalized_values (x:Shared.inputs) : GTot (lseq real hidden_elements) =
  S.init_ghost hidden_elements (fun i ->
    Shared.final_rms_norm x (i / Shared.hidden_size) (i % Shared.hidden_size))

inline_for_extraction noextract
let normalized_values_at
  (x:Shared.inputs) (p:Shared.prompt_index) (d:Shared.hidden_index)
  : Lemma (
      S.index (normalized_values x) (p * Shared.hidden_size + d) ==
      Shared.final_rms_norm x p d) =
  Kuiper.Math.div_mod_of_mul_add Shared.hidden_size p d

inline_for_extraction noextract
let normalization_refines_shared
  (x:lseq bf16 input_elements)
  (gamma:lseq bf16 Shared.hidden_size)
  (hidden:lseq bf16 hidden_elements)
  (rx:lseq real input_elements)
  (rgamma:lseq real Shared.hidden_size)
  (rweights:lseq real embedding_elements)
  : Lemma
      (requires
        normalization_post input_elements Shared.hidden_size Shared.prompt_length
          x gamma hidden /\ x %~ rx /\ gamma %~ rgamma)
      (ensures
        hidden %~ normalized_values (map_inputs rx rgamma rweights) /\
        (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
          S.index hidden (p * Shared.hidden_size + d) %~
            Shared.final_rms_norm (map_inputs rx rgamma rweights) p d)) =
  let logical = map_inputs rx rgamma rweights in
  let _ = introduce forall (p:Shared.prompt_index) (d:Shared.hidden_index).
    S.index hidden (p * Shared.hidden_size + d) %~
      Shared.final_rms_norm logical p d
    with normalized_cell_refines_shared rx rgamma rweights p d in
  let _ = introduce forall (i:natlt hidden_elements).
    S.index hidden i %~ S.index (normalized_values logical) i
    with (
      let p = i / Shared.hidden_size in
      let d = i % Shared.hidden_size in
      assert (p * Shared.hidden_size + d == i);
      normalized_values_at logical p d) in
  ()

inline_for_extraction noextract
let projection_terms (logical:Shared.inputs)
  (p:Shared.prompt_index) (v:Shared.vocab_index) (d:Shared.hidden_index)
  : GTot real =
  Shared.final_rms_norm logical p d *. logical.embedding v d

inline_for_extraction noextract
let shared_projection_definition (logical:Shared.inputs)
  (p:Shared.prompt_index) (v:Shared.vocab_index)
  : Lemma (
      Shared.tied_projection logical.embedding (Shared.final_rms_norm logical p) v ==
      Sum.sum #real #Sum.real_add_semigroup 0 Shared.hidden_size
        (projection_terms logical p v)) =
  assert (
    Shared.tied_projection logical.embedding (Shared.final_rms_norm logical p) v ==
    Sum.sum #real #Sum.real_add_semigroup 0 Shared.hidden_size
      (projection_terms logical p v)) by
    (FStar.Tactics.V2.norm
      [delta_only [`%Shared.tied_projection; `%Shared.sum_hidden;
        `%Shared.hidden_size; `%projection_terms]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let rec projection_prefix_refines_shared
  (logical:Shared.inputs)
  (rweights:lseq real embedding_elements)
  (p:Shared.prompt_index) (v:Shared.vocab_index)
  (count:nat{count <= Shared.hidden_size})
  : Lemma
      (requires forall (d:Shared.hidden_index).
        S.index rweights (v * Shared.hidden_size + d) == logical.embedding v d)
      (ensures dot_prefix Shared.hidden_size Shared.vocab_size Shared.prompt_length
         (normalized_values logical) rweights p v count ==
       Sum.sum #real #Sum.real_add_semigroup 0 count
         (projection_terms logical p v))
      (decreases count) =
  if count > 0 then (
    projection_prefix_refines_shared logical rweights p v (count - 1);
    normalized_values_at logical p (count - 1);
    let f = projection_terms logical p v in
    assert (padded_cell Shared.hidden_size Shared.prompt_length
      (normalized_values logical) p (count - 1) ==
      Shared.final_rms_norm logical p (count - 1));
    assert (padded_cell Shared.hidden_size Shared.vocab_size
      rweights v (count - 1) == logical.embedding v (count - 1));
    assert (dot_prefix Shared.hidden_size Shared.vocab_size Shared.prompt_length
      (normalized_values logical) rweights p v count ==
      dot_prefix Shared.hidden_size Shared.vocab_size Shared.prompt_length
        (normalized_values logical) rweights p v (count - 1) +. f (count - 1));
    assert (dot_prefix Shared.hidden_size Shared.vocab_size Shared.prompt_length
      (normalized_values logical) rweights p v (count - 1) ==
      Sum.sum #real #Sum.real_add_semigroup 0 (count - 1) f);
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 count f;
    real_sum_op (Sum.sum #real #Sum.real_add_semigroup 0 (count - 1) f)
      (f (count - 1));
    assert (dot_prefix Shared.hidden_size Shared.vocab_size Shared.prompt_length
      (normalized_values logical) rweights p v count ==
      Sum.sum #real #Sum.real_add_semigroup 0 count f)
  ) else (
    assert_norm (dot_prefix Shared.hidden_size Shared.vocab_size Shared.prompt_length
      (normalized_values logical) rweights p v 0 == 0.0R);
    assert_norm (Sum.sum #real #Sum.real_add_semigroup 0 0
      (projection_terms logical p v) == 0.0R)
  )

inline_for_extraction noextract
let rec projection_padding_zero
  (hidden:lseq real hidden_elements)
  (weights:lseq real embedding_elements)
  (p:Shared.prompt_index)
  (c:score_column{Shared.vocab_size <= c})
  (count:nat{count <= Shared.hidden_size})
  : Lemma
      (dot_prefix Shared.hidden_size Shared.vocab_size Shared.prompt_length
        hidden weights p c count == 0.0R)
      (decreases count) =
  if count > 0 then projection_padding_zero hidden weights p c (count - 1)

inline_for_extraction noextract
let projected_cell_refines_shared
  (rx:lseq real input_elements)
  (rgamma:lseq real Shared.hidden_size)
  (rweights:lseq real embedding_elements)
  (p:Shared.prompt_index) (c:score_column)
  : Lemma
      (let logical = map_inputs rx rgamma rweights in
       projected_cell Shared.hidden_size Shared.vocab_size Shared.prompt_length
         (normalized_values logical) rweights p c ==
       (expected_outputs logical).scores p c) =
  let logical = map_inputs rx rgamma rweights in
  dot_projected Shared.hidden_size Shared.vocab_size Shared.prompt_length
    (normalized_values logical) rweights p c Shared.hidden_size;
  if c < Shared.vocab_size then (
    projection_prefix_refines_shared logical rweights p c Shared.hidden_size;
    shared_projection_definition logical p c
  )
  else
    projection_padding_zero (normalized_values logical) rweights p c Shared.hidden_size

inline_for_extraction noextract
let complete_shared_refinement
  (x:lseq bf16 input_elements)
  (gamma:lseq bf16 Shared.hidden_size)
  (weights:lseq bf16 embedding_elements)
  (hidden:lseq bf16 hidden_elements)
  (scores:lseq bf16 score_elements)
  : Lemma
      (requires
        final_norm_projection_post input_elements Shared.hidden_size
          Shared.vocab_size Shared.prompt_length score_columns
          x gamma weights hidden scores)
      (ensures shared_refinement_post x gamma weights hidden scores) =
  introduce forall (rx:lseq real input_elements)
                   (rgamma:lseq real Shared.hidden_size)
                   (rweights:lseq real embedding_elements).
    input_representation x gamma weights rx rgamma rweights ==>
    (exists (observed:output_denotation).
      output_representation hidden scores observed /\
      Shared.postcondition (map_inputs rx rgamma rweights)
        (map_state x gamma weights) (map_outputs observed)
        (map_state x gamma weights) /\
      all_scores (map_inputs rx rgamma rweights) observed)
  with (
    if input_representation x gamma weights rx rgamma rweights then (
      let logical = map_inputs rx rgamma rweights in
      let observed = expected_outputs logical in
      normalization_refines_shared x gamma hidden rx rgamma rweights;
      let _ = introduce forall (p:Shared.prompt_index) (c:score_column).
        S.index scores (p * score_columns + c) %~ observed.scores p c
        with (
          assert (S.index scores (p * score_columns + c) %~
            projected_cell Shared.hidden_size Shared.vocab_size Shared.prompt_length
              (normalized_values logical) rweights p c);
          projected_cell_refines_shared rx rgamma rweights p c) in
      assert (output_representation hidden scores observed);
      expected_outputs_refine_shared logical (map_state x gamma weights)
    ))

#pop-options

fn kplay_run
  (input_length : u64{U64.v input_length == input_elements})
  (width : u64{U64.v width == Shared.hidden_size})
  (vocab_rows : u64{U64.v vocab_rows == Shared.vocab_size})
  (rows : u32{U32.v rows == Shared.prompt_length})
  (columns : u32{U32.v columns == score_columns})
  (input0 : larray bf16 (U64.v input_length){is_global_array input0})
  (input1 : larray bf16 (U64.v width){is_global_array input1})
  (input2 : larray bf16 (U64.v vocab_rows * U64.v width){is_global_array input2})
  (output0 : larray bf16 (U32.v rows * U64.v width){is_global_array output0})
  (output1 : larray bf16 (U32.v rows * U32.v columns){is_global_array output1})
  (#x : erased (lseq bf16 (U64.v input_length)))
  (#gamma : erased (lseq bf16 (U64.v width)))
  (#weights : erased (lseq bf16 (U64.v vocab_rows * U64.v width)))
  (#old_hidden : erased (lseq bf16 (U32.v rows * U64.v width)))
  (#old_scores : erased (lseq bf16 (U32.v rows * U32.v columns)))
  (#fx #fgamma #fweights : perm)
  preserves cpu
  preserves on gpu_loc (
    input0 |-> Frac fx x ** input1 |-> Frac fgamma gamma ** input2 |-> Frac fweights weights)
  requires pure (addressable
    (U64.v input_length) (U64.v width) (U64.v vocab_rows) (U32.v rows) (U32.v columns))
  requires on gpu_loc (output0 |-> old_hidden ** output1 |-> old_scores)
  ensures exists* (hidden : lseq bf16 (U32.v rows * U64.v width))
                  (scores : lseq bf16 (U32.v rows * U32.v columns)).
    on gpu_loc (output0 |-> hidden ** output1 |-> scores) **
    pure (final_norm_projection_post
      (U64.v input_length) (U64.v width) (U64.v vocab_rows)
      (U32.v rows) (U32.v columns) x gamma weights hidden scores /\
      shared_refinement_post x gamma weights hidden scores)
{
  let n = FStar.SizeT.uint64_to_sizet input_length;
  let d = FStar.SizeT.uint64_to_sizet width;
  let v = FStar.SizeT.uint64_to_sizet vocab_rows;
  let r = SZ.uint32_to_sizet rows;
  let p = SZ.uint32_to_sizet columns;
  let w : erased (lseq bf16 (v * d)) = hide (reveal weights);
  launch_kernel_1
    (fun () -> normalization_kernel n d r input0 input1 output0
      #x #gamma #old_hidden #fx #fgamma);
  with hidden. assert on gpu_loc (output0 |-> hidden);
  let a = Kuiper.Tensor.alloc0 #bf16 256sz (rm 16 16);
  let b = Kuiper.Tensor.alloc0 #bf16 128sz (cm 16 8);
  let c = Kuiper.Tensor.alloc0 #f32 128sz (rm 16 8);
  Kuiper.Locs.map_loc gpu_loc
    (fun () -> prepare_projection_buffers d v r p output1 a b c #hidden #w #old_scores);
  launch_kernel_full_sync
    (projection_desc d v r p output0 input2 output1 a b c #hidden #w #1.0R #fweights);
  Kuiper.Locs.map_loc gpu_loc
    (fun () -> finish_projection_buffers d v r p output1 a b c #hidden #w);
  with scores. assert on gpu_loc (output1 |-> scores);
  complete_shared_refinement x gamma weights hidden scores;
  Kuiper.Tensor.free a;
  Kuiper.Tensor.free b;
  Kuiper.Tensor.free c;
}
