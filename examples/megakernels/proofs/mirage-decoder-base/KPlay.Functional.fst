module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Approximates.Base
open FStar.Ghost

module Shared = KPlay.SharedSpec
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module Casts = Kuiper.Float.Casts
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module U64 = FStar.UInt64
module Native = Kuiper.Float32.FastMath
module Exp = FStar.Math.Exp
module Root = FStar.Math.Sqrt

(* The base proof needs no native numerical hypotheses. All additions for
   the bilateral scope are erased; the source arithmetic is unchanged. *)

(* Reused from fused-rms-probe; local declarations are namespaced. *)


open Kuiper
open Kuiper.Approximates.Base

module Nr_F32 = Kuiper.Float32
module Nr_Native = Kuiper.Float32.FastMath
module Nr_Seq = FStar.Seq
module Nr_Classical = FStar.Classical
module Nr_BF16 = Kuiper.BFloat16
module Nr_Casts = Kuiper.Float.Casts
module Nr_Root = FStar.Math.Sqrt
module Nr_Shared = KPlay.SharedSpec

let nr_fma_at (x y z result:Nr_F32.t) (xr yr zr:real)
  : Lemma
      (requires
        (forall (a b c:real).
          v_approximates x a /\ v_approximates y b /\ v_approximates z c ==>
          v_approximates result (a *. b +. c)) /\
        v_approximates x xr /\ v_approximates y yr /\ v_approximates z zr)
      (ensures v_approximates result (xr *. yr +. zr))
  = ()

inline_for_extraction noextract
fn nr_rn_add (x y:Nr_F32.t) (#xr #yr:erased real)
  preserves gpu
  requires pure (v_approximates x (reveal xr) /\ v_approximates y (reveal yr))
  returns result:Nr_F32.t
  ensures pure (v_approximates result (reveal xr +. reveal yr))
{
  let one_ok : squash (v_approximates (one #Nr_F32.t) 1.0R) = a1 #Nr_F32.t;
  let result = Nr_Native.fma_rn x one y;
  nr_fma_at x (one #Nr_F32.t) y result (reveal xr) 1.0R (reveal yr);
  result
}

noextract let rec nr_sum_n (n:nat) (f:nat -> GTot real)
  : GTot real (decreases n) =
  if n = 0 then 0.0R else nr_sum_n (n - 1) f +. f (n - 1)

let rec nr_sum_ext (n:nat) (f g:nat -> GTot real)
  : Lemma
      (requires forall (i:nat{i < n}). f i == g i)
      (ensures nr_sum_n n f == nr_sum_n n g)
      (decreases n) =
  if n = 0 then () else nr_sum_ext (n - 1) f g

let rec nr_sum_add (n:nat) (f g:nat -> GTot real)
  : Lemma
      (ensures nr_sum_n n (fun i -> f i +. g i) == nr_sum_n n f +. nr_sum_n n g)
      (decreases n) =
  if n = 0 then () else nr_sum_add (n - 1) f g

let rec nr_sum_append (n m:nat) (f:nat -> GTot real)
  : Lemma
      (ensures nr_sum_n (n + m) f == nr_sum_n n f +. nr_sum_n m (fun i -> f (n + i)))
      (decreases m) =
  if m = 0 then () else nr_sum_append n (m - 1) f

let nr_sum_halves (n:pos{n % 2 == 0}) (f:nat -> GTot real)
  : Lemma (nr_sum_n n f == nr_sum_n (n / 2) (fun i -> f i +. f (i + n / 2))) =
  nr_sum_append (n / 2) (n / 2) f;
  nr_sum_add (n / 2) f (fun i -> f (i + n / 2));
  nr_sum_ext (n / 2) (fun i -> f (n / 2 + i)) (fun i -> f (i + n / 2))

noextract let nr_at (values:seq real) (i:nat) : Tot real =
  if i < Nr_Seq.length values then Nr_Seq.index values i else 0.0R

noextract let nr_sum_values (values:seq real) : GTot real =
  nr_sum_n (Nr_Seq.length values) (nr_at values)

noextract let nr_paired_values (#count:nat) (values:lseq real count)
  : GTot (lseq real (count / 2)) =
  Nr_Seq.init_ghost (count / 2)
    (fun i -> Nr_Seq.index values i +. Nr_Seq.index values (i + count / 2))

let nr_paired_total (#count:pos{count % 2 == 0}) (values:lseq real count)
  : Lemma (nr_sum_values (nr_paired_values values) == nr_sum_values values) =
  nr_sum_halves count (nr_at values);
  nr_sum_ext (count / 2) (nr_at (nr_paired_values values))
    (fun i -> nr_at values i +. nr_at values (i + count / 2))

#push-options "--fuel 4 --ifuel 1"
let nr_singleton_total (values:lseq real 1)
  : Lemma (nr_sum_values values == Nr_Seq.index values 0) = ()
#pop-options

noextract let nr_valid_count (count:nat) : prop =
  count == 1 \/ count == 2 \/ count == 4 \/ count == 8 \/
  count == 16 \/ count == 32 \/ count == 64 \/ count == 128

noextract let nr_represents_prefix
  (storage:lseq Nr_F32.t 128) (count:nat{count <= 128})
  (logical:lseq real count) : prop =
  forall (i:nat{i < count}).
    v_approximates (Nr_Seq.index storage i) (Nr_Seq.index logical i)

noextract let nr_same_outside
  (original current:lseq Nr_F32.t 128) (count:nat) : prop =
  forall (i:nat{count <= i /\ i < 128}).
    Nr_Seq.index current i == Nr_Seq.index original i

noextract let nr_paired_prefix
  (original current:lseq Nr_F32.t 128)
  (count:pos{count <= 128 /\ count % 2 == 0})
  (logical:lseq real count) (steps:nat{steps <= count / 2}) : prop =
  nr_same_outside original current steps /\
  (forall (i:nat{i < steps}).
    v_approximates (Nr_Seq.index current i)
      (Nr_Seq.index logical i +. Nr_Seq.index logical (i + count / 2)))

let nr_pair_progress
  (original current updated:lseq Nr_F32.t 128)
  (count:pos{count <= 128 /\ count % 2 == 0})
  (logical:lseq real count) (step:nat{step < count / 2}) (value:Nr_F32.t)
  : Lemma
      (requires
        nr_paired_prefix original current count logical step /\
        updated == Nr_Seq.upd current step value /\
        v_approximates value
          (Nr_Seq.index logical step +. Nr_Seq.index logical (step + count / 2)))
      (ensures nr_paired_prefix original updated count logical (step + 1)) =
  let outside (i:nat{step + 1 <= i /\ i < 128})
    : Lemma (Nr_Seq.index updated i == Nr_Seq.index original i) =
    assert (i <> step)
  in
  Nr_Classical.forall_intro outside;
  let inside (i:nat{i < step + 1})
    : Lemma (v_approximates (Nr_Seq.index updated i)
        (Nr_Seq.index logical i +. Nr_Seq.index logical (i + count / 2))) =
    if i = step then assert (Nr_Seq.index updated i == value)
    else assert (Nr_Seq.index updated i == Nr_Seq.index current i)
  in
  Nr_Classical.forall_intro inside

[@@CPrologue "__device__"]
fn rec nr_reduce_tree
  (buffer:larray Nr_F32.t 128)
  (count:szp{nr_valid_count count})
  (#original:erased (lseq Nr_F32.t 128))
  (#logical:erased (lseq real count))
  preserves gpu
  requires buffer |-> original ** pure (nr_represents_prefix original count logical)
  returns result:Nr_F32.t
  ensures exists* (current:lseq Nr_F32.t 128). buffer |-> current **
    pure (nr_same_outside original current count /\
      Nr_Seq.index current 0 == result /\
      v_approximates result (nr_sum_values logical))
  decreases (Kuiper.SizeT.v count)
{
  if (count = 1sz) {
    nr_singleton_total logical;
    let result = buffer.(0sz);
    result
  } else {
    let half = count /^ 2sz;
    assert pure (half > 0 /\ half < count /\ nr_valid_count half /\ count % 2 == 0);
    let mut index = 0sz;
    while (!index <^ half)
      invariant exists* (j:sz{j <= half}) (current:lseq Nr_F32.t 128).
        index |-> j ** buffer |-> current **
        pure (nr_paired_prefix original current count logical j)
      decreases (half - !index)
    {
      let j = !index;
      with current. assert buffer |-> current;
      assert pure (Nr_Seq.index current j == Nr_Seq.index original j);
      assert pure (Nr_Seq.index current (j + half) == Nr_Seq.index original (j + half));
      let a = buffer.(j);
      let b = buffer.(j +^ half);
      let value = nr_rn_add a b #(Nr_Seq.index logical j) #(Nr_Seq.index logical (j + half));
      buffer.(j) <- value;
      with updated. assert buffer |-> updated;
      nr_pair_progress original current updated count logical j value;
      index := j +^ 1sz;
    };
    let finished = !index;
    assert pure (Kuiper.SizeT.v finished == Kuiper.SizeT.v half);
    with middle. assert buffer |-> middle;
    let pair : erased (lseq real half) = nr_paired_values logical;
    assert pure (nr_represents_prefix middle half pair);
    let result = nr_reduce_tree buffer half #middle #pair;
    nr_paired_total logical;
    with current. assert buffer |-> current;
    assert pure (nr_same_outside original current count);
    result
  }
}

inline_for_extraction noextract
fn nr_reduction_kernel
  (buffer:larray Nr_F32.t 128)
  (count:szp{nr_valid_count count})
  (#original:erased (lseq Nr_F32.t 128))
  (#logical:erased (lseq real count))
  preserves gpu
  requires buffer |-> original ** pure (nr_represents_prefix original count logical)
  ensures exists* (current:lseq Nr_F32.t 128). buffer |-> current **
    pure (nr_same_outside original current count /\
      v_approximates (Nr_Seq.index current 0) (nr_sum_values logical))
{
  let result = nr_reduce_tree buffer count #original #logical;
}

instance nr_send_reduction_array
  (#a:Type0) (arr:array a{is_global_array arr}) (#f:perm) (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn nr_run_reduce
  (buffer:larray Nr_F32.t 128{is_global_array buffer})
  (count:szp{nr_valid_count count})
  (#original:erased (lseq Nr_F32.t 128))
  (#logical:erased (lseq real count))
  preserves cpu
  requires on gpu_loc (buffer |-> original) **
    pure (nr_represents_prefix original count logical)
  ensures exists* (current:lseq Nr_F32.t 128). on gpu_loc (buffer |-> current) **
    pure (nr_same_outside original current count /\
      v_approximates (Nr_Seq.index current 0) (nr_sum_values logical))
{
  launch_kernel_1 (fun () -> nr_reduction_kernel buffer count #original #logical);
}

inline_for_extraction noextract
let nr_negative_zero ()
  : Tot (value:Nr_F32.t{v_approximates value 0.0R}) =
  let minus_one = of_int #Nr_F32.t (-1L) in
  of_int_approx #Nr_F32.t (-1L);
  a_mul minus_one (zero #Nr_F32.t) (0.0R -. 1.0R) 0.0R;
  mul minus_one (zero #Nr_F32.t)

inline_for_extraction noextract
fn nr_rn_multiply (x y:Nr_F32.t) (#xr #yr:erased real)
  preserves gpu
  requires pure (v_approximates x (reveal xr) /\ v_approximates y (reveal yr))
  returns result:Nr_F32.t
  ensures pure (v_approximates result (reveal xr *. reveal yr))
{
  let z = nr_negative_zero ();
  let result = Nr_Native.fma_rn x y z;
  nr_fma_at x y z result (reveal xr) (reveal yr) 0.0R;
  result
}

inline_for_extraction noextract
let nr_reciprocal_width (width:FStar.Int64.t{FStar.Int64.v width > 0})
  : Tot (value:Nr_F32.t{
      v_approximates value (1.0R /. FStar.Real.of_int (FStar.Int64.v width))}) =
  let denominator = of_int #Nr_F32.t width in
  of_int_approx #Nr_F32.t width;
  div_approx (one #Nr_F32.t) denominator 1.0R (FStar.Real.of_int (FStar.Int64.v width));
  div #Nr_F32.t one denominator

inline_for_extraction noextract
let nr_epsilon () : Tot (value:Nr_F32.t{v_approximates value 0.000001R}) =
  nr_reciprocal_width 1000000L

let nr_sum_step (n:pos) (f:nat -> GTot real)
  : Lemma (nr_sum_n n f == nr_sum_n (n - 1) f +. f (n - 1)) = ()

let rec nr_sum_flat (n m:nat) (f:nat -> GTot real)
  : Lemma
      (ensures nr_sum_n n (fun i -> nr_sum_n m (fun j -> f (i * m + j))) == nr_sum_n (n * m) f)
      (decreases n) =
  if n = 0 then () else (
    nr_sum_flat (n - 1) m f;
    nr_sum_append ((n - 1) * m) m f;
    assert ((n - 1) * m + m == n * m);
    nr_sum_step n (fun i -> nr_sum_n m (fun j -> f (i * m + j)))
  )

let rec nr_sum_shared (n:nat) (f:Nr_Shared.vector n) (g:nat -> GTot real)
  : Lemma
      (requires forall (i:Nr_Shared.index n). f i == g i)
      (ensures Nr_Shared.sum n f == nr_sum_n n g)
      (decreases n) =
  if n = 0 then () else (
    nr_sum_shared (n - 1) f g;
    Kuiper.Sum.sum_pop_right 0 n f
  )

let rec shared_sum_equal (n:nat) (x y:Shared.vector n)
  : Lemma
      (requires forall (i:natlt n). x i == y i)
      (ensures Shared.sum n x == Shared.sum n y)
      (decreases n) =
  if n > 0 then (
    shared_sum_equal (n-1) x y;
    Kuiper.Sum.sum_pop_right #real #Kuiper.Sum.real_add_semigroup 0 n x;
    Kuiper.Sum.sum_pop_right #real #Kuiper.Sum.real_add_semigroup 0 n y
  )

noextract let nr_square_at (values:seq real) (i:nat) : GTot real =
  nr_at values i *. nr_at values i

noextract let nr_four_squares (values:seq real) (base:nat) : GTot real =
  ((nr_square_at values base +. nr_square_at values (base + 1)) +.
    nr_square_at values (base + 2)) +. nr_square_at values (base + 3)

#push-options "--fuel 6 --ifuel 1"
let nr_four_squares_sum (values:seq real) (base:nat)
  : Lemma (nr_four_squares values base ==
      nr_sum_n 4 (fun j -> nr_square_at values (base + j))) = ()
#pop-options

noextract let nr_rms_width (width:nat) : prop = width == 128 \/ width == 1024

noextract let nr_lane_count (width:nat) : nat = if width = 128 then 32 else 128

noextract let nr_lane_total (values:seq real) (lane:nat) : GTot real =
  nr_four_squares values (lane * 4) +.
    (if Nr_Seq.length values = 1024 then nr_four_squares values (lane * 4 + 512) else 0.0R)

noextract let nr_lane_values (#width:nat) (logical:lseq real width)
  : GTot (lseq real (nr_lane_count width)) =
  Nr_Seq.init_ghost (nr_lane_count width) (nr_lane_total logical)

let nr_lane_sums (#width:nat{nr_rms_width width}) (logical:lseq real width)
  : Lemma (nr_sum_values (nr_lane_values logical) == nr_sum_n width (nr_square_at logical)) =
  nr_sum_ext (nr_lane_count width) (nr_at (nr_lane_values logical)) (nr_lane_total logical);
  let low (lane:nat) = nr_four_squares logical (lane * 4) in
  let high (lane:nat) = nr_four_squares logical (lane * 4 + 512) in
  let low_ok (lane:nat)
    : Lemma (low lane == nr_sum_n 4 (fun j -> nr_square_at logical (lane * 4 + j))) =
    nr_four_squares_sum logical (lane * 4)
  in
  Nr_Classical.forall_intro low_ok;
  if width = 128 then (
    nr_sum_ext 32 (nr_lane_total logical)
      (fun lane -> nr_sum_n 4 (fun j -> nr_square_at logical (lane * 4 + j)));
    nr_sum_flat 32 4 (nr_square_at logical)
  ) else (
    let high_ok (lane:nat)
      : Lemma (high lane ==
          nr_sum_n 4 (fun j -> nr_square_at logical (lane * 4 + j + 512))) =
      nr_four_squares_sum logical (lane * 4 + 512);
      nr_sum_ext 4 (fun j -> nr_square_at logical (lane * 4 + 512 + j))
        (fun j -> nr_square_at logical (lane * 4 + j + 512))
    in
    Nr_Classical.forall_intro high_ok;
    nr_sum_ext 128 (nr_lane_total logical) (fun lane -> low lane +. high lane);
    nr_sum_add 128 low high;
    nr_sum_ext 128 low (fun lane -> nr_sum_n 4 (fun j -> nr_square_at logical (lane * 4 + j)));
    nr_sum_ext 128 high (fun lane -> nr_sum_n 4 (fun j -> nr_square_at logical (lane * 4 + j + 512)));
    nr_sum_flat 128 4 (nr_square_at logical);
    nr_sum_flat 128 4 (fun i -> nr_square_at logical (i + 512));
    nr_sum_append 512 512 (nr_square_at logical);
    nr_sum_ext 512 (fun i -> nr_square_at logical (512 + i))
      (fun i -> nr_square_at logical (i + 512))
  )

noextract let nr_rms_total (#width:pos) (logical:lseq real width)
  : GTot Nr_Root.rnonneg =
  let squares : Nr_Shared.vector width =
    fun i -> Nr_Seq.index logical i *. Nr_Seq.index logical i
  in
  Nr_Shared.sum_nonnegative width squares;
  nr_sum_shared width squares (nr_square_at logical);
  nr_sum_n width (nr_square_at logical)

noextract let nr_rms_inverse (#width:pos) (logical:lseq real width) : GTot real =
  Nr_Root.rsqrt (nr_rms_total logical /. FStar.Real.of_int width +. 0.000001R)

#push-options "--fuel 4 --ifuel 1"
let nr_shared_rms_point (#width:pos)
  (weight logical:lseq real width) (i:Nr_Shared.index width)
  : Lemma (
      Nr_Shared.rms_norm #width (Nr_Seq.index weight) (Nr_Seq.index logical) i ==
      Nr_Seq.index weight i *. (Nr_Seq.index logical i *. nr_rms_inverse logical)) =
  let squares : Nr_Shared.vector width =
    fun j -> Nr_Seq.index logical j *. Nr_Seq.index logical j
  in
  Nr_Shared.sum_nonnegative width squares;
  nr_sum_shared width squares (nr_square_at logical);
  assert_norm (
    Nr_Shared.rms_norm #width (Nr_Seq.index weight) (Nr_Seq.index logical) i ==
    Nr_Seq.index weight i *.
      (Nr_Seq.index logical i *. Nr_Root.rsqrt
        (Nr_Shared.sum width squares /. FStar.Real.of_int width +. 0.000001R)))
#pop-options

noextract let nr_bf16_segment
  (#n:nat) (storage:lseq Nr_BF16.t n) (start:nat)
  (width:nat{start + width <= n}) (logical:lseq real width) : prop =
  forall (i:nat{i < width}).
    v_approximates (Nr_Seq.index storage (start + i)) (Nr_Seq.index logical i)

inline_for_extraction noextract
fn nr_square_value (x:Nr_BF16.t) (#xr:erased real)
  preserves gpu
  requires pure (v_approximates x (reveal xr))
  returns result:Nr_F32.t
  ensures pure (v_approximates result (reveal xr *. reveal xr))
{
  Nr_Casts.fcast_approx #Nr_BF16.t #Nr_F32.t x (reveal xr);
  let wide : Nr_F32.t = Nr_Casts.fcast x;
  nr_rn_multiply wide wide #xr #xr
}

inline_for_extraction noextract
fn nr_square_pair (x y:Nr_BF16.t) (#xr #yr:erased real)
  preserves gpu
  requires pure (v_approximates x (reveal xr) /\ v_approximates y (reveal yr))
  returns result:Nr_F32.t
  ensures pure (v_approximates result
    (reveal xr *. reveal xr +. reveal yr *. reveal yr))
{
  let first = nr_square_value x #xr;
  let initial = nr_rn_add (zero #Nr_F32.t) first #0.0R #(reveal xr *. reveal xr);
  let second = nr_square_value y #yr;
  nr_rn_add initial second #(reveal xr *. reveal xr) #(reveal yr *. reveal yr)
}

noextract let nr_component_total (logical:seq real) (i:nat) : GTot real =
  nr_square_at logical i +.
    (if Nr_Seq.length logical = 1024 then nr_square_at logical (i + 512) else 0.0R)

inline_for_extraction noextract
fn nr_rms_component
  (#n:erased nat) (input:larray Nr_BF16.t n) (start:sz)
  (width:szp{nr_rms_width width /\ start + width <= n /\ Kuiper.SizeT.fits (start + width)})
  (i:sz{i < 4 * nr_lane_count width})
  (#values:erased (lseq Nr_BF16.t n)) (#logical:erased (lseq real width))
  preserves gpu ** input |-> values
  requires pure (nr_bf16_segment values start width logical)
  returns result:Nr_F32.t
  ensures pure (v_approximates result (nr_component_total logical i))
{
  let a = input.(start +^ i);
  if (width = 128sz) {
    nr_square_value a #(Nr_Seq.index logical i)
  } else {
    let b = input.(start +^ i +^ 512sz);
    nr_square_pair a b #(Nr_Seq.index logical i) #(Nr_Seq.index logical (i + 512))
  }
}

inline_for_extraction noextract
fn nr_rms_lane
  (#n:erased nat) (input:larray Nr_BF16.t n) (start:sz)
  (width:szp{nr_rms_width width /\ start + width <= n /\ Kuiper.SizeT.fits (start + width)})
  (lane:sz{lane < nr_lane_count width})
  (#values:erased (lseq Nr_BF16.t n)) (#logical:erased (lseq real width))
  preserves gpu ** input |-> values
  requires pure (nr_bf16_segment values start width logical)
  returns result:Nr_F32.t
  ensures pure (v_approximates result (nr_lane_total logical lane))
{
  let base = lane *^ 4sz;
  let q0 = nr_rms_component input start width base #values #logical;
  let q1 = nr_rms_component input start width (base +^ 1sz) #values #logical;
  let q2 = nr_rms_component input start width (base +^ 2sz) #values #logical;
  let q3 = nr_rms_component input start width (base +^ 3sz) #values #logical;
  let r0 : erased real = nr_component_total logical base;
  let r1 : erased real = nr_component_total logical (base + 1);
  let r2 : erased real = nr_component_total logical (base + 2);
  let r3 : erased real = nr_component_total logical (base + 3);
  let first = nr_rn_add q0 q1 #r0 #r1;
  let second = nr_rn_add first q2 #(reveal r0 +. reveal r1) #r2;
  nr_rn_add second q3 #((reveal r0 +. reveal r1) +. reveal r2) #r3
}

let nr_prefix_progress
  (#a:Type0) (#n:nat) (before after:lseq a n)
  (step:nat{step < n}) (value:a) (p:nat -> a -> prop)
  : Lemma
      (requires after == Nr_Seq.upd before step value /\ p step value /\
        (forall (i:nat{i < step}). p i (Nr_Seq.index before i)))
      (ensures forall (i:nat{i < step + 1}). p i (Nr_Seq.index after i)) =
  let one_index (i:nat{i < step + 1})
    : Lemma (p i (Nr_Seq.index after i)) =
    if i = step then assert (Nr_Seq.index after i == value)
    else assert (Nr_Seq.index after i == Nr_Seq.index before i)
  in
  Nr_Classical.forall_intro one_index

inline_for_extraction noextract
fn nr_rms_squares
  (#n:erased nat) (input:larray Nr_BF16.t n) (buffer:larray Nr_F32.t 128)
  (start:sz)
  (width:szp{nr_rms_width width /\ start + width <= n /\ Kuiper.SizeT.fits (start + width)})
  (#values:erased (lseq Nr_BF16.t n)) (#old:erased (lseq Nr_F32.t 128))
  (#logical:erased (lseq real width))
  preserves gpu ** input |-> values
  requires buffer |-> old ** pure (nr_bf16_segment values start width logical)
  returns result:Nr_F32.t
  ensures exists* (current:lseq Nr_F32.t 128). buffer |-> current **
    pure (v_approximates result (nr_rms_total logical))
{
  let lanes = (if width = 128sz then 32sz else 128sz);
  let mut index = 0sz;
  while (!index <^ lanes)
    invariant exists* (j:sz{j <= lanes}) (current:lseq Nr_F32.t 128).
      index |-> j ** buffer |-> current **
      pure (forall (i:nat{i < j}).
        v_approximates (Nr_Seq.index current i) (nr_lane_total logical i))
    decreases (lanes - !index)
  {
    let j = !index;
    let value = nr_rms_lane input start width j #values #logical;
    with before. assert buffer |-> before;
    buffer.(j) <- value;
    with after. assert buffer |-> after;
    nr_prefix_progress before after j value
      (fun i v -> v_approximates v (nr_lane_total logical i));
    index := j +^ 1sz;
  };
  let finished = !index;
  assert pure (Kuiper.SizeT.v finished == Kuiper.SizeT.v lanes);
  with partials. assert buffer |-> partials;
  let real_partials : erased (lseq real lanes) = nr_lane_values logical;
  assert pure (nr_represents_prefix partials lanes real_partials);
  let squared_sum = nr_reduce_tree buffer lanes #partials #real_partials;
  nr_lane_sums logical;
  squared_sum
}

inline_for_extraction noextract
fn nr_rms_scale
  (width:szp{nr_rms_width width}) (squared_sum:Nr_F32.t)
  (#sr:erased real{reveal sr >=. 0.0R})
  preserves gpu
  requires pure (v_approximates squared_sum (reveal sr))
  returns result:Nr_F32.t
  ensures pure (v_approximates result
    (Nr_Root.rsqrt (reveal sr /. FStar.Real.of_int (Kuiper.SizeT.v width) +. 0.000001R)))
{
  let inverse_width = (if width = 128sz then nr_reciprocal_width 128L else nr_reciprocal_width 1024L);
  let mean = nr_rn_multiply squared_sum inverse_width
    #sr #(1.0R /. FStar.Real.of_int (Kuiper.SizeT.v width));
  let shifted = nr_rn_add mean (nr_epsilon ())
    #(reveal sr *. (1.0R /. FStar.Real.of_int (Kuiper.SizeT.v width))) #0.000001R;
  rsqrt_approx shifted (reveal sr /. FStar.Real.of_int (Kuiper.SizeT.v width) +. 0.000001R);
  rsqrt shifted
}

inline_for_extraction noextract
fn nr_normalize_value
  (x gamma:Nr_BF16.t) (scale:Nr_F32.t) (#xr #gr #sr:erased real)
  preserves gpu
  requires pure (v_approximates x (reveal xr) /\ v_approximates gamma (reveal gr) /\
    v_approximates scale (reveal sr))
  returns result:Nr_BF16.t
  ensures pure (v_approximates result (reveal gr *. (reveal xr *. reveal sr)))
{
  Nr_Casts.fcast_approx #Nr_BF16.t #Nr_F32.t x (reveal xr);
  let xf : Nr_F32.t = Nr_Casts.fcast x;
  let scaled = nr_rn_multiply xf scale #xr #sr;
  Nr_Casts.fcast_approx #Nr_F32.t #Nr_BF16.t scaled (reveal xr *. reveal sr);
  let normalized : Nr_BF16.t = Nr_Casts.fcast scaled;
  Nr_Casts.fcast_approx #Nr_BF16.t #Nr_F32.t normalized (reveal xr *. reveal sr);
  Nr_Casts.fcast_approx #Nr_BF16.t #Nr_F32.t gamma (reveal gr);
  let nf : Nr_F32.t = Nr_Casts.fcast normalized;
  let gf : Nr_F32.t = Nr_Casts.fcast gamma;
  let weighted = nr_rn_multiply nf gf #(reveal xr *. reveal sr) #gr;
  Nr_Casts.fcast_approx #Nr_F32.t #Nr_BF16.t weighted ((reveal xr *. reveal sr) *. reveal gr);
  Nr_Casts.fcast weighted
}

[@@"opaque_to_smt"]
noextract let nr_rms_values (#width:pos) (weight logical:lseq real width)
  : GTot (result:lseq real width{
      forall (i:Nr_Shared.index width).
        Nr_Seq.index result i == Nr_Shared.rms_norm #width (Nr_Seq.index weight) (Nr_Seq.index logical) i /\
        Nr_Seq.index result i == Nr_Seq.index weight i *. (Nr_Seq.index logical i *. nr_rms_inverse logical)}) =
  let result = Nr_Seq.init_ghost width
    (Nr_Shared.rms_norm #width (Nr_Seq.index weight) (Nr_Seq.index logical)) in
  let point (i:Nr_Shared.index width)
    : Lemma (
        Nr_Seq.index result i == Nr_Shared.rms_norm #width (Nr_Seq.index weight) (Nr_Seq.index logical) i /\
        Nr_Seq.index result i == Nr_Seq.index weight i *. (Nr_Seq.index logical i *. nr_rms_inverse logical)) =
    nr_shared_rms_point weight logical i
  in
  Nr_Classical.forall_intro point;
  result

noextract let nr_same_outside_segment
  (#n:nat) (original current:lseq Nr_BF16.t n) (start count:nat) : prop =
  forall (i:nat{i < n}).
    i < start \/ start + count <= i ==> Nr_Seq.index current i == Nr_Seq.index original i

noextract let nr_rms_output_prefix
  (#n:nat) (original current:lseq Nr_BF16.t n) (start:nat)
  (width:pos{start + width <= n}) (weight logical:lseq real width)
  (steps:nat{steps <= width}) : prop =
  nr_same_outside_segment original current start steps /\
  (forall (i:nat{i < steps}).
    v_approximates (Nr_Seq.index current (start + i))
      (Nr_Seq.index weight i *. (Nr_Seq.index logical i *. nr_rms_inverse logical)))

let nr_rms_output_progress
  (#n:nat) (original current updated:lseq Nr_BF16.t n) (start:nat)
  (width:pos{start + width <= n}) (weight logical:lseq real width)
  (step:nat{step < width}) (value:Nr_BF16.t)
  : Lemma
      (requires nr_rms_output_prefix original current start width weight logical step /\
        updated == Nr_Seq.upd current (start + step) value /\
        v_approximates value
          (Nr_Seq.index weight step *. (Nr_Seq.index logical step *. nr_rms_inverse logical)))
      (ensures nr_rms_output_prefix original updated start width weight logical (step + 1)) =
  let outside (i:nat{i < n})
    : Lemma
        (requires i < start \/ start + (step + 1) <= i)
        (ensures Nr_Seq.index updated i == Nr_Seq.index original i) =
    assert (i <> start + step)
  in
  Nr_Classical.forall_intro (Nr_Classical.move_requires outside);
  let inside (i:nat{i < step + 1})
    : Lemma (v_approximates (Nr_Seq.index updated (start + i))
        (Nr_Seq.index weight i *. (Nr_Seq.index logical i *. nr_rms_inverse logical))) =
    if i = step then assert (Nr_Seq.index updated (start + i) == value)
    else assert (Nr_Seq.index updated (start + i) == Nr_Seq.index current (start + i))
  in
  Nr_Classical.forall_intro inside

let nr_rms_output_done
  (#n:nat) (original current:lseq Nr_BF16.t n) (start:nat)
  (width:pos{start + width <= n}) (weight logical:lseq real width)
  : Lemma
      (requires nr_rms_output_prefix original current start width weight logical width)
      (ensures nr_same_outside_segment original current start width /\
        nr_bf16_segment current start width (nr_rms_values weight logical)) =
  let one_index (i:Nr_Shared.index width)
    : Lemma (v_approximates (Nr_Seq.index current (start + i))
        (Nr_Seq.index (nr_rms_values weight logical) i)) =
    nr_shared_rms_point weight logical i
  in
  Nr_Classical.forall_intro one_index

inline_for_extraction noextract
fn nr_finish_rms_output
  (#n:erased nat) (start:sz) (width:szp{start + width <= n})
  (#original #current:erased (lseq Nr_BF16.t n))
  (#weight #logical:erased (lseq real width))
  preserves gpu
  requires pure (nr_rms_output_prefix original current start width weight logical width)
  ensures pure (nr_same_outside_segment original current start width /\
    nr_bf16_segment current start width (nr_rms_values weight logical))
{
  nr_rms_output_done original current start width weight logical;
}

inline_for_extraction noextract
fn nr_rms_row
  (#n #w #m:erased nat)
  (input:larray Nr_BF16.t n) (weights:larray Nr_BF16.t w)
  (output:larray Nr_BF16.t m) (buffer:larray Nr_F32.t 128)
  (start weight_start output_start:sz)
  (width:szp{nr_rms_width width /\
    start + width <= n /\ weight_start + width <= w /\ output_start + width <= m /\
    Kuiper.SizeT.fits (start + width) /\ Kuiper.SizeT.fits (weight_start + width) /\
    Kuiper.SizeT.fits (output_start + width)})
  (#values:erased (lseq Nr_BF16.t n)) (#weight_values:erased (lseq Nr_BF16.t w))
  (#old:erased (lseq Nr_BF16.t m)) (#scratch:erased (lseq Nr_F32.t 128))
  (#logical #weight:erased (lseq real width))
  (#weight_perm:perm)
  preserves gpu ** input |-> values ** weights |-> Frac weight_perm weight_values
  requires output |-> old ** buffer |-> scratch **
    pure (nr_bf16_segment values start width logical /\
      nr_bf16_segment weight_values weight_start width weight)
  ensures exists* (current:lseq Nr_BF16.t m) (scratch_values:lseq Nr_F32.t 128).
    output |-> current ** buffer |-> scratch_values **
    pure (nr_same_outside_segment old current output_start width /\
      nr_bf16_segment current output_start width (nr_rms_values weight logical))
{
  let squared_sum = nr_rms_squares input buffer start width #values #scratch #logical;
  let scale = nr_rms_scale width squared_sum #(nr_rms_total logical);
  let mut index = 0sz;
  while (!index <^ width)
    invariant exists* (j:sz{j <= width}) (current:lseq Nr_BF16.t m).
      index |-> j ** output |-> current **
      pure (nr_rms_output_prefix old current output_start width weight logical j)
    decreases (width - !index)
  {
    let j = !index;
    let x = input.(start +^ j);
    let gamma = weights.(weight_start +^ j);
    let value = nr_normalize_value x gamma scale
      #(Nr_Seq.index logical j) #(Nr_Seq.index weight j) #(nr_rms_inverse logical);
    with before. assert output |-> before;
    output.(output_start +^ j) <- value;
    with after. assert output |-> after;
    nr_rms_output_progress old before after output_start width weight logical j value;
    index := j +^ 1sz;
  };
  let finished = !index;
  assert pure (Kuiper.SizeT.v finished == Kuiper.SizeT.v width);
  with current. assert output |-> current;
  nr_finish_rms_output output_start width #old #current #weight #logical;
}

fn nr_run_rms
  (#n #w #m:erased nat)
  (input:larray Nr_BF16.t n{is_global_array input})
  (weights:larray Nr_BF16.t w{is_global_array weights})
  (output:larray Nr_BF16.t m{is_global_array output})
  (buffer:larray Nr_F32.t 128{is_global_array buffer})
  (start weight_start output_start:sz)
  (width:szp{nr_rms_width width /\
    start + width <= n /\ weight_start + width <= w /\ output_start + width <= m /\
    Kuiper.SizeT.fits (start + width) /\ Kuiper.SizeT.fits (weight_start + width) /\
    Kuiper.SizeT.fits (output_start + width)})
  (#values:erased (lseq Nr_BF16.t n)) (#weight_values:erased (lseq Nr_BF16.t w))
  (#old:erased (lseq Nr_BF16.t m)) (#scratch:erased (lseq Nr_F32.t 128))
  (#logical #weight:erased (lseq real width))
  preserves cpu ** on gpu_loc (input |-> values) ** on gpu_loc (weights |-> weight_values)
  requires on gpu_loc (output |-> old) ** on gpu_loc (buffer |-> scratch) **
    pure (nr_bf16_segment values start width logical /\
      nr_bf16_segment weight_values weight_start width weight)
  ensures exists* (current:lseq Nr_BF16.t m) (scratch_values:lseq Nr_F32.t 128).
    on gpu_loc (output |-> current) ** on gpu_loc (buffer |-> scratch_values) **
    pure (nr_same_outside_segment old current output_start width /\
      nr_bf16_segment current output_start width (nr_rms_values weight logical))
{
  launch_kernel_1 (fun () -> nr_rms_row input weights output buffer
    start weight_start output_start width #values #weight_values #old #scratch #logical #weight #1.0R);
}

[@@"opaque_to_smt"]
noextract let nr_added_values (#count:nat) (left right:lseq real count)
  : GTot (result:lseq real count{
      forall (i:nat{i < count}).
        Nr_Seq.index result i == Nr_Seq.index left i +. Nr_Seq.index right i}) =
  Nr_Seq.init_ghost count (fun i -> Nr_Seq.index left i +. Nr_Seq.index right i)

inline_for_extraction noextract
fn nr_add_bf16 (x y:Nr_BF16.t) (#xr #yr:erased real)
  preserves gpu
  requires pure (v_approximates x (reveal xr) /\ v_approximates y (reveal yr))
  returns result:Nr_BF16.t
  ensures pure (v_approximates result (reveal xr +. reveal yr))
{
  Nr_Casts.fcast_approx #Nr_BF16.t #Nr_F32.t x (reveal xr);
  Nr_Casts.fcast_approx #Nr_BF16.t #Nr_F32.t y (reveal yr);
  let xf : Nr_F32.t = Nr_Casts.fcast x;
  let yf : Nr_F32.t = Nr_Casts.fcast y;
  let sum = nr_rn_add xf yf #xr #yr;
  Nr_Casts.fcast_approx #Nr_F32.t #Nr_BF16.t sum (reveal xr +. reveal yr);
  Nr_Casts.fcast sum
}

noextract let nr_segment_prefix
  (#n:nat) (original current:lseq Nr_BF16.t n) (start:nat)
  (count:nat{start + count <= n}) (logical:lseq real count)
  (steps:nat{steps <= count}) : prop =
  nr_same_outside_segment original current start steps /\
  (forall (i:nat{i < steps}).
    v_approximates (Nr_Seq.index current (start + i)) (Nr_Seq.index logical i))

let nr_segment_progress
  (#n:nat) (original current updated:lseq Nr_BF16.t n) (start:nat)
  (count:nat{start + count <= n}) (logical:lseq real count)
  (step:nat{step < count}) (value:Nr_BF16.t)
  : Lemma
      (requires nr_segment_prefix original current start count logical step /\
        updated == Nr_Seq.upd current (start + step) value /\
        v_approximates value (Nr_Seq.index logical step))
      (ensures nr_segment_prefix original updated start count logical (step + 1)) =
  let outside (i:nat{i < n})
    : Lemma
        (requires i < start \/ start + (step + 1) <= i)
        (ensures Nr_Seq.index updated i == Nr_Seq.index original i) =
    assert (i <> start + step)
  in
  Nr_Classical.forall_intro (Nr_Classical.move_requires outside);
  let inside (i:nat{i < step + 1})
    : Lemma (v_approximates (Nr_Seq.index updated (start + i)) (Nr_Seq.index logical i)) =
    if i = step then assert (Nr_Seq.index updated (start + i) == value)
    else assert (Nr_Seq.index updated (start + i) == Nr_Seq.index current (start + i))
  in
  Nr_Classical.forall_intro inside

inline_for_extraction noextract
fn nr_add_segment
  (#n:erased nat)
  (left right combined:larray Nr_BF16.t n)
  (start:sz{start + 1024 <= n /\ Kuiper.SizeT.fits (start + 1024)})
  (#left_values #right_values #old:erased (lseq Nr_BF16.t n))
  (#left_logical #right_logical:erased (lseq real 1024))
  preserves gpu ** left |-> left_values ** right |-> right_values
  requires combined |-> old **
    pure (nr_bf16_segment left_values start 1024 left_logical /\
      nr_bf16_segment right_values start 1024 right_logical)
  ensures exists* (current:lseq Nr_BF16.t n). combined |-> current **
    pure (nr_same_outside_segment old current start 1024 /\
      nr_bf16_segment current start 1024 (nr_added_values left_logical right_logical))
{
  let logical : erased (lseq real 1024) = nr_added_values left_logical right_logical;
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (j:sz{j <= 1024}) (current:lseq Nr_BF16.t n).
      index |-> j ** combined |-> current **
      pure (nr_segment_prefix old current start 1024 logical j)
    decreases (1024 - !index)
  {
    let j = !index;
    let x = left.(start +^ j);
    let y = right.(start +^ j);
    let value = nr_add_bf16 x y #(Nr_Seq.index left_logical j) #(Nr_Seq.index right_logical j);
    with before. assert combined |-> before;
    combined.(start +^ j) <- value;
    with after. assert combined |-> after;
    nr_segment_progress old before after start 1024 logical j value;
    index := j +^ 1sz;
  };
  let finished = !index;
  assert pure (Kuiper.SizeT.v finished == 1024);
}

inline_for_extraction noextract
fn nr_add_norm_row
  (#n #w:erased nat)
  (left right:larray Nr_BF16.t n) (weights:larray Nr_BF16.t w)
  (combined normalized:larray Nr_BF16.t n) (buffer:larray Nr_F32.t 128)
  (start:sz{start + 1024 <= n /\ Kuiper.SizeT.fits (start + 1024)})
  (weight_start:sz{weight_start + 1024 <= w /\ Kuiper.SizeT.fits (weight_start + 1024)})
  (#left_values #right_values #old_combined #old_normalized:erased (lseq Nr_BF16.t n))
  (#weight_values:erased (lseq Nr_BF16.t w)) (#scratch:erased (lseq Nr_F32.t 128))
  (#left_logical #right_logical #weight:erased (lseq real 1024))
  preserves gpu ** left |-> left_values ** right |-> right_values ** weights |-> weight_values
  requires combined |-> old_combined ** normalized |-> old_normalized ** buffer |-> scratch **
    pure (nr_bf16_segment left_values start 1024 left_logical /\
      nr_bf16_segment right_values start 1024 right_logical /\
      nr_bf16_segment weight_values weight_start 1024 weight)
  ensures exists* (current_combined current_normalized:lseq Nr_BF16.t n)
    (scratch_values:lseq Nr_F32.t 128).
    combined |-> current_combined ** normalized |-> current_normalized ** buffer |-> scratch_values **
    pure (nr_same_outside_segment old_combined current_combined start 1024 /\
      nr_same_outside_segment old_normalized current_normalized start 1024 /\
      nr_bf16_segment current_combined start 1024 (nr_added_values left_logical right_logical) /\
      nr_bf16_segment current_normalized start 1024
        (nr_rms_values weight (nr_added_values left_logical right_logical)))
{
  nr_add_segment left right combined start #left_values #right_values #old_combined
    #left_logical #right_logical;
  with current_combined. assert combined |-> current_combined;
  nr_rms_row combined weights normalized buffer start weight_start start 1024sz
    #current_combined #weight_values #old_normalized #scratch
    #(nr_added_values left_logical right_logical) #weight #1.0R;
}

fn nr_run_add_norm
  (#n #w:erased nat)
  (left:larray Nr_BF16.t n{is_global_array left})
  (right:larray Nr_BF16.t n{is_global_array right})
  (weights:larray Nr_BF16.t w{is_global_array weights})
  (combined:larray Nr_BF16.t n{is_global_array combined})
  (normalized:larray Nr_BF16.t n{is_global_array normalized})
  (buffer:larray Nr_F32.t 128{is_global_array buffer})
  (start:sz{start + 1024 <= n /\ Kuiper.SizeT.fits (start + 1024)})
  (weight_start:sz{weight_start + 1024 <= w /\ Kuiper.SizeT.fits (weight_start + 1024)})
  (#left_values #right_values #old_combined #old_normalized:erased (lseq Nr_BF16.t n))
  (#weight_values:erased (lseq Nr_BF16.t w)) (#scratch:erased (lseq Nr_F32.t 128))
  (#left_logical #right_logical #weight:erased (lseq real 1024))
  preserves cpu ** on gpu_loc (left |-> left_values) ** on gpu_loc (right |-> right_values) **
    on gpu_loc (weights |-> weight_values)
  requires on gpu_loc (combined |-> old_combined) ** on gpu_loc (normalized |-> old_normalized) **
    on gpu_loc (buffer |-> scratch) **
    pure (nr_bf16_segment left_values start 1024 left_logical /\
      nr_bf16_segment right_values start 1024 right_logical /\
      nr_bf16_segment weight_values weight_start 1024 weight)
  ensures exists* (current_combined current_normalized:lseq Nr_BF16.t n)
    (scratch_values:lseq Nr_F32.t 128).
    on gpu_loc (combined |-> current_combined) ** on gpu_loc (normalized |-> current_normalized) **
    on gpu_loc (buffer |-> scratch_values) **
    pure (nr_same_outside_segment old_combined current_combined start 1024 /\
      nr_same_outside_segment old_normalized current_normalized start 1024 /\
      nr_bf16_segment current_combined start 1024 (nr_added_values left_logical right_logical) /\
      nr_bf16_segment current_normalized start 1024
        (nr_rms_values weight (nr_added_values left_logical right_logical)))
{
  launch_kernel_1 (fun () -> nr_add_norm_row left right weights combined normalized buffer
    start weight_start #left_values #right_values #old_combined #old_normalized
    #weight_values #scratch #left_logical #right_logical #weight);
}


(* Reused from row-probe; local declarations are namespaced. *)


open Kuiper
open Kuiper.Approximates.Base

module Sm_Shared = KPlay.SharedSpec
module Sm_F32 = Kuiper.Float32
module Sm_BF16 = Kuiper.BFloat16
module Sm_Casts = Kuiper.Float.Casts
module Sm_Native = Kuiper.Float32.FastMath
module Sm_F = Kuiper.Floating.Base
module Sm_Exp = FStar.Math.Exp
module Sm_Sum = Kuiper.Sum
module Sm_Seq = FStar.Seq

#push-options "--fuel 4 --ifuel 1 --z3rlimit 60"

noextract let sm_padded_value
  (n:pos{n<=7}) (f:Sm_Shared.vector n) (i:nat) : GTot real =
  if i < n then f i else 0.0R

let rec sm_sum_padded
  (n:pos{n<=7}) (count:nat{n<=count}) (f:Sm_Shared.vector n)
  : Lemma
      (nr_sum_n count (sm_padded_value n f) == Sm_Shared.sum n f)
      (decreases count) =
  if count = n then nr_sum_shared n f (sm_padded_value n f)
  else (
    sm_sum_padded n (count-1) f;
    nr_sum_step count (sm_padded_value n f)
  )

let sm_tree_sum (n:pos{n<=7}) (f:Sm_Shared.vector n)
  : Lemma
      (let e = sm_padded_value n f in
       (e 0 +. e 4 +. (e 2 +. e 6)) +. (e 1 +. e 5 +. (e 3 +. e 7)) ==
         Sm_Shared.sum n f) =
  let e = sm_padded_value n f in
  sm_sum_padded n 8 f;
  nr_sum_step 8 e;
  nr_sum_step 7 e;
  nr_sum_step 6 e;
  nr_sum_step 5 e;
  nr_sum_step 4 e;
  nr_sum_step 3 e;
  nr_sum_step 2 e;
  nr_sum_step 1 e

noextract let sm_shifted_sum
  (row:Sm_Shared.index 7) (logical:Sm_Shared.vector (row+1)) (maximum:real) : GTot real =
  Sm_Shared.sum (row+1) (fun i -> Sm_Exp.exp (logical i -. maximum))

noextract let sm_padded_exp
  (row:Sm_Shared.index 7) (logical:Sm_Shared.vector (row+1))
  (maximum:real) (i:nat) : GTot real =
  sm_padded_value (row+1) (fun j -> Sm_Exp.exp (logical j -. maximum)) i

let sm_denominator_approx
  (row:Sm_Shared.index 7) (logical:Sm_Shared.vector (row+1))
  (maximum:real) (value:Sm_F32.t)
  : Lemma
      (requires (
        let e = sm_padded_exp row logical maximum in
        value %~ ((e 0 +. e 4 +. (e 2 +. e 6)) +.
          (e 1 +. e 5 +. (e 3 +. e 7)))))
      (ensures value %~ sm_shifted_sum row logical maximum) =
  sm_tree_sum (row+1) (fun i -> Sm_Exp.exp (logical i -. maximum))

let sm_add_div (a b:real) (k:real{k >. 0.0R})
  : Lemma (a /. k +. b /. k == (a +. b) /. k)
  = ()

let sm_div_cancel (a:real) (b:real{b >. 0.0R}) (c:real{c >. 0.0R})
  : Lemma ((a /. c) /. (b /. c) == a /. b)
  = ()

let rec sm_sum_divide (n:nat) (f:nat -> GTot real) (k:real{k >. 0.0R})
  : Lemma
      (nr_sum_n n (fun i -> f i /. k) == nr_sum_n n f /. k)
      (decreases n) =
  if n > 0 then (
    sm_sum_divide (n-1) f k;
    nr_sum_step n f;
    nr_sum_step n (fun i -> f i /. k);
    sm_add_div (nr_sum_n (n-1) f) (f (n-1)) k
  )

#push-options "--fuel 4 --ifuel 1"
let sm_shared_softmax_at (#n:pos) (logical:Sm_Shared.vector n) (i:Sm_Shared.index n)
  : Lemma
      (requires Sm_Shared.sum n (fun j -> Sm_Exp.exp (logical j)) >. 0.0R)
      (ensures Sm_Shared.softmax logical i ==
       Sm_Exp.exp (logical i) /. Sm_Shared.sum n (fun j -> Sm_Exp.exp (logical j)))
  = assert_norm (Sm_Shared.softmax logical i ==
      Sm_Exp.exp (logical i) /. Sm_Shared.sum n (fun j -> Sm_Exp.exp (logical j)))
#pop-options

let sm_normalize_shifted
  (row:Sm_Shared.index 7) (logical:Sm_Shared.vector (row+1)) (maximum:real)
  : Lemma
      (sm_shifted_sum row logical maximum >. 0.0R /\
       (forall (i:Sm_Shared.index (row+1)).
         Sm_Exp.exp (logical i -. maximum) /. sm_shifted_sum row logical maximum ==
           Sm_Shared.prefix_softmax row logical i))
  =
  let e : Sm_Shared.vector (row+1) = fun i -> Sm_Exp.exp (logical i) in
  let shifted : Sm_Shared.vector (row+1) = fun i -> Sm_Exp.exp (logical i -. maximum) in
  Sm_Shared.sum_positive (row+1) e;
  Sm_Shared.sum_positive (row+1) shifted;
  let exp_at (i:Sm_Shared.index (row+1))
    : Lemma (shifted i == e i /. Sm_Exp.exp maximum) =
    Sm_Exp.exp_sub (logical i) maximum
  in
  FStar.Classical.forall_intro exp_at;
  shared_sum_equal (row+1) shifted (fun i -> e i /. Sm_Exp.exp maximum);
  let padded = sm_padded_value (row+1) e in
  nr_sum_shared (row+1) e padded;
  nr_sum_shared (row+1) (fun i -> e i /. Sm_Exp.exp maximum)
    (fun i -> padded i /. Sm_Exp.exp maximum);
  sm_sum_divide (row+1) padded (Sm_Exp.exp maximum);
  assert (sm_shifted_sum row logical maximum ==
    Sm_Shared.sum (row+1) e /. Sm_Exp.exp maximum);
  let at (i:Sm_Shared.index (row+1))
    : Lemma
        (Sm_Exp.exp (logical i -. maximum) /. sm_shifted_sum row logical maximum ==
          Sm_Shared.prefix_softmax row logical i)
    =
    Sm_Exp.exp_sub (logical i) maximum;
    sm_div_cancel (e i) (Sm_Shared.sum (row+1) e) (Sm_Exp.exp maximum);
    sm_shared_softmax_at logical i
  in
  FStar.Classical.forall_intro at

let sm_represents
  (#n:pos) (#t:Type0) {| scalar t, real_like t |}
  (count:pos{count<=7 /\ count<=n}) (values:lseq t 7) (logical:Sm_Shared.vector n) : prop =
  forall (i:Sm_Shared.index count).
    v_approximates (Sm_Seq.index values i) (logical i)

inline_for_extraction noextract
fn sm_shifted_exponential
  (value maximum:Sm_F32.t) (#r #m:erased real)
  preserves gpu
  requires pure (
    v_approximates value (reveal r) /\ v_approximates maximum (reveal m))
  returns result:Sm_F32.t
  ensures pure (v_approximates result (Sm_Exp.exp (reveal r -. reveal m)))
{
  let shifted = Sm_Native.sub_rn value maximum;
  exp_approx shifted (reveal r -. reveal m);
  fexp shifted
}

inline_for_extraction noextract
fn sm_exponential_at
  (inputs:larray Sm_F32.t 7) (row:szlt 7) (key:szlt 7) (maximum:Sm_F32.t)
  (#xs:erased (lseq Sm_F32.t 7)) (#logical:erased (Sm_Shared.vector (row+1)))
  (#m:erased real)
  preserves gpu ** inputs |-> xs
  requires pure (sm_represents (row+1) xs logical /\ maximum %~ reveal m)
  returns result:Sm_F32.t
  ensures pure (result %~ sm_padded_exp row logical (reveal m) key /\
    (key > row ==> result == zero #Sm_F32.t))
{
  if (key <^ (row +^ 1sz)) {
    let value = inputs.(key);
    sm_shifted_exponential value maximum #((reveal logical) key) #m
  } else {
    zero #Sm_F32.t
  }
}

inline_for_extraction noextract
fn sm_divide_probability
  (numerator denominator:Sm_F32.t) (#r:erased real) (#d:erased real{reveal d >. 0.0R})
  preserves gpu
  requires pure (
    v_approximates numerator (reveal r) /\
    v_approximates denominator (reveal d))
  returns result:Sm_BF16.t
  ensures pure (v_approximates result (reveal r /. reveal d))
{
  div_approx numerator denominator (reveal r) (reveal d);
  let quotient = div numerator denominator;
  Sm_Casts.fcast_approx #Sm_F32.t #Sm_BF16.t quotient (reveal r /. reveal d);
  Sm_Casts.fcast #Sm_F32.t #Sm_BF16.t quotient
}

noextract let sm_probability_prefix
  (row:Sm_Shared.index 7) (logical:Sm_Shared.vector (row+1))
  (values:lseq Sm_BF16.t 7) (count:nat{count<=7}) : prop =
  forall (i:Sm_Shared.index (row+1)). i < count ==>
    Seq.index values i %~ Sm_Shared.prefix_softmax row logical i

let sm_probability_write
  (row:Sm_Shared.index 7) (logical:Sm_Shared.vector (row+1))
  (values:lseq Sm_BF16.t 7) (key:Sm_Shared.index 7{key<=row}) (value:Sm_BF16.t)
  : Lemma
      (requires sm_probability_prefix row logical values key /\
        value %~ Sm_Shared.prefix_softmax row logical key)
      (ensures sm_probability_prefix row logical (Seq.upd values key value) (key+1)) =
  let point (i:Sm_Shared.index (row+1))
    : Lemma
        (requires i < key+1)
        (ensures Seq.index (Seq.upd values key value) i %~
          Sm_Shared.prefix_softmax row logical i) =
    if i = key then () else ()
  in FStar.Classical.forall_intro (FStar.Classical.move_requires point)

inline_for_extraction noextract
fn sm_put_probability
  (outputs:larray Sm_BF16.t 7) (row key:szlt 7)
  (numerator denominator:Sm_F32.t)
  (#old:erased (lseq Sm_BF16.t 7))
  (#logical:erased (Sm_Shared.vector (row+1))) (#m:erased real)
  preserves gpu
  requires outputs |-> old **
    pure (sm_probability_prefix row logical old key /\
      numerator %~ sm_padded_exp row logical (reveal m) key /\
      denominator %~ sm_shifted_sum row logical (reveal m))
  ensures exists* (result:lseq Sm_BF16.t 7).
    outputs |-> result ** pure (sm_probability_prefix row logical result (key+1))
{
  if (key <^ (row +^ 1sz)) {
    sm_normalize_shifted row logical (reveal m);
    let value = sm_divide_probability numerator denominator
      #(Sm_Exp.exp ((reveal logical) key -. reveal m))
      #(sm_shifted_sum row logical (reveal m));
    sm_probability_write row logical old key value;
    outputs.(key) <- value;
  } else {
    ();
  };
}

inline_for_extraction noextract
fn sm_row_kernel
  (inputs:larray Sm_F32.t 7) (outputs:larray Sm_BF16.t 7) (row:szlt 7)
  (#xs:erased (lseq Sm_F32.t 7)) (#old:erased (lseq Sm_BF16.t 7))
  (#logical:erased (Sm_Shared.vector (row+1)))
  preserves gpu ** inputs |-> xs
  requires outputs |-> old ** pure (sm_represents (row+1) xs logical)
  ensures exists* (result:lseq Sm_BF16.t 7).
    outputs |-> result **
    pure (sm_represents (row+1) result (Sm_Shared.prefix_softmax row logical))
{
  let s0 = inputs.(0sz);
  let mut current_maximum = s0;
  let mut index = 1sz;
  while (!index <^ (row +^ 1sz))
    invariant exists* (key:sz{1<=key /\ key<=row+1}) (m:Sm_F32.t).
      index |-> key ** current_maximum |-> m
    decreases (row+1 - !index)
  {
    let key = !index;
    let value = inputs.(key);
    current_maximum := fmax !current_maximum value;
    index := key +^ 1sz;
  };
  let maximum = !current_maximum;
  to_real_ok maximum;
  let maximum_real : erased real = to_real maximum;
  let e0 = sm_exponential_at inputs row 0sz maximum #xs #logical #maximum_real;
  let e1 = sm_exponential_at inputs row 1sz maximum #xs #logical #maximum_real;
  let e2 = sm_exponential_at inputs row 2sz maximum #xs #logical #maximum_real;
  let e3 = sm_exponential_at inputs row 3sz maximum #xs #logical #maximum_real;
  let e4 = sm_exponential_at inputs row 4sz maximum #xs #logical #maximum_real;
  let e5 = sm_exponential_at inputs row 5sz maximum #xs #logical #maximum_real;
  let e6 = sm_exponential_at inputs row 6sz maximum #xs #logical #maximum_real;
  let e7:Sm_F32.t = zero #Sm_F32.t;
  let er : erased (nat -> GTot real) = sm_padded_exp row logical (reveal maximum_real);
  a_add e0 e4 ((reveal er) 0) ((reveal er) 4);
  a_add e1 e5 ((reveal er) 1) ((reveal er) 5);
  a_add e2 e6 ((reveal er) 2) ((reveal er) 6);
  a_add e3 e7 ((reveal er) 3) 0.0R;
  let part0 = add e0 e4;
  let part1 = add e1 e5;
  let part2 = add e2 e6;
  let part3 = add e3 e7;
  a_add part0 part2
    ((reveal er) 0 +. (reveal er) 4) ((reveal er) 2 +. (reveal er) 6);
  a_add part1 part3
    ((reveal er) 1 +. (reveal er) 5) ((reveal er) 3 +. (reveal er) 7);
  let low = add part0 part2;
  let high = add part1 part3;
  a_add low high
    ((reveal er) 0 +. (reveal er) 4 +. ((reveal er) 2 +. (reveal er) 6))
    ((reveal er) 1 +. (reveal er) 5 +. ((reveal er) 3 +. (reveal er) 7));
  let denominator = add low high;
  sm_denominator_approx row logical (reveal maximum_real) denominator;
  sm_put_probability outputs row 0sz e0 denominator #old #logical #maximum_real;
  with p1. assert outputs |-> p1;
  sm_put_probability outputs row 1sz e1 denominator #p1 #logical #maximum_real;
  with p2. assert outputs |-> p2;
  sm_put_probability outputs row 2sz e2 denominator #p2 #logical #maximum_real;
  with p3. assert outputs |-> p3;
  sm_put_probability outputs row 3sz e3 denominator #p3 #logical #maximum_real;
  with p4. assert outputs |-> p4;
  sm_put_probability outputs row 4sz e4 denominator #p4 #logical #maximum_real;
  with p5. assert outputs |-> p5;
  sm_put_probability outputs row 5sz e5 denominator #p5 #logical #maximum_real;
  with p6. assert outputs |-> p6;
  sm_put_probability outputs row 6sz e6 denominator #p6 #logical #maximum_real;
}

instance sm_send_global_array_contents
  (#a:Type0) (arr:array a{is_global_array arr}) (#f:perm) (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn sm_run_row
  (inputs:larray Sm_F32.t 7{is_global_array inputs})
  (outputs:larray Sm_BF16.t 7{is_global_array outputs})
  (row:szlt 7)
  (#xs:erased (lseq Sm_F32.t 7))
  (#logical:erased (Sm_Shared.vector (row+1)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires (exists* (old:lseq Sm_BF16.t 7). on gpu_loc (outputs |-> old)) **
    pure (sm_represents (row+1) xs logical)
  ensures exists* (result:lseq Sm_BF16.t 7).
    on gpu_loc (outputs |-> result) **
    pure (sm_represents (row+1) result (Sm_Shared.prefix_softmax row logical))
{
  with old. assert on gpu_loc (outputs |-> old);
  launch_kernel_1 (fun () -> sm_row_kernel inputs outputs row #xs #old #logical);
}

#pop-options

(* Reused from linear-stage-probe; local declarations are namespaced. *)


#set-options "--fuel 4 --ifuel 2 --z3rlimit 60"

open Kuiper
open FStar.Ghost
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }
open Pulse.Lib.Trade

module Mm_Seq = FStar.Seq
module Mm_SZ = Kuiper.SizeT
module Mm_C = Kuiper.Chest
module Mm_T = Kuiper.Tensor
module Mm_MMA = Kuiper.TensorCore.MMA
module Mm_G = Kuiper.Spec.GEMM

inline_for_extraction noextract
instance mm_c2048 : concrete_sz 2048 = { x = 2048sz }

noextract let rec mm_sum_n (n:nat) (f:nat -> GTot real)
  : GTot real (decreases n) =
  if n = 0 then 0.0R else mm_sum_n (n-1) f +. f (n-1)

let rec mm_sum_ext (n:nat) (f g:nat -> GTot real)
  : Lemma
      (requires forall (i:natlt n). f i == g i)
      (ensures mm_sum_n n f == mm_sum_n n g)
      (decreases n) =
  if n > 0 then mm_sum_ext (n-1) f g

let rec mm_sum_append (n m:nat) (f:nat -> GTot real)
  : Lemma (mm_sum_n (n+m) f == mm_sum_n n f +. mm_sum_n m (fun i -> f (n+i)))
      (decreases m) =
  if m > 0 then mm_sum_append n (m-1) f

let mm_sum_step (n:pos) (f:nat -> GTot real)
  : Lemma (mm_sum_n n f == mm_sum_n (n-1) f +. f (n-1)) = ()

inline_for_extraction noextract
let mm_a_layout = rm 16 2048

inline_for_extraction noextract
let mm_b_layout = cm 2048 8

inline_for_extraction noextract
let mm_c_layout = rm 16 8

inline_for_extraction noextract
instance mm_a_concrete : ctlayout mm_a_layout =
  Kuiper.Tensor.Layout.Alg.c_l2_row_major 16 2048sz

inline_for_extraction noextract
instance mm_b_concrete : ctlayout mm_b_layout =
  Kuiper.Tensor.Layout.Alg.c_l2_col_major 2048sz 8

inline_for_extraction noextract
instance mm_c_concrete : ctlayout mm_c_layout =
  Kuiper.Tensor.Layout.Alg.c_l2_row_major 16 8sz

inline_for_extraction noextract
let rec mm_mma_accum
  (a:chest2 bf16 16 2048) (b:chest2 bf16 2048 8)
  (count:nat{count<=128})
  : GTot (chest2 float 16 8) (decreases count) =
  if count = 0 then Mm_C.const (16 @| 8 @| INil) (zero #float)
  else Mm_MMA.emma (mm_mma_accum a b (count-1))
    (ematrix_subtile a 16 16 0 (count-1))
    (ematrix_subtile b 16 8 (count-1) 0)

let mm_mma_accum_zero (a:chest2 bf16 16 2048) (b:chest2 bf16 2048 8)
  : Lemma (mm_mma_accum a b 0 == Mm_C.const (16 @| 8 @| INil) (zero #float)) = ()

let mm_mma_accum_step
  (a:chest2 bf16 16 2048) (b:chest2 bf16 2048 8)
  (k:natlt 128)
  : Lemma (mm_mma_accum a b (k+1) ==
      Mm_MMA.emma (mm_mma_accum a b k)
        (ematrix_subtile a 16 16 0 k)
        (ematrix_subtile b 16 8 k 0)) = ()

noextract let mm_matrix_term
  (a:chest2 real 16 16) (b:chest2 real 16 8)
  (row:natlt 16) (col:natlt 8) (k:nat) : GTot real =
  if k < 16 then Mm_C.acc2 a row k *. Mm_C.acc2 b k col else 0.0R

let rec mm_matmul_sum
  (a:chest2 real 16 16) (b:chest2 real 16 8)
  (row:natlt 16) (col:natlt 8) (count:nat{count<=16})
  : Lemma (Mm_G.__matmul_single a b row col count ==
      mm_sum_n count (mm_matrix_term a b row col))
      (decreases count) =
  if count = 0 then Mm_G.matmul_zero_lemma a b row col
  else (
    mm_matmul_sum a b row col (count-1);
    Mm_G.matmul_single_lemma a b row col count;
    mm_sum_step count (mm_matrix_term a b row col)
  )

noextract let mm_dot_term (a b:lseq real 16) (k:nat) : Tot real =
  if k < 16 then Mm_Seq.index a k *. Mm_Seq.index b k else 0.0R

let mm_mma_cell_approx
  (a:chest2 bf16 16 16) (b:chest2 bf16 16 8)
  (c:chest2 float 16 8)
  (row:natlt 16) (col:natlt 8)
  (ar br:lseq real 16) (cr:real)
  : Lemma
      (requires
        (forall (k:natlt 16). Mm_C.acc2 a row k %~ Mm_Seq.index ar k) /\
        (forall (k:natlt 16). Mm_C.acc2 b k col %~ Mm_Seq.index br k) /\
        Mm_C.acc2 c row col %~ cr)
      (ensures Mm_C.acc2 (Mm_MMA.emma c a b) row col %~
        (cr +. mm_sum_n 16 (mm_dot_term ar br))) =
  let ra = Mm_C.mk2 (fun i k ->
    if i = row then Mm_Seq.index ar k else to_real (Mm_C.acc2 a i k)) in
  let rb = Mm_C.mk2 (fun k j ->
    if j = col then Mm_Seq.index br k else to_real (Mm_C.acc2 b k j)) in
  let rc = Mm_C.mk2 (fun i j ->
    if i = row && j = col then cr else to_real (Mm_C.acc2 c i j)) in
  Mm_C.lemma_to_real_chest_approximates a;
  Mm_C.lemma_to_real_chest_approximates b;
  Mm_C.lemma_to_real_chest_approximates c;
  assert (a %~ ra /\ b %~ rb /\ c %~ rc);
  Mm_MMA.emma_approx_lemma c a b rc ra rb;
  Mm_G.lemma_matmul_index ra rb row col;
  mm_matmul_sum ra rb row col 16;
  mm_sum_ext 16 (mm_matrix_term ra rb row col) (mm_dot_term ar br);
  assert (Mm_C.acc2 (Mm_G.matplus rc (Mm_G.matmul ra rb)) row col ==
    cr +. mm_sum_n 16 (mm_dot_term ar br))

noextract let mm_product_term
  (a:chest2 real 16 2048) (b:chest2 real 2048 8)
  (row:natlt 16) (col:natlt 8) (k:nat) : GTot real =
  if k < 2048 then Mm_C.acc2 a row k *. Mm_C.acc2 b k col else 0.0R

let rec mm_mma_accum_cell
  (a:chest2 bf16 16 2048) (b:chest2 bf16 2048 8)
  (ar:chest2 real 16 2048) (br:chest2 real 2048 8)
  (row:natlt 16) (col:natlt 8) (count:nat{count<=128})
  : Lemma
      (requires a %~ ar /\ b %~ br)
      (ensures Mm_C.acc2 (mm_mma_accum a b count) row col %~
        mm_sum_n (16*count) (mm_product_term ar br row col))
      (decreases count) =
  if count = 0 then mm_mma_accum_zero a b
  else (
    let k = count-1 in
    let ra = Mm_Seq.init_ghost 16 (fun i -> Mm_C.acc2 ar row (16*k+i)) in
    let rb = Mm_Seq.init_ghost 16 (fun i -> Mm_C.acc2 br (16*k+i) col) in
    mm_mma_accum_cell a b ar br row col k;
    mm_mma_cell_approx
      (ematrix_subtile a 16 16 0 k) (ematrix_subtile b 16 8 k 0)
      (mm_mma_accum a b k) row col ra rb
      (mm_sum_n (16*k) (mm_product_term ar br row col));
    mm_sum_ext 16 (mm_dot_term ra rb) (fun i -> mm_product_term ar br row col (16*k+i));
    mm_sum_append (16*k) 16 (mm_product_term ar br row col);
    mm_mma_accum_step a b k
  )

inline_for_extraction noextract
fn mm_mma_kernel
  (a:array2 bf16 mm_a_layout) (b:array2 bf16 mm_b_layout) (c:array2 float mm_c_layout)
  (count:sz{count<=128})
  (#av:chest2 bf16 16 2048) (#bv:chest2 bf16 2048 8) (#cv:chest2 float 16 8)
  preserves gpu **
    a |-> Frac (1.0R /. 32.0R) av **
    b |-> Frac (1.0R /. 32.0R) bv
  requires c |-> Frac (1.0R /. 32.0R) cv
  ensures c |-> Frac (1.0R /. 32.0R) (mm_mma_accum av bv count)
{
  let fr = Mm_MMA.alloc_fragment ();
  Mm_MMA.fill fr (zero #float);
  mm_mma_accum_zero av bv;
  let mut i = 0sz;
  while (!i <^ count)
    invariant exists* (k:sz{k<=count}).
      i |-> k ** fr |-> mm_mma_accum av bv k
    decreases (count - !i)
  {
    let k = !i;
    let at = array2_extract_tile_ro' a 16 16 0 (Mm_SZ.v k);
    let bt = array2_extract_tile_ro' b 16 8 (Mm_SZ.v k) 0;
    Mm_MMA.mma_sync at bt fr;
    elim_trade
      (at |-> Frac (1.0R /. 32.0R) (ematrix_subtile av 16 16 0 k))
      (a |-> Frac (1.0R /. 32.0R) av);
    elim_trade
      (bt |-> Frac (1.0R /. 32.0R) (ematrix_subtile bv 16 8 k 0))
      (b |-> Frac (1.0R /. 32.0R) bv);
    mm_mma_accum_step av bv k;
    i := k +^ 1sz;
  };
  Mm_MMA.store fr c;
  drop_ (fr |-> mm_mma_accum av bv count);
}

inline_for_extraction noextract
fn mm_mma_thread
  (a:array2 bf16 mm_a_layout) (b:array2 bf16 mm_b_layout) (c:array2 float mm_c_layout)
  (count:sz{count<=128})
  (#av:chest2 bf16 16 2048) (#bv:chest2 bf16 2048 8) (#cv:chest2 float 16 8)
  (tid:szlt 32) ()
  norewrite
  requires gpu **
    (a |-> Frac (1.0R /. 32.0R) av **
     b |-> Frac (1.0R /. 32.0R) bv **
     c |-> Frac (1.0R /. 32.0R) cv) ** thread_id 32 tid
  ensures gpu **
    (a |-> Frac (1.0R /. 32.0R) av **
     b |-> Frac (1.0R /. 32.0R) bv **
     c |-> Frac (1.0R /. 32.0R) (mm_mma_accum av bv count)) ** thread_id 32 tid
{
  mm_mma_kernel a b c count #av #bv #cv;
}

ghost
fn mm_mma_setup
  (a:array2 bf16 mm_a_layout) (b:array2 bf16 mm_b_layout) (c:array2 float mm_c_layout)
  (#av:chest2 bf16 16 2048) (#bv:chest2 bf16 2048 8) (#cv:chest2 float 16 8)
  ()
  norewrite
  requires a |-> av ** b |-> bv ** c |-> cv
  ensures (forall+ (tid:natlt 32).
    a |-> Frac (1.0R /. 32.0R) av **
    b |-> Frac (1.0R /. 32.0R) bv **
    c |-> Frac (1.0R /. 32.0R) cv) ** emp
{
  Mm_T.tensor_share_n a 32;
  Mm_T.tensor_share_n b 32;
  Mm_T.tensor_share_n c 32;
  forevery_zip3
    (fun (_:natlt 32) -> a |-> Frac (1.0R /. 32.0R) av)
    (fun (_:natlt 32) -> b |-> Frac (1.0R /. 32.0R) bv)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) cv);
}

ghost
fn mm_mma_teardown
  (a:array2 bf16 mm_a_layout) (b:array2 bf16 mm_b_layout) (c:array2 float mm_c_layout)
  (count:sz{count<=128})
  (#av:chest2 bf16 16 2048) (#bv:chest2 bf16 2048 8)
  ()
  norewrite
  requires (forall+ (tid:natlt 32).
    a |-> Frac (1.0R /. 32.0R) av **
    b |-> Frac (1.0R /. 32.0R) bv **
    c |-> Frac (1.0R /. 32.0R) (mm_mma_accum av bv count)) ** emp
  ensures a |-> av ** b |-> bv ** c |-> mm_mma_accum av bv count
{
  forevery_unzip3
    (fun (_:natlt 32) -> a |-> Frac (1.0R /. 32.0R) av)
    (fun (_:natlt 32) -> b |-> Frac (1.0R /. 32.0R) bv)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) (mm_mma_accum av bv count));
  Mm_T.tensor_gather_n a 32;
  Mm_T.tensor_gather_n b 32;
  Mm_T.tensor_gather_n c 32;
}

inline_for_extraction noextract
let mm_mma_desc
  (a:array2 bf16 mm_a_layout{Mm_T.is_global a})
  (b:array2 bf16 mm_b_layout{Mm_T.is_global b})
  (c:array2 float mm_c_layout{Mm_T.is_global c})
  (count:sz{count<=128})
  (#av:chest2 bf16 16 2048) (#bv:chest2 bf16 2048 8) (#cv:chest2 float 16 8)
  : kernel_desc_1_n
      (a |-> av ** b |-> bv ** c |-> cv)
      (a |-> av ** b |-> bv ** c |-> mm_mma_accum av bv count) =
  {
    nthr = 32sz;
    frame = emp;
    kpre = (fun _ ->
      a |-> Frac (1.0R /. 32.0R) av **
      b |-> Frac (1.0R /. 32.0R) bv **
      c |-> Frac (1.0R /. 32.0R) cv);
    kpost = (fun _ ->
      a |-> Frac (1.0R /. 32.0R) av **
      b |-> Frac (1.0R /. 32.0R) bv **
      c |-> Frac (1.0R /. 32.0R) (mm_mma_accum av bv count));
    f = mm_mma_thread a b c count #av #bv #cv;
    block_setup = mm_mma_setup a b c #av #bv #cv;
    block_teardown = mm_mma_teardown a b c count #av #bv;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve
  }

fn mm_run_mma
  (a:array2 bf16 mm_a_layout{Mm_T.is_global a})
  (b:array2 bf16 mm_b_layout{Mm_T.is_global b})
  (c:array2 float mm_c_layout{Mm_T.is_global c})
  (count:sz{count<=128})
  (#av:chest2 bf16 16 2048) (#bv:chest2 bf16 2048 8) (#cv:chest2 float 16 8)
  preserves cpu **
    on gpu_loc (a |-> av) **
    on gpu_loc (b |-> bv)
  requires on gpu_loc (c |-> cv)
  ensures on gpu_loc (c |-> mm_mma_accum av bv count)
{
  launch_sync (mm_mma_desc a b c count #av #bv #cv);
}

module Mm_Shared = KPlay.SharedSpec
module Mm_Sum = Kuiper.Sum
module Mm_Cast = Kuiper.Float.Casts.Base
module Mm_Native = Kuiper.Float32.FastMath

inline_for_extraction noeq
type mm_linear_parameters = {
  width: szp;
  columns: szp;
  splits: szp
}

noextract let mm_linear_valid (p:mm_linear_parameters) : prop =
  (Mm_SZ.v p.width) <= 3072 /\ (Mm_SZ.v p.columns) <= 6144 /\ (Mm_SZ.v p.columns) % 16 = 0 /\
  ((Mm_SZ.v p.splits) = 1 \/ (Mm_SZ.v p.splits) = 2) /\
  (Mm_SZ.v p.width) % (Mm_SZ.v p.splits) = 0 /\
  (Mm_SZ.v p.width) / (Mm_SZ.v p.splits) <= 2048 /\
  ((Mm_SZ.v p.width) / (Mm_SZ.v p.splits)) % 16 = 0

noextract let mm_chunk (p:mm_linear_parameters{mm_linear_valid p}) : pos =
  (Mm_SZ.v p.width) / (Mm_SZ.v p.splits)

noextract let mm_a_values
  (p:mm_linear_parameters{mm_linear_valid p})
  (#n:nat) (weights:lseq bf16 n)
  (offset:nat{offset+(Mm_SZ.v p.columns)*(Mm_SZ.v p.width)<=n})
  (tile:nat{tile*16+16<=(Mm_SZ.v p.columns)}) (part:natlt (Mm_SZ.v p.splits))
  : GTot (chest2 bf16 16 2048) =
  Mm_C.mk2 (fun r d ->
    if d < mm_chunk p then
      Mm_Seq.index weights (offset+(tile*16+r)*(Mm_SZ.v p.width)+part*mm_chunk p+d)
    else zero #bf16)

noextract let mm_b_values
  (p:mm_linear_parameters{mm_linear_valid p})
  (input:lseq bf16 (7*(Mm_SZ.v p.width))) (part:natlt (Mm_SZ.v p.splits))
  : GTot (chest2 bf16 2048 8) =
  Mm_C.mk2 (fun d token ->
    if token < 7 && d < mm_chunk p then
      Mm_Seq.index input (token*(Mm_SZ.v p.width)+part*mm_chunk p+d)
    else zero #bf16)

noextract let mm_a_real
  (p:mm_linear_parameters{mm_linear_valid p})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width))
  (tile:nat{tile*16+16<=(Mm_SZ.v p.columns)}) (part:natlt (Mm_SZ.v p.splits))
  : GTot (chest2 real 16 2048) =
  Mm_C.mk2 (fun r d ->
    if d < mm_chunk p then weights (tile*16+r) (part*mm_chunk p+d) else 0.0R)

noextract let mm_b_real
  (p:mm_linear_parameters{mm_linear_valid p})
  (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width)) (part:natlt (Mm_SZ.v p.splits))
  : GTot (chest2 real 2048 8) =
  Mm_C.mk2 (fun d token ->
    if token < 7 && d < mm_chunk p then input token (part*mm_chunk p+d) else 0.0R)

noextract let mm_weights_rep
  (p:mm_linear_parameters{mm_linear_valid p})
  (#n:nat) (values:lseq bf16 n)
  (offset:nat{offset+(Mm_SZ.v p.columns)*(Mm_SZ.v p.width)<=n})
  (logical:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) : prop =
  forall (r:natlt (Mm_SZ.v p.columns)) (d:natlt (Mm_SZ.v p.width)).
    Mm_Seq.index values (offset+r*(Mm_SZ.v p.width)+d) %~ logical r d

noextract let mm_input_rep
  (p:mm_linear_parameters{mm_linear_valid p})
  (values:lseq bf16 (7*(Mm_SZ.v p.width))) (logical:Mm_Shared.matrix 7 (Mm_SZ.v p.width)) : prop =
  forall (token:natlt 7) (d:natlt (Mm_SZ.v p.width)).
    Mm_Seq.index values (token*(Mm_SZ.v p.width)+d) %~ logical token d

let mm_operands_approx
  (p:mm_linear_parameters{mm_linear_valid p})
  (#n:nat) (weights:lseq bf16 n) (input:lseq bf16 (7*(Mm_SZ.v p.width)))
  (offset:nat{offset+(Mm_SZ.v p.columns)*(Mm_SZ.v p.width)<=n})
  (tile:nat{tile*16+16<=(Mm_SZ.v p.columns)}) (part:natlt (Mm_SZ.v p.splits))
  (wr:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (xr:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  : Lemma
      (requires mm_weights_rep p weights offset wr /\ mm_input_rep p input xr)
      (ensures mm_a_values p weights offset tile part %~ mm_a_real p wr tile part /\
        mm_b_values p input part %~ mm_b_real p xr part) = ()

inline_for_extraction noextract
fn mm_stage_operands
  (p:mm_linear_parameters{mm_linear_valid p})
  (#n:erased nat{n<=2147483647})
  (weights:larray bf16 n) (input:larray bf16 (7*(Mm_SZ.v p.width)))
  (offset:sz{offset+(Mm_SZ.v p.columns)*(Mm_SZ.v p.width)<=n})
  (tile:sz{tile*16+16<=(Mm_SZ.v p.columns)}) (part:szlt (Mm_SZ.v p.splits))
  (a:array2 bf16 mm_a_layout) (b:array2 bf16 mm_b_layout)
  (#wv:erased (lseq bf16 n)) (#xv:erased (lseq bf16 (7*(Mm_SZ.v p.width))))
  (#av:chest2 bf16 16 2048) (#bv:chest2 bf16 2048 8)
  (#wp #xp:perm)
  preserves gpu ** weights |-> Frac wp wv ** input |-> Frac xp xv
  requires a |-> av ** b |-> bv
  ensures a |-> mm_a_values p (reveal wv) offset tile part **
    b |-> mm_b_values p (reveal xv) part
{
  open Pulse.Lib.Array;
  let inner = p.width /^ p.splits;
  let start = part *^ inner;
  let mut i = 0sz;
  while (!i <^ 32768sz)
    invariant exists* (j:sz{j<=32768}) (values:chest2 bf16 16 2048).
      i |-> j ** a |-> values **
      pure (forall (r:natlt 16) (d:natlt 2048). r*2048+d < j ==>
        Mm_C.acc2 values r d == Mm_C.acc2 (mm_a_values p (reveal wv) offset tile part) r d)
    decreases (32768 - !i)
  {
    let j = !i;
    let r = j /^ 2048sz;
    let d = j %^ 2048sz;
    assert pure (Mm_SZ.v tile*16+Mm_SZ.v r < Mm_SZ.v p.columns);
    assert pure (Mm_SZ.v d < Mm_SZ.v inner ==>
      Mm_SZ.v start+Mm_SZ.v d < Mm_SZ.v p.width /\
      Mm_SZ.v offset+(Mm_SZ.v tile*16+Mm_SZ.v r)*Mm_SZ.v p.width+
        Mm_SZ.v start+Mm_SZ.v d < n);
    if (d <^ inner) {
      let value = weights.(offset +^ (tile *^ 16sz +^ r) *^ p.width +^ start +^ d);
      Mm_T.tensor_write a (r, (d, ())) value;
    } else {
      Mm_T.tensor_write a (r, (d, ())) (zero #bf16);
    };
    i := j +^ 1sz;
  };
  with values. assert a |-> values;
  assert pure (Mm_C.equal values (mm_a_values p (reveal wv) offset tile part));
  let mut j = 0sz;
  while (!j <^ 16384sz)
    invariant exists* (k:sz{k<=16384}) (values:chest2 bf16 2048 8).
      j |-> k ** b |-> values **
      pure (forall (token:natlt 8) (d:natlt 2048). token*2048+d < k ==>
        Mm_C.acc2 values d token == Mm_C.acc2 (mm_b_values p (reveal xv) part) d token)
    decreases (16384 - !j)
  {
    let k = !j;
    let token = k /^ 2048sz;
    let d = k %^ 2048sz;
    assert pure (Mm_SZ.v d < Mm_SZ.v inner ==>
      Mm_SZ.v start+Mm_SZ.v d < Mm_SZ.v p.width);
    if (token <^ 7sz && d <^ inner) {
      let value = input.(token *^ p.width +^ start +^ d);
      Mm_T.tensor_write b (d, (token, ())) value;
    } else {
      Mm_T.tensor_write b (d, (token, ())) (zero #bf16);
    };
    j := k +^ 1sz;
  };
  with values. assert b |-> values;
  assert pure (Mm_C.equal values (mm_b_values p (reveal xv) part));
}

noextract let mm_segment_term
  (p:mm_linear_parameters{mm_linear_valid p})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (column:natlt (Mm_SZ.v p.columns)) (token:natlt 7)
  (part:natlt (Mm_SZ.v p.splits)) (d:nat) : GTot real =
  if d < mm_chunk p then
    input token (part*mm_chunk p+d) *. weights column (part*mm_chunk p+d)
  else 0.0R

noextract let mm_segment_result
  (p:mm_linear_parameters{mm_linear_valid p})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (tile:nat{tile*16+16<=(Mm_SZ.v p.columns)}) (part:natlt (Mm_SZ.v p.splits))
  (values:chest2 float 16 8) : prop =
  forall (r:natlt 16) (token:natlt 7).
    Mm_C.acc2 values r token %~
      mm_sum_n (mm_chunk p) (mm_segment_term p weights input (tile*16+r) token part)

let mm_segment_from_mma
  (p:mm_linear_parameters{mm_linear_valid p})
  (a:chest2 bf16 16 2048) (b:chest2 bf16 2048 8)
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (tile:nat{tile*16+16<=(Mm_SZ.v p.columns)}) (part:natlt (Mm_SZ.v p.splits))
  : Lemma
      (requires a %~ mm_a_real p weights tile part /\ b %~ mm_b_real p input part)
      (ensures mm_segment_result p weights input tile part
        (mm_mma_accum a b (mm_chunk p / 16))) =
  let point (r:natlt 16) (token:natlt 7)
    : Lemma (Mm_C.acc2 (mm_mma_accum a b (mm_chunk p / 16)) r token %~
        mm_sum_n (mm_chunk p) (mm_segment_term p weights input (tile*16+r) token part)) =
    mm_mma_accum_cell a b (mm_a_real p weights tile part) (mm_b_real p input part)
      r token (mm_chunk p / 16);
    mm_sum_ext (mm_chunk p)
      (mm_product_term (mm_a_real p weights tile part) (mm_b_real p input part) r token)
      (mm_segment_term p weights input (tile*16+r) token part)
  in FStar.Classical.forall_intro_2 point

noextract let mm_full_term
  (p:mm_linear_parameters{mm_linear_valid p})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (column:natlt (Mm_SZ.v p.columns)) (token:natlt 7) (d:nat) : GTot real =
  if d < (Mm_SZ.v p.width) then input token d *. weights column d else 0.0R

let rec mm_sum_shared (n:nat) (f:nat -> GTot real)
  : Lemma (mm_sum_n n f == Mm_Sum.sum #real #Mm_Sum.real_add_semigroup 0 n f)
      (decreases n) =
  if n > 0 then (
    mm_sum_shared (n-1) f;
    Mm_Sum.sum_pop_right #real #Mm_Sum.real_add_semigroup 0 n f
  )

let rec mm_shared_sum_ext (n:nat) (f g:Mm_Shared.vector n)
  : Lemma
      (requires forall (i:natlt n). f i == g i)
      (ensures Mm_Shared.sum n f == Mm_Shared.sum n g)
      (decreases n) =
  if n > 0 then (
    mm_shared_sum_ext (n-1) f g;
    Mm_Sum.sum_pop_right #real #Mm_Sum.real_add_semigroup 0 n f;
    Mm_Sum.sum_pop_right #real #Mm_Sum.real_add_semigroup 0 n g
  )

let mm_full_term_shared
  (p:mm_linear_parameters{mm_linear_valid p})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (column:natlt (Mm_SZ.v p.columns)) (token:natlt 7)
  : Lemma (mm_sum_n (Mm_SZ.v p.width) (mm_full_term p weights input column token) ==
      Mm_Shared.linear_rows weights input token column) =
  mm_sum_shared (Mm_SZ.v p.width) (mm_full_term p weights input column token);
  mm_shared_sum_ext (Mm_SZ.v p.width)
    (mm_full_term p weights input column token)
    (fun d -> input token d *. weights column d);
  assert (Mm_Shared.linear_rows weights input token column ==
    Mm_Shared.sum (Mm_SZ.v p.width) (fun d -> input token d *. weights column d)) by
    (FStar.Tactics.V2.norm
      [delta_only [`%Mm_Shared.linear_rows; `%Mm_Shared.linear; `%Mm_Shared.dot]];
     FStar.Tactics.V2.trivial ())

noextract let mm_tile_result
  (p:mm_linear_parameters{mm_linear_valid p})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (tile:nat{tile*16+16<=(Mm_SZ.v p.columns)}) (values:chest2 float 16 8) : prop =
  forall (r:natlt 16) (token:natlt 7).
    Mm_C.acc2 values r token %~ Mm_Shared.linear_rows weights input token (tile*16+r)

let mm_single_segment
  (p:mm_linear_parameters{mm_linear_valid p /\ (Mm_SZ.v p.splits)=1})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (tile:nat{tile*16+16<=(Mm_SZ.v p.columns)}) (values:chest2 float 16 8)
  : Lemma
      (requires mm_segment_result p weights input tile 0 values)
      (ensures mm_tile_result p weights input tile values) =
  let point (r:natlt 16) (token:natlt 7)
    : Lemma (Mm_C.acc2 values r token %~
        Mm_Shared.linear_rows weights input token (tile*16+r)) =
    mm_sum_ext (Mm_SZ.v p.width) (mm_segment_term p weights input (tile*16+r) token 0)
      (mm_full_term p weights input (tile*16+r) token);
    mm_full_term_shared p weights input (tile*16+r) token
  in FStar.Classical.forall_intro_2 point

let mm_paired_segments
  (p:mm_linear_parameters{mm_linear_valid p /\ (Mm_SZ.v p.splits)=2})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (column:natlt (Mm_SZ.v p.columns)) (token:natlt 7)
  : Lemma (
      mm_sum_n (mm_chunk p) (mm_segment_term p weights input column token 0) +.
      mm_sum_n (mm_chunk p) (mm_segment_term p weights input column token 1) ==
      Mm_Shared.linear_rows weights input token column) =
  mm_sum_ext (mm_chunk p) (mm_segment_term p weights input column token 0)
    (mm_full_term p weights input column token);
  mm_sum_ext (mm_chunk p) (mm_segment_term p weights input column token 1)
    (fun d -> mm_full_term p weights input column token (mm_chunk p+d));
  mm_sum_append (mm_chunk p) (mm_chunk p) (mm_full_term p weights input column token);
  mm_full_term_shared p weights input column token

let mm_fma_at (x y z result:float) (xr yr zr:real)
  : Lemma
      (requires
        (forall (a b c:real).
          x %~ a /\ y %~ b /\ z %~ c ==> result %~ (a *. b +. c)) /\
        x %~ xr /\ y %~ yr /\ z %~ zr)
      (ensures result %~ (xr *. yr +. zr)) = ()

inline_for_extraction noextract
fn mm_rn_add (x y:float) (#xr #yr:erased real)
  preserves gpu
  requires pure (x %~ reveal xr /\ y %~ reveal yr)
  returns result:float
  ensures pure (result %~ (reveal xr +. reveal yr))
{
  let one_ok : squash (one #float %~ 1.0R) = a1 #float;
  let result = Mm_Native.fma_rn x one y;
  mm_fma_at x (one #float) y result (reveal xr) 1.0R (reveal yr);
  result
}

noextract let mm_split_progress
  (p:mm_linear_parameters{mm_linear_valid p})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width))
  (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (tile:nat{tile*16+16<=Mm_SZ.v p.columns})
  (original values:chest2 float 16 8) (count:nat{count<=112}) : prop =
  forall (r:natlt 16) (token:natlt 7).
    (token*16+r < count ==>
      Mm_C.acc2 values r token %~ Mm_Shared.linear_rows weights input token (tile*16+r)) /\
    (token*16+r >= count ==> Mm_C.acc2 values r token == Mm_C.acc2 original r token)

let mm_split_write
  (p:mm_linear_parameters{mm_linear_valid p})
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width))
  (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (tile:nat{tile*16+16<=Mm_SZ.v p.columns})
  (original values:chest2 float 16 8) (i:natlt 112) (value:float)
  : Lemma
      (requires mm_split_progress p weights input tile original values i /\
        value %~ Mm_Shared.linear_rows weights input (i/16) (tile*16+i%16))
      (ensures mm_split_progress p weights input tile original
        (Mm_C.upd values (i%16, (i/16, ())) value) (i+1)) =
  let point (r:natlt 16) (token:natlt 7)
    : Lemma (
        (token*16+r < i+1 ==>
          Mm_C.acc2 (Mm_C.upd values (i%16, (i/16, ())) value) r token %~
            Mm_Shared.linear_rows weights input token (tile*16+r)) /\
        (token*16+r >= i+1 ==>
          Mm_C.acc2 (Mm_C.upd values (i%16, (i/16, ())) value) r token ==
            Mm_C.acc2 original r token)) =
    Kuiper.Math.div_mod_of_mul_add 16 token r;
    if token*16+r = i then () else ()
  in FStar.Classical.forall_intro_2 point

inline_for_extraction noextract
fn mm_add_split_results
  (p:mm_linear_parameters{mm_linear_valid p /\ (Mm_SZ.v p.splits)=2})
  (left:array2 float mm_c_layout) (right:array2 float mm_c_layout)
  (#lv #rv:chest2 float 16 8)
  (#weights:erased (Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)))
  (#input:erased (Mm_Shared.matrix 7 (Mm_SZ.v p.width)))
  (tile:sz{tile*16+16<=(Mm_SZ.v p.columns)})
  preserves gpu ** right |-> rv
  requires left |-> lv
  requires pure (mm_segment_result p (reveal weights) (reveal input) tile 0 lv /\
    mm_segment_result p (reveal weights) (reveal input) tile 1 rv)
  ensures exists* (values:chest2 float 16 8).
    left |-> values ** pure (mm_tile_result p (reveal weights) (reveal input) tile values)
{
  let mut i = 0sz;
  while (!i <^ 112sz)
    invariant exists* (j:sz{j<=112}) (values:chest2 float 16 8).
      i |-> j ** left |-> values **
      pure (mm_split_progress p weights input tile lv values j)
    decreases (112 - !i)
  {
    let j = !i;
    let token = j /^ 16sz;
    let r = j %^ 16sz;
    let x = Mm_T.tensor_read left (r, (token, ()));
    let y = Mm_T.tensor_read right (r, (token, ()));
    let result = mm_rn_add x y
      #(hide (mm_sum_n (mm_chunk p) (mm_segment_term p (reveal weights) (reveal input) (tile*16+r) token 0)))
      #(hide (mm_sum_n (mm_chunk p) (mm_segment_term p (reveal weights) (reveal input) (tile*16+r) token 1)));
    mm_paired_segments p (reveal weights) (reveal input) (tile*16+r) token;
    with values. assert left |-> values;
    mm_split_write p weights input tile lv values j result;
    Mm_T.tensor_write left (r, (token, ())) result;
    i := j +^ 1sz;
  };
}

instance mm_send_global_array_contents
  (#a:Type0) (arr:array a{is_global_array arr}) (#f:perm) (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents) =
  Kuiper.Array.Core.is_send_pts_to arr #f contents

inline_for_extraction noextract
fn mm_finish_splits
  (p:mm_linear_parameters{mm_linear_valid p})
  (#n:erased nat{n<=2147483647})
  (weights:larray bf16 n{is_global_array weights})
  (input:larray bf16 (7*(Mm_SZ.v p.width)){is_global_array input})
  (offset:sz{offset+(Mm_SZ.v p.columns)*(Mm_SZ.v p.width)<=n})
  (tile:sz{tile*16+16<=(Mm_SZ.v p.columns)})
  (a:array2 bf16 mm_a_layout{Mm_T.is_global a})
  (b:array2 bf16 mm_b_layout{Mm_T.is_global b})
  (c0:array2 float mm_c_layout{Mm_T.is_global c0})
  (c1:array2 float mm_c_layout{Mm_T.is_global c1})
  (#wv:erased (lseq bf16 n)) (#xv:erased (lseq bf16 (7*(Mm_SZ.v p.width))))
  (#av:chest2 bf16 16 2048) (#bv:chest2 bf16 2048 8)
  (#cv0 #cv1:chest2 float 16 8)
  (#wp #xp:perm)
  (#wr:erased (Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)))
  (#xr:erased (Mm_Shared.matrix 7 (Mm_SZ.v p.width)))
  preserves cpu **
    on gpu_loc (weights |-> Frac wp wv) **
    on gpu_loc (input |-> Frac xp xv)
  requires on gpu_loc (a |-> av) ** on gpu_loc (b |-> bv) **
    on gpu_loc (c0 |-> cv0) ** on gpu_loc (c1 |-> cv1)
  requires pure (mm_weights_rep p (reveal wv) offset (reveal wr) /\
    mm_input_rep p (reveal xv) (reveal xr) /\
    mm_segment_result p (reveal wr) (reveal xr) tile 0 cv0)
  ensures exists* (af:chest2 bf16 16 2048) (bf:chest2 bf16 2048 8)
    (cf0:chest2 float 16 8) (cf1:chest2 float 16 8).
    on gpu_loc (a |-> af) ** on gpu_loc (b |-> bf) **
    on gpu_loc (c0 |-> cf0) ** on gpu_loc (c1 |-> cf1) **
    pure (mm_tile_result p (reveal wr) (reveal xr) tile cf0)
{
  if (p.splits = 2sz) {
    let steps = p.width /^ p.splits /^ 16sz;
    launch_kernel_1 (fun () ->
      mm_stage_operands p weights input offset tile 1sz a b #wv #xv #av #bv #wp #xp);
    let av1 = mm_a_values p (reveal wv) offset tile 1;
    let bv1 = mm_b_values p (reveal xv) 1;
    mm_run_mma a b c1 steps #av1 #bv1 #cv1;
    mm_operands_approx p (reveal wv) (reveal xv) offset tile 1 (reveal wr) (reveal xr);
    mm_segment_from_mma p av1 bv1 (reveal wr) (reveal xr) tile 1;
    launch_kernel_1 (fun () ->
      mm_add_split_results p c0 c1 #cv0 #(mm_mma_accum av1 bv1 steps) #wr #xr tile);
    ();
  } else {
    mm_single_segment p (reveal wr) (reveal xr) tile cv0;
    ();
  };
}

noextract let mm_output_prefix
  (p:mm_linear_parameters{mm_linear_valid p})
  (values:lseq bf16 (7*(Mm_SZ.v p.columns)))
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (columns:nat{columns<=(Mm_SZ.v p.columns)}) : prop =
  forall (token:natlt 7) (column:natlt (Mm_SZ.v p.columns)). column < columns ==>
    Mm_Seq.index values (token*(Mm_SZ.v p.columns)+column) %~
      Mm_Shared.linear_rows weights input token column

noextract let mm_tile_progress
  (p:mm_linear_parameters{mm_linear_valid p})
  (values:lseq bf16 (7*(Mm_SZ.v p.columns)))
  (weights:Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)) (input:Mm_Shared.matrix 7 (Mm_SZ.v p.width))
  (tile:nat{tile*16+16<=(Mm_SZ.v p.columns)}) (count:nat{count<=112}) : prop =
  forall (token:natlt 7) (column:natlt (Mm_SZ.v p.columns)).
    (column < tile*16 \/
     (tile*16 <= column /\ column < tile*16+16 /\ token*16+column-tile*16 < count)) ==>
    Mm_Seq.index values (token*(Mm_SZ.v p.columns)+column) %~
      Mm_Shared.linear_rows weights input token column

inline_for_extraction noextract
fn mm_write_tile
  (p:mm_linear_parameters{mm_linear_valid p})
  (c:array2 float mm_c_layout) (output:larray bf16 (7*(Mm_SZ.v p.columns)))
  (#cv:chest2 float 16 8) (#old:erased (lseq bf16 (7*(Mm_SZ.v p.columns))))
  (#weights:erased (Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)))
  (#input:erased (Mm_Shared.matrix 7 (Mm_SZ.v p.width)))
  (tile:sz{tile*16+16<=(Mm_SZ.v p.columns)})
  preserves gpu ** c |-> cv
  requires output |-> old
  requires pure (mm_tile_result p (reveal weights) (reveal input) tile cv /\
    mm_output_prefix p (reveal old) (reveal weights) (reveal input) (tile*16))
  ensures exists* (values:lseq bf16 (7*(Mm_SZ.v p.columns))).
    output |-> values **
    pure (mm_output_prefix p values (reveal weights) (reveal input) ((tile+1)*16))
{
  open Pulse.Lib.Array;
  let mut i = 0sz;
  while (!i <^ 112sz)
    invariant exists* (j:sz{j<=112}) (values:lseq bf16 (7*(Mm_SZ.v p.columns))).
      i |-> j ** output |-> values **
      pure (mm_tile_progress p values (reveal weights) (reveal input) tile j)
    decreases (112 - !i)
  {
    let j = !i;
    let token = j /^ 16sz;
    let r = j %^ 16sz;
    let column = tile *^ 16sz +^ r;
    let score = Mm_T.tensor_read c (r, (token, ()));
    Mm_Cast.cast_f32_to_bf16_ok score
      (Mm_Shared.linear_rows (reveal weights) (reveal input) token column);
    let value = Mm_Cast.cast_f32_to_bf16 score;
    output.(token *^ p.columns +^ column) <- value;
    i := j +^ 1sz;
  };
}

fn mm_run_linear
  (p:mm_linear_parameters{mm_linear_valid p})
  (#n:erased nat{n<=2147483647})
  (weights:larray bf16 n{is_global_array weights})
  (input:larray bf16 (7*(Mm_SZ.v p.width)){is_global_array input})
  (output:larray bf16 (7*(Mm_SZ.v p.columns)){is_global_array output})
  (offset:sz{offset+(Mm_SZ.v p.columns)*(Mm_SZ.v p.width)<=n})
  (#wv:erased (lseq bf16 n)) (#xv:erased (lseq bf16 (7*(Mm_SZ.v p.width))))
  (#old:erased (seq bf16))
  (#wp #xp:perm)
  (#wr:erased (Mm_Shared.matrix (Mm_SZ.v p.columns) (Mm_SZ.v p.width)))
  (#xr:erased (Mm_Shared.matrix 7 (Mm_SZ.v p.width)))
  preserves cpu **
    on gpu_loc (weights |-> Frac wp wv) **
    on gpu_loc (input |-> Frac xp xv)
  requires on gpu_loc (output |-> old)
  requires pure (Mm_Seq.length old == 7*(Mm_SZ.v p.columns))
  requires pure (mm_weights_rep p (reveal wv) offset (reveal wr) /\
    mm_input_rep p (reveal xv) (reveal xr))
  ensures exists* (values:lseq bf16 (7*(Mm_SZ.v p.columns))).
    on gpu_loc (output |-> values) **
    pure (mm_output_prefix p values (reveal wr) (reveal xr) (Mm_SZ.v p.columns))
{
  let a = Mm_T.alloc0 #bf16 32768sz mm_a_layout;
  let b = Mm_T.alloc0 #bf16 16384sz mm_b_layout;
  let c0 = Mm_T.alloc0 #float 128sz mm_c_layout;
  let c1 = Mm_T.alloc0 #float 128sz mm_c_layout;
  let steps = p.width /^ p.splits /^ 16sz;
  let tiles = p.columns /^ 16sz;
  let mut tile = 0sz;
  while (!tile <^ tiles)
    invariant exists* (j:sz{j<=tiles})
      (av:chest2 bf16 16 2048) (bv:chest2 bf16 2048 8)
      (cv0:chest2 float 16 8) (cv1:chest2 float 16 8)
      (values:lseq bf16 (7*(Mm_SZ.v p.columns))).
      tile |-> j **
      on gpu_loc (a |-> av) ** on gpu_loc (b |-> bv) **
      on gpu_loc (c0 |-> cv0) ** on gpu_loc (c1 |-> cv1) **
      on gpu_loc (output |-> values) **
      pure (mm_output_prefix p values (reveal wr) (reveal xr) (j*16))
    decreases (tiles - !tile)
  {
    let j = !tile;
    with av. assert on gpu_loc (a |-> av);
    with bv. assert on gpu_loc (b |-> bv);
    with cv0. assert on gpu_loc (c0 |-> cv0);
    launch_kernel_1 (fun () ->
      mm_stage_operands p weights input offset j 0sz a b #wv #xv #av #bv #wp #xp);
    let av0 = mm_a_values p (reveal wv) offset j 0;
    let bv0 = mm_b_values p (reveal xv) 0;
    mm_run_mma a b c0 steps #av0 #bv0 #cv0;
    mm_operands_approx p (reveal wv) (reveal xv) offset j 0 (reveal wr) (reveal xr);
    mm_segment_from_mma p av0 bv0 (reveal wr) (reveal xr) j 0;
    with cv1. assert on gpu_loc (c1 |-> cv1);
    mm_finish_splits p weights input offset j a b c0 c1
      #wv #xv #av0 #bv0 #(mm_mma_accum av0 bv0 steps) #cv1 #wp #xp #wr #xr;
    with cv. assert on gpu_loc (c0 |-> cv);
    with ov. assert on gpu_loc (output |-> ov);
    launch_kernel_1 (fun () -> mm_write_tile p c0 output #cv #ov #wr #xr j);
    tile := j +^ 1sz;
  };
  Mm_T.free a;
  Mm_T.free b;
  Mm_T.free c0;
  Mm_T.free c1;
}

#set-options "--fuel 4 --ifuel 1 --z3rlimit 80"

noextract let matrix_prefix
  (#rows #cols:pos) (values:lseq BF16.t (rows*cols))
  (logical:Shared.matrix rows cols) (count:nat{count<=rows*cols}) : prop =
  forall (r:natlt rows) (c:natlt cols). r*cols+c < count ==>
    Seq.index values (r*cols+c) %~ logical r c

noextract let matrix_rep
  (#rows #cols:pos) (values:lseq BF16.t (rows*cols))
  (logical:Shared.matrix rows cols) : prop =
  matrix_prefix values logical (rows*cols)

let matrix_at
  (#rows #cols:pos) (values:lseq BF16.t (rows*cols))
  (logical:Shared.matrix rows cols) (row:natlt rows) (col:natlt cols)
  : Lemma
      (requires matrix_rep values logical)
      (ensures Seq.index values (row*cols+col) %~ logical row col) = ()

let matrix_write
  (#rows #cols:pos) (values:lseq BF16.t (rows*cols))
  (logical:Shared.matrix rows cols) (i:natlt (rows*cols)) (value:BF16.t)
  : Lemma
      (requires matrix_prefix values logical i /\ value %~ logical (i/cols) (i%cols))
      (ensures matrix_prefix (Seq.upd values i value) logical (i+1)) =
  let point (r:natlt rows) (c:natlt cols)
    : Lemma
        (requires r*cols+c < i+1)
        (ensures Seq.index (Seq.upd values i value) (r*cols+c) %~ logical r c) =
    Kuiper.Math.div_mod_of_mul_add cols r c
  in FStar.Classical.forall_intro_2 (fun r -> FStar.Classical.move_requires (point r))

let dot_equal (#n:pos) (x y u v:Shared.vector n)
  : Lemma
      (requires forall (i:natlt n). x i == u i /\ y i == v i)
      (ensures Shared.dot x y == Shared.dot u v) =
  shared_sum_equal n (fun i -> x i *. y i) (fun i -> u i *. v i)

let rms_equal (#n:pos) (weight0 weight1 x0 x1:Shared.vector n) (i:natlt n)
  : Lemma
      (requires forall (j:natlt n). weight0 j == weight1 j /\ x0 j == x1 j)
      (ensures Shared.rms_norm weight0 x0 i == Shared.rms_norm weight1 x1 i) =
  Shared.sum_nonnegative n (fun j -> x0 j *. x0 j);
  Shared.sum_nonnegative n (fun j -> x1 j *. x1 j);
  shared_sum_equal n (fun j -> x0 j *. x0 j) (fun j -> x1 j *. x1 j);
  assert_norm (Shared.rms_norm weight0 x0 i ==
    weight0 i *. (x0 i *. Root.rsqrt
      (Shared.sum n (fun j -> x0 j *. x0 j) /. FStar.Real.of_int n +. 0.000001R)));
  assert_norm (Shared.rms_norm weight1 x1 i ==
    weight1 i *. (x1 i *. Root.rsqrt
      (Shared.sum n (fun j -> x1 j *. x1 j) /. FStar.Real.of_int n +. 0.000001R)))

let softmax_equal (#n:pos) (x y:Shared.vector n) (i:natlt n)
  : Lemma
      (requires forall (j:natlt n). x j == y j)
      (ensures Shared.softmax x i == Shared.softmax y i) =
  Shared.sum_positive n (fun j -> Exp.exp (x j));
  Shared.sum_positive n (fun j -> Exp.exp (y j));
  shared_sum_equal n (fun j -> Exp.exp (x j)) (fun j -> Exp.exp (y j));
  assert_norm (Shared.softmax x i ==
    Exp.exp (x i) /. Shared.sum n (fun j -> Exp.exp (x j)));
  assert_norm (Shared.softmax y i ==
    Exp.exp (y i) /. Shared.sum n (fun j -> Exp.exp (y j)))

let prefix_softmax_equal
  (row:Shared.index 7) (x y:Shared.vector (row+1)) (i:Shared.index 7)
  : Lemma
      (requires forall (j:Shared.index (row+1)). x j == y j)
      (ensures Shared.prefix_softmax row x i == Shared.prefix_softmax row y i) =
  if i <= row then softmax_equal x y i

let linear_rows_equal (#rows #src #dst:pos)
  (weight:Shared.matrix dst src) (x y:Shared.matrix rows src)
  : Lemma
      (requires forall (r:natlt rows) (c:natlt src). x r c == y r c)
      (ensures forall (r:natlt rows) (c:natlt dst).
        Shared.linear_rows weight x r c == Shared.linear_rows weight y r c) =
  let point (r:natlt rows) (c:natlt dst)
    : Lemma (Shared.linear_rows weight x r c == Shared.linear_rows weight y r c) =
    dot_equal (x r) (weight c) (y r) (weight c)
  in FStar.Classical.forall_intro_2 point

[@@"opaque_to_smt"]
noextract let vector_values (#n:pos) (x:Shared.vector n)
  : GTot (values:lseq real n{forall (i:natlt n). Seq.index values i == x i}) =
  Seq.init_ghost n x

let rms_vector_at (#n:pos) (weight x:Shared.vector n) (i:natlt n)
  : Lemma (Seq.index (nr_rms_values (vector_values weight) (vector_values x)) i ==
      Shared.rms_norm weight x i) =
  rms_equal (Seq.index (vector_values weight)) weight (Seq.index (vector_values x)) x i

noextract let selected_hidden (fused:bool) (left right:Shared.hidden_states)
  : GTot Shared.hidden_states =
  if fused then Shared.residual_add left right else left

inline_for_extraction noextract
fn select_hidden_value (fused:bool) (left right:BF16.t) (#lr #rr:erased real)
  preserves gpu
  requires pure (left %~ reveal lr /\ (fused ==> right %~ reveal rr))
  returns result:BF16.t
  ensures pure (result %~ (if fused then reveal lr +. reveal rr else reveal lr))
{
  if fused {
    nr_add_bf16 left right #lr #rr
  } else {
    left
  }
}

inline_for_extraction noextract
fn combine_hidden
  (fused:bool)
  (left right output:larray BF16.t 7168)
  (#lv #rv:erased (lseq BF16.t 7168)) (#old:erased (seq BF16.t))
  (#lr #rr:erased Shared.hidden_states) (#lp #rp:perm)
  preserves gpu ** left |-> Frac lp lv ** right |-> Frac rp rv
  requires output |-> old
  requires pure (matrix_rep lv lr /\ (fused ==> matrix_rep rv rr))
  ensures exists* (values:lseq BF16.t 7168).
    output |-> values ** pure (matrix_rep values (selected_hidden fused lr rr))
{
  open Pulse.Lib.Array;
  pts_to_len output;
  let mut i = 0sz;
  while (!i <^ 7168sz)
    invariant exists* (j:sz{j<=7168}) (values:lseq BF16.t 7168).
      i |-> j ** output |-> values **
      pure (matrix_prefix values (selected_hidden fused lr rr) j)
    decreases (7168 - !i)
  {
    let j = !i;
    let r = j /^ 1024sz;
    let c = j %^ 1024sz;
    let x = left.(j);
    let y = right.(j);
    let value = select_hidden_value fused x y #((reveal lr) r c) #((reveal rr) r c);
    with values. assert output |-> values;
    matrix_write values (selected_hidden fused lr rr) j value;
    output.(j) <- value;
    i := j +^ 1sz;
  };
}

let normalized_row_progress
  (old current:lseq BF16.t 7168)
  (weight:Shared.vector 1024) (input:Shared.hidden_states) (row:natlt 7)
  : Lemma
      (requires
        matrix_prefix old (Shared.normalize_hidden weight input) (row*1024) /\
        nr_same_outside_segment old current (row*1024) 1024 /\
        nr_bf16_segment current (row*1024) 1024
          (nr_rms_values (vector_values weight) (vector_values (input row))))
      (ensures matrix_prefix current (Shared.normalize_hidden weight input) ((row+1)*1024)) =
  let point (r:natlt 7) (c:natlt 1024)
    : Lemma
        (requires r*1024+c < (row+1)*1024)
        (ensures Seq.index current (r*1024+c) %~
          Shared.normalize_hidden weight input r c) =
    if r = row then rms_vector_at weight (input row) c
    else assert (Seq.index current (r*1024+c) == Seq.index old (r*1024+c))
  in FStar.Classical.forall_intro_2 (fun r -> FStar.Classical.move_requires (point r))

inline_for_extraction noextract
fn normalize_hidden_kernel
  (#n:erased nat{n<=2147483647})
  (weights:larray BF16.t n) (input output:larray BF16.t 7168)
  (buffer:larray F32.t 128) (offset:sz{offset+1024<=n})
  (#wv:erased (lseq BF16.t n)) (#xv:erased (lseq BF16.t 7168))
  (#old:erased (seq BF16.t)) (#scratch:erased (seq F32.t))
  (#wr:erased (Shared.vector 1024)) (#xr:erased Shared.hidden_states) (#wp:perm)
  preserves gpu ** weights |-> Frac wp wv ** input |-> xv
  requires output |-> old ** buffer |-> scratch
  requires pure (matrix_rep xv xr /\
    (forall (d:natlt 1024). Seq.index wv (offset+d) %~ (reveal wr) d))
  ensures exists* (values:lseq BF16.t 7168) (sv:seq F32.t).
    output |-> values ** buffer |-> sv **
    pure (Seq.length sv == 128 /\ matrix_rep values (Shared.normalize_hidden wr xr))
{
  pts_to_len output;
  pts_to_len buffer;
  let mut i = 0sz;
  while (!i <^ 7sz)
    invariant exists* (row:sz{row<=7}) (values:lseq BF16.t 7168) (sv:lseq F32.t 128).
      i |-> row ** output |-> values ** buffer |-> sv **
      pure (matrix_prefix values (Shared.normalize_hidden wr xr) (row*1024))
    decreases (7 - !i)
  {
    let row = !i;
    with before. assert output |-> before;
    with sv. assert buffer |-> sv;
    nr_rms_row input weights output buffer (row *^ 1024sz) offset (row *^ 1024sz) 1024sz
      #xv #wv #before #sv #(vector_values ((reveal xr) row)) #(vector_values wr) #wp;
    with after. assert output |-> after;
    normalized_row_progress before after wr xr row;
    i := row +^ 1sz;
  };
}

inline_for_extraction noextract
fn rn_negate (x:F32.t) (#xr:erased real)
  preserves gpu
  requires pure (x %~ reveal xr)
  returns result:F32.t
  ensures pure (result %~ (0.0R -. reveal xr))
{
  let minus_one = of_int #F32.t (-1L);
  of_int_approx #F32.t (-1L);
  nr_rn_multiply minus_one x #(0.0R -. 1.0R) #xr
}

inline_for_extraction noextract
fn silu_value (gate up:BF16.t) (#gr #ur:erased real)
  preserves gpu
  requires pure (gate %~ reveal gr /\ up %~ reveal ur)
  returns result:BF16.t
  ensures pure (result %~ (Shared.silu (reveal gr) *. reveal ur))
{
  Casts.fcast_approx #BF16.t #F32.t gate (reveal gr);
  Casts.fcast_approx #BF16.t #F32.t up (reveal ur);
  let g:F32.t = Casts.fcast gate;
  let u:F32.t = Casts.fcast up;
  let negative = rn_negate g #gr;
  exp_approx negative (0.0R -. reveal gr);
  let exponential = fexp negative;
  let one_ok = a1 #F32.t;
  let denominator = nr_rn_add (one #F32.t) exponential
    #1.0R #(Exp.exp (0.0R -. reveal gr));
  let dr:erased real = 1.0R +. Exp.exp (0.0R -. reveal gr);
  div_approx g denominator (reveal gr) (reveal dr);
  let quotient = div g denominator;
  Casts.fcast_approx #F32.t #BF16.t quotient (Shared.silu (reveal gr));
  let activated:BF16.t = Casts.fcast quotient;
  Casts.fcast_approx #BF16.t #F32.t activated (Shared.silu (reveal gr));
  let narrowed:F32.t = Casts.fcast activated;
  let product = nr_rn_multiply narrowed u #(Shared.silu (reveal gr)) #ur;
  Casts.fcast_approx #F32.t #BF16.t product (Shared.silu (reveal gr) *. reveal ur);
  Casts.fcast product
}

noextract let gated_model (input:Shared.matrix 7 6144) : GTot (Shared.matrix 7 3072) =
  fun row col -> Shared.silu (input row col) *. input row (3072+col)

inline_for_extraction noextract
fn silu_kernel
  (input:larray BF16.t 43008) (output:larray BF16.t 21504)
  (#xv:erased (lseq BF16.t 43008)) (#old:erased (seq BF16.t))
  (#xr:erased (Shared.matrix 7 6144))
  preserves gpu ** input |-> xv
  requires output |-> old ** pure (matrix_rep xv xr)
  ensures exists* (values:lseq BF16.t 21504).
    output |-> values ** pure (matrix_rep values (gated_model xr))
{
  open Pulse.Lib.Array;
  pts_to_len output;
  let mut i = 0sz;
  while (!i <^ 21504sz)
    invariant exists* (j:sz{j<=21504}) (values:lseq BF16.t 21504).
      i |-> j ** output |-> values ** pure (matrix_prefix values (gated_model xr) j)
    decreases (21504 - !i)
  {
    let j = !i;
    let row = j /^ 3072sz;
    let col = j %^ 3072sz;
    let gate = input.(row *^ 6144sz +^ col);
    let up = input.(row *^ 6144sz +^ 3072sz +^ col);
    let value = silu_value gate up #((reveal xr) row col) #((reveal xr) row (3072+col));
    with values. assert output |-> values;
    matrix_write values (gated_model xr) j value;
    output.(j) <- value;
    i := j +^ 1sz;
  };
}

inline_for_extraction noextract
fn signed_partner (lower:bool) (x:F32.t) (#xr:erased real)
  preserves gpu
  requires pure (x %~ reveal xr)
  returns result:F32.t
  ensures pure (result %~ (if lower then 0.0R -. reveal xr else reveal xr))
{
  if lower { rn_negate x #xr } else { x }
}

inline_for_extraction noextract
fn rotary_value (x partner cosine sine:BF16.t) (lower:bool)
  (#xr #pr #cr #sr:erased real)
  preserves gpu
  requires pure (x %~ reveal xr /\ partner %~ reveal pr /\
    cosine %~ reveal cr /\ sine %~ reveal sr)
  returns result:BF16.t
  ensures pure (result %~ (reveal xr *. reveal cr +.
    (if lower then 0.0R -. reveal pr else reveal pr) *. reveal sr))
{
  Casts.fcast_approx #BF16.t #F32.t x (reveal xr);
  Casts.fcast_approx #BF16.t #F32.t partner (reveal pr);
  Casts.fcast_approx #BF16.t #F32.t cosine (reveal cr);
  Casts.fcast_approx #BF16.t #F32.t sine (reveal sr);
  let xf:F32.t = Casts.fcast x;
  let pf:F32.t = Casts.fcast partner;
  let cf:F32.t = Casts.fcast cosine;
  let sf:F32.t = Casts.fcast sine;
  let rotated = signed_partner lower pf #pr;
  let rr:erased real = (if lower then 0.0R -. reveal pr else reveal pr);
  let product0 = nr_rn_multiply xf cf #xr #cr;
  let product1 = nr_rn_multiply rotated sf #rr #sr;
  Casts.fcast_approx #F32.t #BF16.t product0 (reveal xr *. reveal cr);
  Casts.fcast_approx #F32.t #BF16.t product1 (reveal rr *. reveal sr);
  let a:BF16.t = Casts.fcast product0;
  let b:BF16.t = Casts.fcast product1;
  nr_add_bf16 a b #(reveal xr *. reveal cr) #(reveal rr *. reveal sr)
}

noextract let qk_gamma (weights:Shared.parameters) (head:natlt 24)
  : GTot (Shared.vector 128) =
  if head < 16 then weights.q_norm else weights.k_norm

[@@"opaque_to_smt"]
noextract let normalized_qk (weights:Shared.parameters) (input:Shared.matrix 7 4096)
  : GTot (Shared.heads 24) =
  fun row head -> Shared.rms_norm (qk_gamma weights head)
    (fun col -> input row (head*128+col))

let normalized_qk_at
  (weights:Shared.parameters) (input:Shared.matrix 7 4096)
  (row:natlt 7) (head:natlt 24) (col:natlt 128)
  : Lemma (normalized_qk weights input row head col ==
    Shared.rms_norm (qk_gamma weights head) (fun lane -> input row (head*128+lane)) col) =
  assert (normalized_qk weights input row head col ==
    Shared.rms_norm (qk_gamma weights head) (fun lane -> input row (head*128+lane)) col) by
    (FStar.Tactics.V2.norm [delta_only [`%normalized_qk]];
     FStar.Tactics.V2.trivial ())

[@@"opaque_to_smt"]
noextract let qk_model
  (weights:Shared.parameters) (input:Shared.matrix 7 4096) (data:Shared.attention_data)
  : GTot (Shared.heads 24) =
  Shared.rotary data (normalized_qk weights input)

noextract let flat_heads (input:Shared.heads 24) : GTot (Shared.matrix 7 3072) =
  fun row col -> input row (col/128) (col%128)

let qk_read (values:lseq BF16.t 21504) (logical:Shared.heads 24)
  (row:natlt 7) (head:natlt 24) (col:natlt 128)
  : Lemma
      (requires matrix_rep values (flat_heads logical))
      (ensures Seq.index values (row*3072+head*128+col) %~ logical row head col) =
  matrix_at values (flat_heads logical) row (head*128+col);
  Kuiper.Math.div_mod_of_mul_add 128 head col

let qk_flat_at (logical:Shared.heads 24) (j:natlt 168) (c:natlt 128)
  : Lemma (flat_heads logical ((j*128+c)/3072) ((j*128+c)%3072) ==
    logical (j/24) (j%24) c) =
  Kuiper.Math.div_mod_of_mul_add 3072 (j/24) ((j%24)*128+c);
  Kuiper.Math.div_mod_of_mul_add 128 (j%24) c

let qk_rotary_at
  (weights:Shared.parameters) (input:Shared.matrix 7 4096) (data:Shared.attention_data)
  (row:natlt 7) (head:natlt 24) (col:natlt 128)
  : Lemma (
      qk_model weights input data row head col ==
      normalized_qk weights input row head col *. data.cosine row col +.
        (if col < 64 then 0.0R -. normalized_qk weights input row head ((col+64)%128)
         else normalized_qk weights input row head ((col+64)%128)) *. data.sine row col) =
  assert ((col+64)%128 == (if col < 64 then col+64 else col-64));
  assert (qk_model weights input data row head col ==
    normalized_qk weights input row head col *. data.cosine row col +.
      (if col < 64 then 0.0R -. normalized_qk weights input row head (col+64)
       else normalized_qk weights input row head (col-64)) *. data.sine row col) by
    (FStar.Tactics.V2.norm [delta_only [`%qk_model; `%Shared.rotary]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
fn normalize_rotate_kernel
  (#n:erased nat{n<=2147483647})
  (weights:larray BF16.t n) (input:larray BF16.t 28672)
  (rotary:larray BF16.t 1792) (output:larray BF16.t 21504)
  (normalized:larray BF16.t 128) (buffer:larray F32.t 128)
  (offset:sz{offset+256<=n})
  (#wv:erased (lseq BF16.t n)) (#xv:erased (lseq BF16.t 28672))
  (#rv:erased (lseq BF16.t 1792)) (#old:erased (seq BF16.t))
  (#nv:erased (seq BF16.t)) (#sv:erased (seq F32.t))
  (#wr:erased Shared.parameters) (#xr:erased (Shared.matrix 7 4096))
  (#data:erased Shared.attention_data) (#wp #rp:perm)
  preserves gpu ** weights |-> Frac wp wv ** input |-> xv ** rotary |-> Frac rp rv
  requires output |-> old ** normalized |-> nv ** buffer |-> sv
  requires pure (matrix_rep xv xr /\ rotary_representation rv data /\
    (forall (d:natlt 128).
      Seq.index wv (offset+d) %~ (reveal wr).q_norm d /\
      Seq.index wv (offset+128+d) %~ (reveal wr).k_norm d))
  ensures exists* (values:lseq BF16.t 21504) (nval:seq BF16.t) (sval:seq F32.t).
    output |-> values ** normalized |-> nval ** buffer |-> sval **
    pure (Seq.length nval == 128 /\ Seq.length sval == 128 /\
      matrix_rep values (flat_heads (qk_model wr xr data)))
{
  open Pulse.Lib.Array;
  pts_to_len output;
  pts_to_len normalized;
  pts_to_len buffer;
  let mut index = 0sz;
  while (!index <^ 168sz)
    invariant exists* (j:sz{j<=168}) (values:lseq BF16.t 21504)
      (nval:lseq BF16.t 128) (sval:lseq F32.t 128).
      index |-> j ** output |-> values ** normalized |-> nval ** buffer |-> sval **
      pure (matrix_prefix values (flat_heads (qk_model wr xr data)) (j*128))
    decreases (168 - !index)
  {
    let j = !index;
    let row = j /^ 24sz;
    let head = j %^ 24sz;
    let weight_start = (if head <^ 16sz then offset else offset +^ 128sz);
    with nval. assert normalized |-> nval;
    with sval. assert buffer |-> sval;
    nr_rms_row input weights normalized buffer
      (row *^ 4096sz +^ head *^ 128sz) weight_start 0sz 128sz
      #xv #wv #nval #sval
      #(vector_values (fun col -> (reveal xr) row (head*128+col)))
      #(vector_values (qk_gamma wr head)) #wp;
    with current_normalized. assert normalized |-> current_normalized;
    let mut lane = 0sz;
    while (!lane <^ 128sz)
      invariant exists* (c:sz{c<=128}) (values:lseq BF16.t 21504).
        lane |-> c ** output |-> values **
        pure (matrix_prefix values (flat_heads (qk_model wr xr data)) (j*128+c))
      decreases (128 - !lane)
    {
      let c = !lane;
      let partner = (c +^ 64sz) %^ 128sz;
      let x = normalized.(c);
      let other = normalized.(partner);
      let cosine = rotary.(row *^ 128sz +^ c);
      let sine = rotary.(896sz +^ row *^ 128sz +^ c);
      rms_vector_at (qk_gamma wr head) (fun d -> (reveal xr) row (head*128+d)) c;
      rms_vector_at (qk_gamma wr head) (fun d -> (reveal xr) row (head*128+d)) partner;
      normalized_qk_at wr xr row head c;
      normalized_qk_at wr xr row head partner;
      let value = rotary_value x other cosine sine (c <^ 64sz)
        #(normalized_qk wr xr row head c) #(normalized_qk wr xr row head partner)
        #((reveal data).cosine row c) #((reveal data).sine row c);
      qk_rotary_at wr xr data row head c;
      assert pure (value %~ qk_model wr xr data row head c);
      qk_flat_at (qk_model wr xr data) j c;
      with values. assert output |-> values;
      matrix_write values (flat_heads (qk_model wr xr data)) (j*128+c) value;
      output.(j *^ 128sz +^ c) <- value;
      lane := c +^ 1sz;
    };
    index := j +^ 1sz;
  };
}

let rec sum_zero (n:nat)
  : Lemma (nr_sum_n n (fun _ -> 0.0R) == 0.0R) (decreases n) =
  if n > 0 then sum_zero (n-1)

let rec sum_swap (n m:nat) (f:nat -> nat -> GTot real)
  : Lemma (nr_sum_n n (fun i -> nr_sum_n m (f i)) ==
      nr_sum_n m (fun j -> nr_sum_n n (fun i -> f i j)))
      (decreases n) =
  if n = 0 then (
    sum_zero m;
    nr_sum_ext m (fun j -> nr_sum_n n (fun i -> f i j)) (fun _ -> 0.0R)
  ) else (
    sum_swap (n-1) m f;
    nr_sum_step n (fun i -> nr_sum_n m (f i));
    nr_sum_ext m (fun j -> nr_sum_n n (fun i -> f i j))
      (fun j -> nr_sum_n (n-1) (fun i -> f i j) +. f (n-1) j);
    nr_sum_add m (fun j -> nr_sum_n (n-1) (fun i -> f i j)) (f (n-1))
  )

noextract let score_term
  (qk:Shared.heads 24) (row key:natlt 7) (head:natlt 16) (col:nat) : GTot real =
  if col < 128 then qk row head col *. qk key (16+head/2) col else 0.0R

noextract let score_lane_term
  (qk:Shared.heads 24) (row key:natlt 7) (head:natlt 16) (lane:natlt 16)
  (step:nat) : GTot real =
  score_term qk row key head (step*16+lane)

[@@"opaque_to_smt"]
noextract let score_lanes
  (qk:Shared.heads 24) (row key:natlt 7) (head:natlt 16)
  : GTot (values:lseq real 16{forall (lane:natlt 16).
    Seq.index values lane == nr_sum_n 8 (score_lane_term qk row key head lane)}) =
  Seq.init_ghost 16 (fun lane -> nr_sum_n 8 (score_lane_term qk row key head lane))

let score_lanes_sum
  (qk:Shared.heads 24) (row key:natlt 7) (head:natlt 16)
  : Lemma (nr_sum_values (score_lanes qk row key head) ==
      Shared.dot (qk row head) (qk key (16+head/2))) =
  let f = score_term qk row key head in
  let lane_at (lane:natlt 16)
    : Lemma (nr_at (score_lanes qk row key head) lane ==
        nr_sum_n 8 (fun step -> f (step*16+lane))) =
    nr_sum_ext 8 (score_lane_term qk row key head lane)
      (fun step -> f (step*16+lane))
  in
  FStar.Classical.forall_intro lane_at;
  nr_sum_ext 16 (nr_at (score_lanes qk row key head))
    (fun lane -> nr_sum_n 8 (fun step -> f (step*16+lane)));
  sum_swap 16 8 (fun lane step -> f (step*16+lane));
  nr_sum_flat 8 16 f;
  nr_sum_shared 128 (fun col -> qk row head col *. qk key (16+head/2) col) f;
  assert (Shared.dot (qk row head) (qk key (16+head/2)) ==
    Shared.sum 128 (fun col -> qk row head col *. qk key (16+head/2) col)) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.dot]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
fn score_lane
  (input:larray BF16.t 21504) (row key:szlt 7) (head:szlt 16) (lane:szlt 16)
  (#values:erased (lseq BF16.t 21504)) (#qk:erased (Shared.heads 24))
  preserves gpu ** input |-> values
  requires pure (matrix_rep values (flat_heads qk))
  returns result:F32.t
  ensures pure (result %~ nr_sum_n 8 (score_lane_term qk row key head lane))
{
  open Pulse.Lib.Array;
  let mut index = 0sz;
  let mut acc:F32.t = zero #F32.t;
  while (!index <^ 8sz)
    invariant exists* (j:sz{j<=8}) (value:F32.t).
      index |-> j ** acc |-> value **
      pure (value %~ nr_sum_n j (score_lane_term qk row key head lane))
    decreases (8 - !index)
  {
    let j = !index;
    let col = lane +^ 16sz *^ j;
    let x = input.(row *^ 3072sz +^ head *^ 128sz +^ col);
    let y = input.(key *^ 3072sz +^ 2048sz +^ (head /^ 2sz) *^ 128sz +^ col);
    qk_read values qk row head col;
    qk_read values qk key (16+head/2) col;
    Casts.fcast_approx #BF16.t #F32.t x ((reveal qk) row head col);
    Casts.fcast_approx #BF16.t #F32.t y ((reveal qk) key (16+head/2) col);
    let xf:F32.t = Casts.fcast x;
    let yf:F32.t = Casts.fcast y;
    let previous = !acc;
    let value = Native.fma_rn xf yf previous;
    nr_fma_at xf yf previous value
      ((reveal qk) row head col) ((reveal qk) key (16+head/2) col)
      (nr_sum_n j (score_lane_term qk row key head lane));
    nr_sum_step (j+1) (score_lane_term qk row key head lane);
    acc := value;
    index := j +^ 1sz;
  };
  !acc
}

inline_for_extraction noextract
fn score_dot
  (input:larray BF16.t 21504) (buffer:larray F32.t 128)
  (row key:szlt 7) (head:szlt 16)
  (#values:erased (lseq BF16.t 21504)) (#old:erased (lseq F32.t 128))
  (#qk:erased (Shared.heads 24))
  preserves gpu ** input |-> values
  requires buffer |-> old ** pure (matrix_rep values (flat_heads qk))
  returns result:F32.t
  ensures exists* (scratch:lseq F32.t 128). buffer |-> scratch **
    pure (result %~ Shared.dot ((reveal qk) row head) ((reveal qk) key (16+head/2)))
{
  open Pulse.Lib.Array;
  let mut lane = 0sz;
  while (!lane <^ 16sz)
    invariant exists* (j:sz{j<=16}) (scratch:lseq F32.t 128).
      lane |-> j ** buffer |-> scratch **
      pure (forall (i:natlt j).
        Seq.index scratch i %~ Seq.index (score_lanes qk row key head) i)
    decreases (16 - !lane)
  {
    let j = !lane;
    let value = score_lane input row key head j #values #qk;
    buffer.(j) <- value;
    lane := j +^ 1sz;
  };
  with scratch. assert buffer |-> scratch;
  let result = nr_reduce_tree buffer 16sz #scratch #(score_lanes qk row key head);
  score_lanes_sum qk row key head;
  result
}

inline_for_extraction noextract
let attention_scaling ()
  : Tot (value:F32.t{v_approximates value (Root.rsqrt 128.0R)}) =
  let width = of_int #F32.t 128L in
  of_int_approx #F32.t 128L;
  rsqrt_approx width 128.0R;
  rsqrt width

inline_for_extraction noextract
fn scaled_score (value:F32.t) (#vr:erased real)
  preserves gpu
  requires pure (value %~ reveal vr)
  returns result:F32.t
  ensures pure (result %~ (reveal vr *. Root.rsqrt 128.0R))
{
  Casts.fcast_approx #F32.t #BF16.t value (reveal vr);
  let dot:BF16.t = Casts.fcast value;
  Casts.fcast_approx #BF16.t #F32.t dot (reveal vr);
  let widened:F32.t = Casts.fcast dot;
  let scale = attention_scaling ();
  let product = nr_rn_multiply widened scale #vr #(Root.rsqrt 128.0R);
  Casts.fcast_approx #F32.t #BF16.t product (reveal vr *. Root.rsqrt 128.0R);
  let scaled:BF16.t = Casts.fcast product;
  Casts.fcast_approx #BF16.t #F32.t scaled (reveal vr *. Root.rsqrt 128.0R);
  Casts.fcast scaled
}

noextract let score_model
  (qk:Shared.heads 24) (data:Shared.attention_data) (row:natlt 7) (head:natlt 16)
  : GTot (Shared.vector (row+1)) =
  fun key -> Shared.dot (qk row head) (qk key (16+head/2)) *. Root.rsqrt 128.0R

let score_model_at
  (qk:Shared.heads 24) (data:Shared.attention_data)
  (row:natlt 7) (key:natlt (row+1)) (head:natlt 16)
  : Lemma
      (ensures score_model qk data row head key ==
        Shared.dot (qk row head) (qk key (16+head/2)) *. Root.rsqrt 128.0R) = ()

let scores_write (row:natlt 7) (values:lseq F32.t 7) (logical:Shared.vector (row+1))
  (key:natlt (row+1)) (value:F32.t)
  : Lemma
      (requires (forall (i:natlt key). Seq.index values i %~ logical i) /\
        value %~ logical key)
      (ensures forall (i:natlt (key+1)).
        Seq.index (Seq.upd values key value) i %~ logical i) =
  let point (i:natlt (key+1))
    : Lemma (Seq.index (Seq.upd values key value) i %~ logical i) =
    if i = key then () else ()
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn scores_kernel
  (input:larray BF16.t 21504) (buffer:larray F32.t 128) (scores:larray F32.t 7)
  (row:szlt 7) (head:szlt 16)
  (#values:erased (lseq BF16.t 21504))
  (#old_buffer:erased (lseq F32.t 128)) (#old_scores:erased (lseq F32.t 7))
  (#qk:erased (Shared.heads 24)) (#data:erased Shared.attention_data)
  preserves gpu ** input |-> values
  requires buffer |-> old_buffer ** scores |-> old_scores
  requires pure (matrix_rep values (flat_heads qk) /\ Shared.valid_attention_data data)
  ensures exists* (scratch:lseq F32.t 128) (sv:lseq F32.t 7).
    buffer |-> scratch ** scores |-> sv **
    pure (sm_represents (row+1) sv (score_model qk data row head))
{
  open Pulse.Lib.Array;
  let mut index = 0sz;
  while (!index <^ (row +^ 1sz))
    invariant exists* (key:sz{key<=row+1}) (scratch:lseq F32.t 128) (sv:lseq F32.t 7).
      index |-> key ** buffer |-> scratch ** scores |-> sv **
      pure (forall (i:natlt key). Seq.index sv i %~ score_model qk data row head i)
    decreases (row+1 - !index)
  {
    let key = !index;
    with scratch. assert buffer |-> scratch;
    let dot = score_dot input buffer row key head #values #scratch #qk;
    let value = scaled_score dot
      #(Shared.dot ((reveal qk) row head) ((reveal qk) key (16+head/2)));
    score_model_at qk data row key head;
    with sv. assert scores |-> sv;
    scores_write row sv (score_model qk data row head) key value;
    scores.(key) <- value;
    index := key +^ 1sz;
  };
}

#set-options "--z3rlimit 10"

noextract let mixed_term
  (qkv:Shared.matrix 7 4096) (probability:Shared.vector 7)
  (row:natlt 7) (head:natlt 16) (col:natlt 128) (key:nat) : GTot real =
  if key <= row then probability key *. qkv key (3072+(head/2)*128+col) else 0.0R

[@@"opaque_to_smt"]
noextract let mixed_result
  (qkv:Shared.matrix 7 4096) (probability:Shared.vector 7)
  (row:natlt 7) (head:natlt 16) (col:natlt 128) : GTot real =
  Shared.sum (row+1) (fun key -> probability key *. qkv key (3072+(head/2)*128+col))

let mixed_result_definition
  (qkv:Shared.matrix 7 4096) (probability:Shared.vector 7)
  (row:natlt 7) (head:natlt 16) (col:natlt 128)
  : Lemma (mixed_result qkv probability row head col ==
    Shared.sum (row+1) (fun key -> probability key *. qkv key (3072+(head/2)*128+col))) =
  assert (mixed_result qkv probability row head col ==
    Shared.sum (row+1) (fun key -> probability key *. qkv key (3072+(head/2)*128+col))) by
    (FStar.Tactics.V2.norm [delta_only [`%mixed_result]];
     FStar.Tactics.V2.trivial ())

let mixed_result_sum
  (qkv:Shared.matrix 7 4096) (probability:Shared.vector 7)
  (row:natlt 7) (head:natlt 16) (col:natlt 128)
  : Lemma (mixed_result qkv probability row head col ==
    nr_sum_n (row+1) (mixed_term qkv probability row head col)) =
  mixed_result_definition qkv probability row head col;
  nr_sum_shared (row+1) (fun key -> probability key *. qkv key (3072+(head/2)*128+col))
    (mixed_term qkv probability row head col)

inline_for_extraction noextract
fn mixed_value
  (qkv:larray BF16.t 28672) (probability:larray BF16.t 7)
  (row:szlt 7) (head:szlt 16) (col:szlt 128)
  (#qv:erased (lseq BF16.t 28672)) (#pv:erased (lseq BF16.t 7))
  (#qr:erased (Shared.matrix 7 4096)) (#pr:erased (Shared.vector 7))
  preserves gpu ** qkv |-> qv ** probability |-> pv
  requires pure (matrix_rep qv qr /\ sm_represents (row+1) pv pr)
  returns result:BF16.t
  ensures pure (result %~ mixed_result qr pr row head col)
{
  open Pulse.Lib.Array;
  let mut index = 0sz;
  let mut acc:F32.t = zero #F32.t;
  while (!index <^ (row +^ 1sz))
    invariant exists* (key:sz{key<=row+1}) (value:F32.t).
      index |-> key ** acc |-> value **
      pure (value %~ nr_sum_n key (mixed_term qr pr row head col))
    decreases (row+1 - !index)
  {
    let key = !index;
    let p = probability.(key);
    let v = qkv.(key *^ 4096sz +^ 3072sz +^ (head /^ 2sz) *^ 128sz +^ col);
    matrix_at qv qr key (3072+(head/2)*128+col);
    Casts.fcast_approx #BF16.t #F32.t p ((reveal pr) key);
    Casts.fcast_approx #BF16.t #F32.t v ((reveal qr) key (3072+(head/2)*128+col));
    let pf:F32.t = Casts.fcast p;
    let vf:F32.t = Casts.fcast v;
    let previous = !acc;
    let value = Native.fma_rn pf vf previous;
    nr_fma_at pf vf previous value
      ((reveal pr) key) ((reveal qr) key (3072+(head/2)*128+col))
      (nr_sum_n key (mixed_term qr pr row head col));
    nr_sum_step (key+1) (mixed_term qr pr row head col);
    acc := value;
    index := key +^ 1sz;
  };
  let result = !acc;
  mixed_result_sum qr pr row head col;
  Casts.fcast_approx #F32.t #BF16.t result (mixed_result qr pr row head col);
  Casts.fcast result
}

[@@"opaque_to_smt"]
noextract let attention_model
  (qkv:Shared.matrix 7 4096) (qk:Shared.heads 24) (data:Shared.attention_data)
  : GTot (Shared.matrix 7 2048) =
  fun row col -> Shared.sum (row+1) (fun key ->
    Shared.prefix_softmax row (score_model qk data row (col/128)) key *.
      qkv key (3072+((col/128)/2)*128+col%128))

let attention_at
  (qkv:Shared.matrix 7 4096) (qk:Shared.heads 24) (data:Shared.attention_data)
  (row:natlt 7) (head:natlt 16) (lane:natlt 128)
  : Lemma (attention_model qkv qk data row (head*128+lane) ==
    mixed_result qkv (Shared.prefix_softmax row (score_model qk data row head)) row head lane) =
  let column = head*128+lane in
  mixed_result_definition qkv (Shared.prefix_softmax row (score_model qk data row head)) row head lane;
  Kuiper.Math.div_mod_of_mul_add 128 head lane;
  shared_sum_equal (row+1)
    (fun key -> Shared.prefix_softmax row (score_model qk data row (column/128)) key *.
      qkv key (3072+((column/128)/2)*128+column%128))
    (fun key -> Shared.prefix_softmax row (score_model qk data row head) key *.
      qkv key (3072+(head/2)*128+lane));
  assert (attention_model qkv qk data row column ==
    Shared.sum (row+1) (fun key ->
      Shared.prefix_softmax row (score_model qk data row (column/128)) key *.
        qkv key (3072+((column/128)/2)*128+column%128))) by
    (FStar.Tactics.V2.norm [delta_only [`%attention_model]];
     FStar.Tactics.V2.trivial ())

let attention_value_rep
  (qkv:Shared.matrix 7 4096) (qk:Shared.heads 24) (data:Shared.attention_data)
  (row:natlt 7) (head:natlt 16) (lane:natlt 128) (value:BF16.t)
  : Lemma
      (requires value %~
        mixed_result qkv (Shared.prefix_softmax row (score_model qk data row head)) row head lane)
      (ensures value %~ attention_model qkv qk data row (head*128+lane)) =
  attention_at qkv qk data row head lane

inline_for_extraction noextract
fn attend_kernel
  (qkv:larray BF16.t 28672) (qk:larray BF16.t 21504)
  (output:larray BF16.t 14336) (buffer:larray F32.t 128)
  (scores:larray F32.t 7) (probability:larray BF16.t 7)
  (#qv:erased (lseq BF16.t 28672)) (#kv:erased (lseq BF16.t 21504))
  (#old:erased (seq BF16.t)) (#bv:erased (seq F32.t))
  (#sv:erased (seq F32.t)) (#pv:erased (seq BF16.t))
  (#qr:erased (Shared.matrix 7 4096)) (#kr:erased (Shared.heads 24))
  (#data:erased Shared.attention_data)
  preserves gpu ** qkv |-> qv ** qk |-> kv
  requires output |-> old ** buffer |-> bv ** scores |-> sv ** probability |-> pv
  requires pure (matrix_rep qv qr /\ matrix_rep kv (flat_heads kr) /\
    Shared.valid_attention_data data)
  ensures exists* (values:lseq BF16.t 14336) (bval:seq F32.t)
    (sval:seq F32.t) (pval:seq BF16.t).
    output |-> values ** buffer |-> bval ** scores |-> sval ** probability |-> pval **
    pure (Seq.length bval == 128 /\ Seq.length sval == 7 /\ Seq.length pval == 7 /\
      matrix_rep values (attention_model qr kr data))
{
  open Pulse.Lib.Array;
  pts_to_len output;
  pts_to_len buffer;
  pts_to_len scores;
  pts_to_len probability;
  let mut index = 0sz;
  while (!index <^ 112sz)
    invariant exists* (j:sz{j<=112}) (values:lseq BF16.t 14336)
      (bval:lseq F32.t 128) (sval:lseq F32.t 7) (pval:lseq BF16.t 7).
      index |-> j ** output |-> values ** buffer |-> bval **
      scores |-> sval ** probability |-> pval **
      pure (matrix_prefix values (attention_model qr kr data) (j*128))
    decreases (112 - !index)
  {
    let j = !index;
    let row = j /^ 16sz;
    let head = j %^ 16sz;
    with bval. assert buffer |-> bval;
    with sval. assert scores |-> sval;
    with pval. assert probability |-> pval;
    scores_kernel qk buffer scores row head #kv #bval #sval #kr #data;
    with current_scores. assert scores |-> current_scores;
    sm_row_kernel scores probability row #current_scores #pval #(score_model kr data row head);
    with probabilities. assert probability |-> probabilities;
    let mut lane = 0sz;
    while (!lane <^ 128sz)
      invariant exists* (col:sz{col<=128}) (values:lseq BF16.t 14336).
        lane |-> col ** output |-> values **
        pure (matrix_prefix values (attention_model qr kr data) (j*128+col))
      decreases (128 - !lane)
    {
      let col = !lane;
      let value = mixed_value qkv probability row head col
        #qv #probabilities #qr #(Shared.prefix_softmax row (score_model kr data row head));
      attention_value_rep qr kr data row head col value;
      Kuiper.Math.div_mod_of_mul_add 2048 row (head*128+col);
      Kuiper.Math.div_mod_of_mul_add 128 head col;
      with values. assert output |-> values;
      matrix_write values (attention_model qr kr data) (j*128+col) value;
      output.(j *^ 128sz +^ col) <- value;
      lane := col +^ 1sz;
    };
    index := j +^ 1sz;
  };
}

noextract let qkv_weights (w:Shared.parameters) : GTot (Shared.matrix 4096 1024) =
  fun row col ->
    if row < 2048 then w.q_proj row col
    else if row < 3072 then w.k_proj (row-2048) col
    else w.v_proj (row-3072) col

noextract let gateup_weights (w:Shared.parameters) : GTot (Shared.matrix 6144 1024) =
  fun row col -> if row < 3072 then w.gate_proj row col else w.up_proj (row-3072) col

noextract let query_view (qk:Shared.heads 24) : GTot (Shared.heads 16) =
  fun row head col -> qk row head col

noextract let key_view (qk:Shared.heads 24) : GTot (Shared.heads 8) =
  fun row head col -> qk row (16+head) col

noextract let value_view (qkv:Shared.matrix 7 4096) : GTot (Shared.heads 8) =
  fun row head col -> qkv row (3072+head*128+col)

let qkv_channel (w:Shared.parameters) (x:Shared.hidden_states)
  (row:natlt 7) (col:natlt 4096)
  : Lemma (Shared.linear_rows (qkv_weights w) x row col ==
    (if col < 2048 then Shared.linear_rows w.q_proj x row col
     else if col < 3072 then Shared.linear_rows w.k_proj x row (col-2048)
     else Shared.linear_rows w.v_proj x row (col-3072))) =
  if col < 2048 then dot_equal (x row) (qkv_weights w col) (x row) (w.q_proj col)
  else if col < 3072 then
    dot_equal (x row) (qkv_weights w col) (x row) (w.k_proj (col-2048))
  else dot_equal (x row) (qkv_weights w col) (x row) (w.v_proj (col-3072))

let qkv_views (w:Shared.parameters) (x:Shared.hidden_states)
  : Lemma (
    (forall (row:natlt 7) (head:natlt 16) (col:natlt 128).
      Shared.linear_rows (qkv_weights w) x row (head*128+col) ==
        Shared.split_heads #16 (Shared.linear_rows w.q_proj x) row head col) /\
    (forall (row:natlt 7) (head:natlt 8) (col:natlt 128).
      Shared.linear_rows (qkv_weights w) x row (2048+head*128+col) ==
        Shared.split_heads #8 (Shared.linear_rows w.k_proj x) row head col /\
      value_view (Shared.linear_rows (qkv_weights w) x) row head col ==
        Shared.split_heads #8 (Shared.linear_rows w.v_proj x) row head col)) =
  let qp (row:natlt 7) (head:natlt 16) (col:natlt 128)
    : Lemma (Shared.linear_rows (qkv_weights w) x row (head*128+col) ==
        Shared.split_heads #16 (Shared.linear_rows w.q_proj x) row head col) =
    qkv_channel w x row (head*128+col)
  in
  let kp (row:natlt 7) (head:natlt 8) (col:natlt 128)
    : Lemma (Shared.linear_rows (qkv_weights w) x row (2048+head*128+col) ==
        Shared.split_heads #8 (Shared.linear_rows w.k_proj x) row head col) =
    qkv_channel w x row (2048+head*128+col)
  in
  let vp (row:natlt 7) (head:natlt 8) (col:natlt 128)
    : Lemma (value_view (Shared.linear_rows (qkv_weights w) x) row head col ==
        Shared.split_heads #8 (Shared.linear_rows w.v_proj x) row head col) =
    qkv_channel w x row (3072+head*128+col)
  in
  FStar.Classical.forall_intro_3 qp;
  FStar.Classical.forall_intro_3 kp;
  FStar.Classical.forall_intro_3 vp

let rotary_equal_at (#left_count #right_count:pos)
  (data:Shared.attention_data) (x:Shared.heads left_count) (y:Shared.heads right_count)
  (row:natlt 7) (left_head:natlt left_count) (right_head:natlt right_count) (col:natlt 128)
  : Lemma
      (requires forall (i:natlt 128). x row left_head i == y row right_head i)
      (ensures Shared.rotary data x row left_head col ==
        Shared.rotary data y row right_head col) = ()

let qk_views (w:Shared.parameters) (x:Shared.hidden_states) (data:Shared.attention_data)
  : Lemma (
    (forall (row:natlt 7) (head:natlt 16) (col:natlt 128).
      query_view (qk_model w (Shared.linear_rows (qkv_weights w) x) data) row head col ==
        Shared.rotary data (Shared.normalize_heads w.q_norm
          (Shared.split_heads #16 (Shared.linear_rows w.q_proj x))) row head col) /\
    (forall (row:natlt 7) (head:natlt 8) (col:natlt 128).
      key_view (qk_model w (Shared.linear_rows (qkv_weights w) x) data) row head col ==
        Shared.rotary data (Shared.normalize_heads w.k_norm
          (Shared.split_heads #8 (Shared.linear_rows w.k_proj x))) row head col)) =
  qkv_views w x;
  let packed = Shared.linear_rows (qkv_weights w) x in
  let normalized = normalized_qk w (Shared.linear_rows (qkv_weights w) x) in
  let query = Shared.normalize_heads w.q_norm (Shared.split_heads #16 (Shared.linear_rows w.q_proj x)) in
  let key = Shared.normalize_heads w.k_norm (Shared.split_heads #8 (Shared.linear_rows w.k_proj x)) in
  let nq (row:natlt 7) (head:natlt 16) (col:natlt 128)
    : Lemma (normalized row head col == query row head col) =
    normalized_qk_at w packed row head col;
    rms_equal (qk_gamma w head) w.q_norm
      (fun i -> packed row (head*128+i))
      (Shared.split_heads #16 (Shared.linear_rows w.q_proj x) row head) col
  in
  let nk (row:natlt 7) (head:natlt 8) (col:natlt 128)
    : Lemma (normalized row (16+head) col == key row head col) =
    normalized_qk_at w packed row (16+head) col;
    rms_equal (qk_gamma w (16+head)) w.k_norm
      (fun i -> packed row ((16+head)*128+i))
      (Shared.split_heads #8 (Shared.linear_rows w.k_proj x) row head) col
  in
  FStar.Classical.forall_intro_3 nq;
  FStar.Classical.forall_intro_3 nk;
  let qp (row:natlt 7) (head:natlt 16) (col:natlt 128)
    : Lemma (query_view (qk_model w packed data) row head col ==
        Shared.rotary data query row head col) =
    qk_rotary_at w packed data row head col;
    rotary_equal_at data normalized query row head head col
  in
  let kp (row:natlt 7) (head:natlt 8) (col:natlt 128)
    : Lemma (key_view (qk_model w packed data) row head col ==
        Shared.rotary data key row head col) =
    qk_rotary_at w packed data row (16+head) col;
    rotary_equal_at data normalized key row (16+head) head col
  in
  FStar.Classical.forall_intro_3 qp;
  FStar.Classical.forall_intro_3 kp

let shared_probability_at
  (data:Shared.attention_data) (query:Shared.heads 16) (key:Shared.heads 8)
  (row:natlt 7) (head:natlt 16) (k:natlt 7)
  : Lemma (Shared.attention_probabilities data query (Shared.repeat_kv key) row head k ==
    Shared.prefix_softmax row (fun t ->
      Shared.dot (query row head) (key t (head/2)) *. Root.rsqrt 128.0R) k) =
  let expanded : Shared.vector (row+1) = fun t ->
    Shared.dot (query row head) (fun lane -> key t (head/2) lane) *. Root.rsqrt 128.0R in
  let scores : Shared.vector (row+1) = fun t ->
    Shared.dot (query row head) (key t (head/2)) *. Root.rsqrt 128.0R in
  assert (Shared.attention_probabilities data query (Shared.repeat_kv key) row head k ==
    Shared.prefix_softmax row (fun t ->
      Shared.dot (query row head) (fun lane -> key t (head/2) lane) *. Root.rsqrt 128.0R) k) by
    (FStar.Tactics.V2.norm [delta_only [
      `%Shared.attention_probabilities; `%Shared.repeat_kv]];
     FStar.Tactics.V2.trivial ());
  let point (t:Shared.index (row+1))
    : Lemma (expanded t == scores t) =
    dot_equal (query row head) (fun lane -> key t (head/2) lane)
      (query row head) (key t (head/2))
  in
  FStar.Classical.forall_intro point;
  prefix_softmax_equal row expanded scores k

[@@"opaque_to_smt"]
noextract let shared_mix_result
  (probability:Shared.attention_rows) (value:Shared.heads 8)
  (row:natlt 7) (head:natlt 16) (lane:natlt 128) : GTot real =
  Shared.sum (row+1) (fun k -> probability row head k *. value k (head/2) lane)

let shared_mix_definition
  (probability:Shared.attention_rows) (value:Shared.heads 8)
  (row:natlt 7) (head:natlt 16) (lane:natlt 128)
  : Lemma (shared_mix_result probability value row head lane ==
    Shared.sum (row+1) (fun k -> probability row head k *. value k (head/2) lane)) =
  assert (shared_mix_result probability value row head lane ==
    Shared.sum (row+1) (fun k -> probability row head k *. value k (head/2) lane)) by
    (FStar.Tactics.V2.norm [delta_only [`%shared_mix_result]];
     FStar.Tactics.V2.trivial ())

let shared_mix_at
  (probability:Shared.attention_rows) (value:Shared.heads 8)
  (row:natlt 7) (col:natlt 2048)
  : Lemma (Shared.concatenate_heads
      (Shared.attention_values probability (Shared.repeat_kv value)) row col ==
    shared_mix_result probability value row (col/128) (col%128)) =
  shared_mix_definition probability value row (col/128) (col%128);
  assert (Shared.concatenate_heads
      (Shared.attention_values probability (Shared.repeat_kv value)) row col ==
    Shared.sum (row+1) (fun k -> probability row (col/128) k *. value k ((col/128)/2) (col%128))) by
    (FStar.Tactics.V2.norm [delta_only [
      `%Shared.concatenate_heads; `%Shared.attention_values; `%Shared.repeat_kv]];
     FStar.Tactics.V2.trivial ())

let mixed_result_equal
  (qkv:Shared.matrix 7 4096) (probability:Shared.vector 7)
  (reference:Shared.attention_rows) (value:Shared.heads 8)
  (row:natlt 7) (head:natlt 16) (lane:natlt 128)
  : Lemma
      (requires forall (k:natlt (row+1)).
        probability k == reference row head k /\
        qkv k (3072+(head/2)*128+lane) == value k (head/2) lane)
      (ensures mixed_result qkv probability row head lane ==
        shared_mix_result reference value row head lane) =
  mixed_result_definition qkv probability row head lane;
  shared_mix_definition reference value row head lane;
  shared_sum_equal (row+1)
    (fun k -> probability k *. qkv k (3072+(head/2)*128+lane))
    (fun k -> reference row head k *. value k (head/2) lane)

let packed_attention_math
  (packed:Shared.matrix 7 4096) (qk:Shared.heads 24) (data:Shared.attention_data)
  (query:Shared.heads 16) (key value:Shared.heads 8)
  : Lemma
      (requires
        (forall (row:natlt 7) (head:natlt 16) (lane:natlt 128).
          qk row head lane == query row head lane) /\
        (forall (row:natlt 7) (head:natlt 8) (lane:natlt 128).
          qk row (16+head) lane == key row head lane /\
          packed row (3072+head*128+lane) == value row head lane))
      (ensures forall (row:natlt 7) (col:natlt 2048).
        attention_model packed qk data row col ==
          Shared.concatenate_heads (Shared.attention_values
            (Shared.attention_probabilities data query (Shared.repeat_kv key))
            (Shared.repeat_kv value)) row col) =
  let point (row:natlt 7) (col:natlt 2048)
    : Lemma (attention_model packed qk data row col ==
      Shared.concatenate_heads (Shared.attention_values
        (Shared.attention_probabilities data query (Shared.repeat_kv key))
        (Shared.repeat_kv value)) row col) =
    let head = col/128 in
    attention_at packed qk data row head (col%128);
    let reference_score : Shared.vector (row+1) = fun k ->
      Shared.dot (query row head) (key k (head/2)) *. Root.rsqrt 128.0R in
    let score_at (k:natlt (row+1))
      : Lemma (score_model qk data row head k == reference_score k) =
      dot_equal (qk row head) (qk k (16+head/2)) (query row head) (key k (head/2))
    in
    FStar.Classical.forall_intro score_at;
    let probability_at (k:natlt 7)
      : Lemma (Shared.prefix_softmax row (score_model qk data row head) k ==
        Shared.attention_probabilities data query (Shared.repeat_kv key) row head k) =
      prefix_softmax_equal row (score_model qk data row head) reference_score k;
      shared_probability_at data query key row head k
    in
    FStar.Classical.forall_intro probability_at;
    mixed_result_equal packed (Shared.prefix_softmax row (score_model qk data row head))
      (Shared.attention_probabilities data query (Shared.repeat_kv key)) value row head (col%128);
    shared_mix_at (Shared.attention_probabilities data query (Shared.repeat_kv key))
      value row col
  in FStar.Classical.forall_intro_2 point

let gateup_channel (w:Shared.parameters) (x:Shared.hidden_states)
  (row:natlt 7) (col:natlt 6144)
  : Lemma (Shared.linear_rows (gateup_weights w) x row col ==
    (if col < 3072 then Shared.linear_rows w.gate_proj x row col
     else Shared.linear_rows w.up_proj x row (col-3072))) =
  if col < 3072 then dot_equal (x row) (gateup_weights w col) (x row) (w.gate_proj col)
  else dot_equal (x row) (gateup_weights w col) (x row) (w.up_proj (col-3072))

let gated_math (w:Shared.parameters) (x y:Shared.hidden_states)
  : Lemma
      (requires same_hidden x y)
      (ensures same_hidden
        (Shared.linear_rows w.down_proj (gated_model (Shared.linear_rows (gateup_weights w) x)))
        (Shared.feed_forward w y)) =
  linear_rows_equal w.gate_proj x y;
  linear_rows_equal w.up_proj x y;
  let reference_gated : Shared.matrix 7 3072 = fun row col ->
    Shared.silu (Shared.linear_rows w.gate_proj y row col) *.
      Shared.linear_rows w.up_proj y row col in
  let point (row:natlt 7) (col:natlt 3072)
    : Lemma (gated_model (Shared.linear_rows (gateup_weights w) x) row col ==
        reference_gated row col) =
    gateup_channel w x row col;
    gateup_channel w x row (3072+col)
  in
  FStar.Classical.forall_intro_2 point;
  linear_rows_equal w.down_proj (gated_model (Shared.linear_rows (gateup_weights w) x))
    reference_gated

[@@erasable]
noeq
type decoder_stages = {
  hidden: Shared.hidden_states;
  normalized: Shared.hidden_states;
  qkv: Shared.matrix 7 4096;
  qk: Shared.heads 24;
  mixed: Shared.matrix 7 2048;
  projected: Shared.hidden_states;
  residual: Shared.hidden_states;
  postnorm: Shared.hidden_states;
  gateup: Shared.matrix 7 6144;
  gated: Shared.matrix 7 3072;
  delta: Shared.hidden_states
}

[@@"opaque_to_smt"]
noextract let build_stages (input:Shared.prefill_input)
  : GTot (s:decoder_stages{
    s.hidden == input.hidden /\
    s.normalized == Shared.normalize_hidden input.weight.input_layernorm s.hidden /\
    s.qkv == Shared.linear_rows (qkv_weights input.weight) s.normalized /\
    s.qk == qk_model input.weight s.qkv input.data /\
    s.mixed == attention_model s.qkv s.qk input.data /\
    s.projected == Shared.linear_rows input.weight.o_proj s.mixed /\
    s.residual == Shared.residual_add s.hidden s.projected /\
    s.postnorm == Shared.normalize_hidden input.weight.post_attention_layernorm s.residual /\
    s.gateup == Shared.linear_rows (gateup_weights input.weight) s.postnorm /\
    s.gated == gated_model s.gateup /\
    s.delta == Shared.linear_rows input.weight.down_proj s.gated /\
    same_hidden s.residual (decoder_parts input).residual /\
    same_hidden s.delta (decoder_parts input).delta /\
    (forall (row:natlt 7) (head:natlt 8) (lane:natlt 128).
      key_view s.qk row head lane == (decoder_parts input).keys row head lane /\
      value_view s.qkv row head lane == (decoder_parts input).values row head lane)}) =
  let w = input.weight in
  let normalized = Shared.normalize_hidden w.input_layernorm input.hidden in
  let qkv = Shared.linear_rows (qkv_weights w) normalized in
  let qk = qk_model w qkv input.data in
  let mixed = attention_model qkv qk input.data in
  let projected = Shared.linear_rows w.o_proj mixed in
  let residual = Shared.residual_add input.hidden projected in
  let postnorm = Shared.normalize_hidden w.post_attention_layernorm residual in
  let gateup = Shared.linear_rows (gateup_weights w) postnorm in
  let gated = gated_model gateup in
  let delta = Shared.linear_rows w.down_proj gated in
  qkv_views w normalized;
  qk_views w normalized input.data;
  let query = Shared.rotary input.data (Shared.normalize_heads w.q_norm
    (Shared.split_heads #16 (Shared.linear_rows w.q_proj normalized))) in
  let key = Shared.rotary input.data (Shared.normalize_heads w.k_norm
    (Shared.split_heads #8 (Shared.linear_rows w.k_proj normalized))) in
  let value = Shared.split_heads #8 (Shared.linear_rows w.v_proj normalized) in
  assert (key == (decoder_parts input).keys);
  assert (value == (decoder_parts input).values);
  packed_attention_math qkv qk input.data query key value;
  let reference_mixed = Shared.concatenate_heads (Shared.attention_values
    (Shared.attention_probabilities input.data query (Shared.repeat_kv key))
    (Shared.repeat_kv value)) in
  linear_rows_equal w.o_proj mixed reference_mixed;
  let reference_residual = Shared.residual_add input.hidden
    (Shared.linear_rows w.o_proj reference_mixed) in
  assert (reference_residual == (decoder_parts input).residual);
  assert (same_hidden residual (decoder_parts input).residual);
  let reference_postnorm = Shared.normalize_hidden w.post_attention_layernorm
    (decoder_parts input).residual in
  let norm_at (row:natlt 7) (col:natlt 1024)
    : Lemma (postnorm row col == reference_postnorm row col) =
    rms_equal w.post_attention_layernorm w.post_attention_layernorm
      (residual row) ((decoder_parts input).residual row) col
  in
  FStar.Classical.forall_intro_2 norm_at;
  gated_math w postnorm reference_postnorm;
  { hidden = input.hidden; normalized = normalized; qkv = qkv; qk = qk;
    mixed = mixed; projected = projected; residual = residual; postnorm = postnorm;
    gateup = gateup; gated = gated; delta = delta }

noextract let stage_pending (stages:decoder_stages) : GTot pending_output = {
  residual = stages.residual;
  delta = stages.delta;
  keys = key_view stages.qk;
  values = value_view stages.qkv
}

let computed_pending_refines_shared (input:Shared.prefill_input)
  : Lemma (same_pending (stage_pending (build_stages input)) (decoder_parts input)) = ()

let record_computed_cache
  (stages:decoder_stages) (qkv:lseq BF16.t 28672) (qk:lseq BF16.t 21504)
  : Lemma
      (requires matrix_rep qkv stages.qkv /\ matrix_rep qk (flat_heads stages.qk))
      (ensures computed_cache_representation qkv qk (stage_pending stages)) =
  let point (row:natlt 7) (head:natlt 8) (lane:natlt 128)
    : Lemma
        (Seq.index qk (row*3072+2048+head*128+lane) %~ key_view stages.qk row head lane /\
         Seq.index qkv (row*4096+3072+head*128+lane) %~ value_view stages.qkv row head lane) =
    qk_read qk stages.qk row (16+head) lane;
    matrix_at qkv stages.qkv row (3072+head*128+lane)
  in FStar.Classical.forall_intro_3 point

inline_for_extraction noextract
ghost
fn snapshot_computed_cache
  (qkv:bf16_array 28672) (qk:bf16_array 21504)
  (#qkv_values:erased (bf16_values 28672))
  (#qk_values:erased (bf16_values 21504))
  (#fqkv #fqk:perm)
  preserves
    on gpu_loc (qkv |-> Frac fqkv qkv_values ** qk |-> Frac fqk qk_values)
  returns observation:erased cache_observation
  ensures pure (
    (reveal observation).qkv_values == reveal qkv_values /\
    (reveal observation).qk_values == reveal qk_values)
{
  hide { qkv_values = reveal qkv_values; qk_values = reveal qk_values }
}

inline_for_extraction noextract
let qkv_plan : p:mm_linear_parameters{mm_linear_valid p} =
  { width = 1024sz; columns = 4096sz; splits = 1sz }

inline_for_extraction noextract
let output_plan : p:mm_linear_parameters{mm_linear_valid p} =
  { width = 2048sz; columns = 1024sz; splits = 1sz }

inline_for_extraction noextract
let gateup_plan : p:mm_linear_parameters{mm_linear_valid p} =
  { width = 1024sz; columns = 6144sz; splits = 1sz }

inline_for_extraction noextract
let down_plan : p:mm_linear_parameters{mm_linear_valid p} =
  { width = 3072sz; columns = 1024sz; splits = 2sz }

let packed_weight_views (#n:nat)
  (values:lseq BF16.t n) (offsets:weight_offsets{valid_weight_offsets n offsets})
  (weights:Shared.parameters)
  : Lemma
      (requires weights_representation values offsets weights)
      (ensures
        mm_weights_rep qkv_plan values offsets.qkv (qkv_weights weights) /\
        mm_weights_rep output_plan values offsets.attention_output weights.o_proj /\
        mm_weights_rep gateup_plan values offsets.gate_up (gateup_weights weights) /\
        mm_weights_rep down_plan values offsets.down weights.down_proj) = ()

let linear_input_view (p:mm_linear_parameters{mm_linear_valid p})
  (values:lseq BF16.t (7*SZ.v p.width)) (logical:Shared.matrix 7 (SZ.v p.width))
  : Lemma
      (requires matrix_rep values logical)
      (ensures mm_input_rep p values logical) = ()

let linear_output_view (p:mm_linear_parameters{mm_linear_valid p})
  (values:lseq BF16.t (7*SZ.v p.columns))
  (weights:Shared.matrix (SZ.v p.columns) (SZ.v p.width))
  (input:Shared.matrix 7 (SZ.v p.width))
  : Lemma
      (requires mm_output_prefix p values weights input (SZ.v p.columns))
      (ensures matrix_rep values (Shared.linear_rows weights input)) = ()

module IC = FStar.Int.Cast

inline_for_extraction noextract
let offset_size (offset:I64.t{0<=I64.v offset /\ I64.v offset<=2147483647})
  : Tot (n:sz{SZ.v n == I64.v offset}) =
  SZ.uint32_to_sizet (IC.int64_to_uint32 offset)

fn kplay_run
  (weight_count:U64.t{U64.v weight_count <= 2147483647})
  (input0:bf16_array (U64.v weight_count){is_global_array input0})
  (input1:bf16_array hidden_elements{is_global_array input1})
  (input2:bf16_array hidden_elements{is_global_array input2})
  (input3:bf16_array rotary_elements{is_global_array input3})
  (input4 input5 input6 input7 input8 input9:I64.t)
  (input10:I64.t{
    valid_weight_offsets (U64.v weight_count)
      (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10)})
  (input11:bool)
  (output0:bf16_array hidden_elements{is_global_array output0})
  (output1:bf16_array hidden_elements{is_global_array output1})
  (#weights:erased (bf16_values (U64.v weight_count)))
  (#residual #delta:erased (bf16_values hidden_elements))
  (#rotary:erased (bf16_values rotary_elements))
  (#old_output0 #old_output1:erased (bf16_values hidden_elements))
  (#f0 #f1 #f2 #f3:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  preserves cpu **
    on gpu_loc (input0 |-> Frac f0 weights) **
    on gpu_loc (input1 |-> Frac f1 residual) **
    on gpu_loc (input2 |-> Frac f2 delta) **
    on gpu_loc (input3 |-> Frac f3 rotary)
  requires on gpu_loc (output0 |-> old_output0 ** output1 |-> old_output1) **
    pure (input_representation weights residual delta rotary
      (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10)
      input11 logical)
  returns observation:erased cache_observation
  ensures exists* (result0:bf16_values hidden_elements) (result1:bf16_values hidden_elements).
    on gpu_loc (output0 |-> result0 ** output1 |-> result1) **
    pure (semantic_postcondition (input_mapping logical input11 layer before) result0 result1 (reveal observation))
{
  let pre_offset = offset_size input4;
  let qkv_offset = offset_size input5;
  let qk_offset = offset_size input6;
  let output_offset = offset_size input7;
  let post_offset = offset_size input8;
  let gate_offset = offset_size input9;
  let down_offset = offset_size input10;
  let mapped = hide (input_mapping logical input11 layer before);
  let stages = hide (build_stages mapped);
  let pending = hide (stage_pending (reveal stages));
  let wr = hide (reveal logical).weights;
  packed_weight_views weights
    (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10) wr;
  let hidden = gpu_array_alloc #BF16.t 7168sz;
  let normalized = gpu_array_alloc #BF16.t 7168sz;
  let qkv = gpu_array_alloc #BF16.t 28672sz;
  let qk = gpu_array_alloc #BF16.t 21504sz;
  let normalized_head = gpu_array_alloc #BF16.t 128sz;
  let mixed = gpu_array_alloc #BF16.t 14336sz;
  let projected = gpu_array_alloc #BF16.t 7168sz;
  let postnorm = gpu_array_alloc #BF16.t 7168sz;
  let gateup = gpu_array_alloc #BF16.t 43008sz;
  let gated = gpu_array_alloc #BF16.t 21504sz;
  let buffer = gpu_array_alloc #F32.t 128sz;
  let scores = gpu_array_alloc #F32.t 7sz;
  let probability = gpu_array_alloc #BF16.t 7sz;

  with (h0:lseq BF16.t 7168). assert on gpu_loc (hidden |-> h0);
  launch_kernel_1 (fun () ->
    combine_hidden input11 input1 input2 hidden #residual #delta
      #(hide (reveal h0 <: seq BF16.t))
      #(hide (reveal logical).incoming_residual) #(hide (reveal logical).incoming_delta) #f1 #f2);
  with (hv:lseq BF16.t 7168). assert on gpu_loc (hidden |-> hv);
  with (n0:lseq BF16.t 7168). assert on gpu_loc (normalized |-> n0);
  with (b0:seq F32.t). assert on gpu_loc (buffer |-> b0);
  launch_kernel_1 (fun () ->
    normalize_hidden_kernel input0 hidden normalized buffer pre_offset #weights
      #(hide (reveal hv <: lseq BF16.t 7168))
      #(hide (reveal n0 <: seq BF16.t))
      #(hide (reveal b0 <: seq F32.t))
      #(hide (reveal wr).input_layernorm) #(hide (reveal stages).hidden) #f0);
  with (nv:lseq BF16.t 7168). assert on gpu_loc (normalized |-> nv);
  with (q0:lseq BF16.t 28672). assert on gpu_loc (qkv |-> q0);
  linear_input_view qkv_plan nv (reveal stages).normalized;
  mm_run_linear qkv_plan input0 normalized qkv qkv_offset #weights
    #(hide (reveal nv <: lseq BF16.t 7168))
    #(hide (reveal q0 <: seq BF16.t)) #f0 #1.0R
    #(qkv_weights wr) #(hide (reveal stages).normalized);
  with (qv:lseq BF16.t 28672). assert on gpu_loc (qkv |-> qv);
  linear_output_view qkv_plan qv (qkv_weights wr) (reveal stages).normalized;

  with (k0:lseq BF16.t 21504). assert on gpu_loc (qk |-> k0);
  with (nh0:seq BF16.t). assert on gpu_loc (normalized_head |-> nh0);
  with (b1:seq F32.t). assert on gpu_loc (buffer |-> b1);
  launch_kernel_1 (fun () ->
    normalize_rotate_kernel input0 qkv input3 qk normalized_head buffer qk_offset
      #weights #(hide (reveal qv <: lseq BF16.t 28672)) #rotary
      #(hide (reveal k0 <: seq BF16.t))
      #(hide (reveal nh0 <: seq BF16.t))
      #(hide (reveal b1 <: seq F32.t)) #wr #(hide (reveal stages).qkv)
      #(hide (reveal logical).attention) #f0 #f3);
  with (kv:lseq BF16.t 21504). assert on gpu_loc (qk |-> kv);
  record_computed_cache stages qv kv;
  let observation = snapshot_computed_cache qkv qk #qv #kv #1.0R #1.0R;
  with (a0:lseq BF16.t 14336). assert on gpu_loc (mixed |-> a0);
  with (b2:seq F32.t). assert on gpu_loc (buffer |-> b2);
  with (s0:seq F32.t). assert on gpu_loc (scores |-> s0);
  with (p0:seq BF16.t). assert on gpu_loc (probability |-> p0);
  launch_kernel_1 (fun () ->
    attend_kernel qkv qk mixed buffer scores probability
      #(hide (reveal qv <: lseq BF16.t 28672))
      #(hide (reveal kv <: lseq BF16.t 21504))
      #(hide (reveal a0 <: seq BF16.t))
      #(hide (reveal b2 <: seq F32.t))
      #(hide (reveal s0 <: seq F32.t))
      #(hide (reveal p0 <: seq BF16.t))
      #(hide (reveal stages).qkv) #(hide (reveal stages).qk) #(hide (reveal logical).attention));
  with (av:lseq BF16.t 14336). assert on gpu_loc (mixed |-> av);
  with (o0:lseq BF16.t 7168). assert on gpu_loc (projected |-> o0);
  linear_input_view output_plan av (reveal stages).mixed;
  mm_run_linear output_plan input0 mixed projected output_offset #weights
    #(hide (reveal av <: lseq BF16.t 14336))
    #(hide (reveal o0 <: seq BF16.t)) #f0 #1.0R
    #(hide (reveal wr).o_proj) #(hide (reveal stages).mixed);
  with (ov:lseq BF16.t 7168). assert on gpu_loc (projected |-> ov);
  linear_output_view output_plan ov (reveal wr).o_proj (reveal stages).mixed;
  launch_kernel_1 (fun () ->
    combine_hidden true hidden projected output0
      #(hide (reveal hv <: lseq BF16.t 7168))
      #(hide (reveal ov <: lseq BF16.t 7168))
      #(hide (reveal old_output0 <: seq BF16.t))
      #(hide (reveal stages).hidden) #(hide (reveal stages).projected) #1.0R #1.0R);
  with (result0:lseq BF16.t 7168). assert on gpu_loc (output0 |-> result0);
  with (pn0:lseq BF16.t 7168). assert on gpu_loc (postnorm |-> pn0);
  with (b3:seq F32.t). assert on gpu_loc (buffer |-> b3);
  launch_kernel_1 (fun () ->
    normalize_hidden_kernel input0 output0 postnorm buffer post_offset
      #weights #(hide (reveal result0 <: lseq BF16.t 7168))
      #(hide (reveal pn0 <: seq BF16.t))
      #(hide (reveal b3 <: seq F32.t)) #(hide (reveal wr).post_attention_layernorm)
      #(hide (reveal stages).residual) #f0);
  with (pnv:lseq BF16.t 7168). assert on gpu_loc (postnorm |-> pnv);
  with (gu0:lseq BF16.t 43008). assert on gpu_loc (gateup |-> gu0);
  linear_input_view gateup_plan pnv (reveal stages).postnorm;
  mm_run_linear gateup_plan input0 postnorm gateup gate_offset #weights
    #(hide (reveal pnv <: lseq BF16.t 7168))
    #(hide (reveal gu0 <: seq BF16.t)) #f0 #1.0R
    #(gateup_weights wr) #(hide (reveal stages).postnorm);
  with (guv:lseq BF16.t 43008). assert on gpu_loc (gateup |-> guv);
  linear_output_view gateup_plan guv (gateup_weights wr) (reveal stages).postnorm;
  with (g0:lseq BF16.t 21504). assert on gpu_loc (gated |-> g0);
  launch_kernel_1 (fun () -> silu_kernel gateup gated
    #(hide (reveal guv <: lseq BF16.t 43008))
    #(hide (reveal g0 <: seq BF16.t)) #(hide (reveal stages).gateup));
  with (gv:lseq BF16.t 21504). assert on gpu_loc (gated |-> gv);
  linear_input_view down_plan gv (reveal stages).gated;
  mm_run_linear down_plan input0 gated output1 down_offset #weights
    #(hide (reveal gv <: lseq BF16.t 21504))
    #(hide (reveal old_output1 <: seq BF16.t)) #f0 #1.0R
    #(hide (reveal wr).down_proj) #(hide (reveal stages).gated);
  with (result1:lseq BF16.t 7168). assert on gpu_loc (output1 |-> result1);
  linear_output_view down_plan result1 (reveal wr).down_proj (reveal stages).gated;
  computed_pending_refines_shared mapped;
  assert pure (computed_cache_representation qv kv pending);
  computed_cache_refines_shared mapped pending qv kv;
  mapped_cache_shape_and_frame mapped pending;
  computed_outputs_establish_semantic_postcondition mapped pending result0 result1 (reveal observation);
  gpu_array_free hidden;
  gpu_array_free normalized;
  gpu_array_free qkv;
  gpu_array_free qk;
  gpu_array_free normalized_head;
  gpu_array_free mixed;
  gpu_array_free projected;
  gpu_array_free postnorm;
  gpu_array_free gateup;
  gpu_array_free gated;
  gpu_array_free buffer;
  gpu_array_free scores;
  gpu_array_free probability;
  observation
}

inline_for_extraction noextract
fn kplay_run_safety
  (weight_count:U64.t{U64.v weight_count <= 2147483647})
  (input0:bf16_array (U64.v weight_count){is_global_array input0})
  (input1:bf16_array hidden_elements{is_global_array input1})
  (input2:bf16_array hidden_elements{is_global_array input2})
  (input3:bf16_array rotary_elements{is_global_array input3})
  (input4 input5 input6 input7 input8 input9:I64.t)
  (input10:I64.t{
    valid_weight_offsets (U64.v weight_count)
      (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10)})
  (input11:bool)
  (output0:bf16_array hidden_elements{is_global_array output0})
  (output1:bf16_array hidden_elements{is_global_array output1})
  (#weights:erased (bf16_values (U64.v weight_count)))
  (#residual #delta:erased (bf16_values hidden_elements))
  (#rotary:erased (bf16_values rotary_elements))
  (#old_output0 #old_output1:erased (bf16_values hidden_elements))
  (#f0 #f1 #f2 #f3:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  preserves cpu **
    on gpu_loc (input0 |-> Frac f0 weights) **
    on gpu_loc (input1 |-> Frac f1 residual) **
    on gpu_loc (input2 |-> Frac f2 delta) **
    on gpu_loc (input3 |-> Frac f3 rotary)
  requires on gpu_loc (output0 |-> old_output0 ** output1 |-> old_output1) **
    pure (input_representation weights residual delta rotary
      (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10)
      input11 logical)
  ensures exists* (result0:bf16_values hidden_elements) (result1:bf16_values hidden_elements).
    on gpu_loc (output0 |-> result0 ** output1 |-> result1)
{
  let _ = kplay_run weight_count input0 input1 input2 input3
    input4 input5 input6 input7 input8 input9 input10 input11 output0 output1
    #weights #residual #delta #rotary #old_output0 #old_output1
    #f0 #f1 #f2 #f3 #logical #layer #before;
  ()
}
