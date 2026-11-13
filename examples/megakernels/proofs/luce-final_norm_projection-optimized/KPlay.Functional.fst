module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module R = FStar.Real
module Sqrt = FStar.Math.Sqrt
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module Cast = Kuiper.Float.Casts
module C = FStar.Classical
module SC = Kuiper.Seq.Common
module Approx = Kuiper.Approximates.Base
module Shared = KPlay.SharedSpec
module Sum = Kuiper.Sum

#set-options "--fuel 2 --ifuel 1 --z3rlimit 30"

inline_for_extraction noextract
let rec prefix (n : nat) (f : nat -> GTot real)
  : GTot real (decreases n)
  = if n = 0 then 0.0R else prefix (n - 1) f +. f (n - 1)

inline_for_extraction noextract
let prefix_step (n : nat) (f : nat -> GTot real)
  : Lemma (prefix (n + 1) f == prefix n f +. f n)
  = ()

inline_for_extraction noextract
let rec prefix_ext (n : nat) (f g : nat -> GTot real)
  : Lemma
      (requires forall (i : nat). i < n ==> f i == g i)
      (ensures prefix n f == prefix n g)
      (decreases n)
  = if n > 0 then prefix_ext (n - 1) f g

inline_for_extraction noextract
let rec prefix_add (n : nat) (f g : nat -> GTot real)
  : Lemma
      (ensures prefix n (fun i -> f i +. g i) == prefix n f +. prefix n g)
      (decreases n)
  = if n > 0 then prefix_add (n - 1) f g

inline_for_extraction noextract
let rec prefix_zero (n : nat)
  : Lemma (ensures prefix n (fun _ -> 0.0R) == 0.0R) (decreases n)
  = if n > 0 then prefix_zero (n - 1)

inline_for_extraction noextract
let rec prefix_split (n m : nat) (f : nat -> GTot real)
  : Lemma
      (ensures prefix (n + m) f ==
        prefix n f +. prefix m (fun i -> f (n + i)))
      (decreases m)
  = if m > 0 then prefix_split n (m - 1) f

inline_for_extraction noextract
let rec prefix_swap (n m : nat) (f : nat -> nat -> GTot real)
  : Lemma
      (ensures prefix n (fun i -> prefix m (f i)) ==
        prefix m (fun j -> prefix n (fun i -> f i j)))
      (decreases n)
  = if n = 0 then begin
      prefix_zero m;
      prefix_ext m (fun j -> prefix n (fun i -> f i j)) (fun _ -> 0.0R)
    end
    else begin
      prefix_swap (n - 1) m f;
      prefix_add m (fun j -> prefix (n - 1) (fun i -> f i j)) (f (n - 1));
      prefix_ext m
        (fun j -> prefix n (fun i -> f i j))
        (fun j -> prefix (n - 1) (fun i -> f i j) +. f (n - 1) j)
    end

inline_for_extraction noextract
let rec prefix_blocks (blocks width : nat) (f : nat -> GTot real)
  : Lemma
      (ensures prefix blocks (fun b ->
        prefix width (fun i -> f (b * width + i))) ==
        prefix (blocks * width) f)
      (decreases blocks)
  = if blocks > 0 then begin
      prefix_step (blocks - 1)
        (fun b -> prefix width (fun i -> f (b * width + i)));
      prefix_blocks (blocks - 1) width f;
      prefix_split ((blocks - 1) * width) width f;
      assert (blocks * width == (blocks - 1) * width + width)
    end

inline_for_extraction noextract
let seq_value (s : seq real) (i : nat) : GTot real =
  if i < S.length s then S.index s i else 0.0R

inline_for_extraction noextract
let rec prefix_rsum (s : seq real) (n : nat{n <= S.length s})
  : Lemma
      (ensures prefix n (seq_value s) == rsum (S.slice s 0 n))
      (decreases n)
  = if n = 0 then
      assert (S.equal (S.slice s 0 0) (S.empty #real))
    else begin
      prefix_rsum s (n - 1);
      SC.lemma_seq_fold_left_slice 0.0R (+.) s 0 (n - 1)
    end

inline_for_extraction noextract
let prefix_init (n : nat) (f : nat -> GTot real)
  : Lemma (prefix n f == rsum (S.init_ghost n f))
  = let s = S.init_ghost n f in
    prefix_ext n f (seq_value s);
    prefix_rsum s n;
    assert (S.equal (S.slice s 0 n) s)

inline_for_extraction noextract
let tree_width (n : nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32

inline_for_extraction noextract
let rec tree_value
  (f : nat -> GTot real)
  (width : nat{tree_width width})
  (lane : nat)
  : GTot real (decreases 32 - width)
  = if width = 32 then f lane
    else tree_value f (2 * width) lane +.
      tree_value f (2 * width) (lane + width)

inline_for_extraction noextract
let rec tree_total
  (f : nat -> GTot real)
  (width : nat{tree_width width})
  : Lemma
      (ensures prefix width (tree_value f width) == prefix 32 f)
      (decreases 32 - width)
  = if width = 32 then prefix_ext 32 (tree_value f 32) f
    else begin
      tree_total f (2 * width);
      prefix_ext width (tree_value f width)
        (fun i -> tree_value f (2 * width) i +.
          tree_value f (2 * width) (i + width));
      prefix_add width (tree_value f (2 * width))
        (fun i -> tree_value f (2 * width) (i + width));
      prefix_split width width (tree_value f (2 * width));
      prefix_ext width
        (fun i -> tree_value f (2 * width) (i + width))
        (fun i -> tree_value f (2 * width) (width + i))
    end

inline_for_extraction noextract
let tree_root (f : nat -> GTot real)
  : Lemma (tree_value f 1 0 == prefix 32 f)
  = tree_total f 1

inline_for_extraction noextract
let lift_sum32
  (#parameter : Type)
  (premise : parameter -> prop)
  (model : parameter -> nat -> GTot real)
  (values : lseq f32 32)
  (result : f32)
  : Lemma
      (requires
        (forall (p : parameter). premise p ==>
          (forall (i : nat{i < 32}). S.index values i %~ model p i)) /\
        (forall (r : lseq real 32).
          values %~ r ==> result %~ prefix 32 (seq_value r)))
      (ensures forall (p : parameter).
        premise p ==> result %~ prefix 32 (model p))
  = let aux (p : parameter{premise p})
      : Lemma (result %~ prefix 32 (model p))
      = let r = S.init_ghost 32 (model p) in
        assert (values %~ r);
        prefix_ext 32 (seq_value r) (model p)
    in
    C.forall_intro aux

inline_for_extraction noextract
let lift_fma
  (#parameter : Type)
  (premise : parameter -> prop)
  (xm ym zm : parameter -> GTot real)
  (x y z result : f32)
  : Lemma
      (requires
        (forall (p : parameter). premise p ==>
          x %~ xm p /\ y %~ ym p /\ z %~ zm p) /\
        (forall (xr yr zr : real).
          x %~ xr /\ y %~ yr /\ z %~ zr ==>
          result %~ (xr *. yr +. zr)))
      (ensures forall (p : parameter). premise p ==>
        result %~ (xm p *. ym p +. zm p))
  = let aux (p : parameter{premise p})
      : Lemma (result %~ (xm p *. ym p +. zm p))
      = ()
    in
    C.forall_intro aux

inline_for_extraction noextract
let lift_add
  (#parameter : Type)
  (premise : parameter -> prop)
  (xm ym : parameter -> GTot real)
  (x y result : f32)
  : Lemma
      (requires
        (forall (p : parameter). premise p ==> x %~ xm p /\ y %~ ym p) /\
        (forall (xr yr : real).
          x %~ xr /\ y %~ yr ==> result %~ (xr +. yr)))
      (ensures forall (p : parameter). premise p ==> result %~ (xm p +. ym p))
  = let aux (p : parameter{premise p})
      : Lemma (result %~ (xm p +. ym p))
      = ()
    in
    C.forall_intro aux

inline_for_extraction noextract
let lift_mul
  (#parameter : Type)
  (premise : parameter -> prop)
  (xm ym : parameter -> GTot real)
  (x y result : f32)
  : Lemma
      (requires
        (forall (p : parameter). premise p ==> x %~ xm p /\ y %~ ym p) /\
        (forall (xr yr : real).
          x %~ xr /\ y %~ yr ==> result %~ (xr *. yr)))
      (ensures forall (p : parameter). premise p ==> result %~ (xm p *. ym p))
  = let aux (p : parameter{premise p})
      : Lemma (result %~ (xm p *. ym p))
      = ()
    in
    C.forall_intro aux

inline_for_extraction noextract
let inverse_width () : Pure f32
  (requires True)
  (ensures fun result -> result %~ (1.0R /. 1024.0R))
  = let numerator : f32 = of_int 1L in
    let denominator : f32 = of_int 1024L in
    let _ = of_int_approx #f32 1L in
    let _ = of_int_approx #f32 1024L in
    div_approx numerator denominator 1.0R 1024.0R;
    div numerator denominator

(* Constant division is folded to the accepted binary32 literal by NVCC. *)
inline_for_extraction noextract
let epsilon32 () : Pure f32
  (requires True)
  (ensures fun result -> result %~ 0.000001R)
  = let numerator : f32 = of_int 1L in
    let denominator : f32 = of_int 1000000L in
    let _ = of_int_approx #f32 1L in
    let _ = of_int_approx #f32 1000000L in
    div_approx numerator denominator 1.0R 1000000.0R;
    div numerator denominator

inline_for_extraction noextract
let tree_invariant
  (original current : lseq f32 32)
  (width : nat{tree_width width})
  : prop
  = forall (model : lseq real 32).
      original %~ model ==>
      (forall (i : nat{i < width}).
        S.index current i %~ tree_value (seq_value model) width i)

inline_for_extraction noextract
let tree_progress
  (original current : lseq f32 32)
  (distance : nat{tree_width distance /\ distance < 32})
  (done : nat{done <= distance})
  : prop
  = forall (model : lseq real 32).
      original %~ model ==>
      (forall (i : nat{i < 2 * distance}).
        S.index current i %~
          (if i < done then tree_value (seq_value model) distance i
           else tree_value (seq_value model) (2 * distance) i))

inline_for_extraction noextract
let tree_initial (values : lseq f32 32)
  : Lemma (tree_invariant values values 32)
  = ()

inline_for_extraction noextract
let tree_step
  (original current : lseq f32 32)
  (distance : nat{tree_width distance /\ distance < 32})
  (i : nat{i < distance})
  (value : f32)
  : Lemma
      (requires tree_progress original current distance i /\
        (forall (xr yr : real).
          S.index current i %~ xr /\
          S.index current (i + distance) %~ yr ==>
          value %~ (xr +. yr)))
      (ensures tree_progress original (S.upd current i value) distance (i + 1))
  = ()

inline_for_extraction noextract
let tree_finished (original current : lseq f32 32)
  : Lemma
      (requires tree_invariant original current 1)
      (ensures forall (model : lseq real 32).
        original %~ model ==>
        S.index current 0 %~ prefix 32 (seq_value model))
  = let aux (model : lseq real 32)
      : Lemma (tree_value (seq_value model) 1 0 ==
        prefix 32 (seq_value model))
      = tree_root (seq_value model)
    in
    C.forall_intro aux

inline_for_extraction noextract
fn reduce_level
  (partial : larray f32 32)
  (distance : sz{tree_width (SZ.v distance) /\ distance < 32})
  (#original #before : erased (lseq f32 32))
  preserves gpu
  requires partial |-> before **
    pure (tree_invariant original before (2 * SZ.v distance))
  ensures exists* (after : lseq f32 32).
    partial |-> after ** pure (tree_invariant original after (SZ.v distance))
{
  let mut lane = 0sz;
  while (!lane <^ distance)
    invariant exists* (i : sz{i <= distance}) (values : lseq f32 32).
      lane |-> i ** partial |-> values **
      pure (tree_progress original values (SZ.v distance) (SZ.v i))
    decreases (distance - !lane)
  {
    let i = !lane;
    with values. assert partial |-> values;
    let a = partial.(i);
    let b = partial.(i +^ distance);
    let value = F32.add_rn_ftz a b;
    tree_step original values (SZ.v distance) (SZ.v i) value;
    partial.(i) <- value;
    lane := i +^ 1sz;
  }
}

inline_for_extraction noextract
fn sum32
  (partial : larray f32 32)
  (#before : erased (lseq f32 32))
  preserves gpu
  requires partial |-> before
  returns result : f32
  ensures exists* (after : lseq f32 32).
    partial |-> after **
    pure (forall (model : lseq real 32).
      before %~ model ==> result %~ prefix 32 (seq_value model))
{
  tree_initial before;
  let mut width = 32sz;
  while (1sz <^ !width)
    invariant exists* (w : sz{tree_width (SZ.v w)}) (values : lseq f32 32).
      width |-> w ** partial |-> values **
      pure (tree_invariant before values (SZ.v w))
    decreases (SZ.v (!width))
  {
    let w = !width;
    let distance = w /^ 2sz;
    reduce_level partial distance #before;
    width := distance;
  };
  with values. assert partial |-> values;
  tree_finished before values;
  partial.(0sz)
}

inline_for_extraction noextract
let square_cell (hidden : lseq real hidden_width) (i : nat) : GTot real =
  if i < hidden_width then
    let x = S.index hidden i in x *. x
  else 0.0R

inline_for_extraction noextract
let square_pair (hidden : lseq real hidden_width) (warp lane : nat) : GTot real =
  square_cell hidden (warp * 32 + lane) +.
  square_cell hidden (512 + (warp * 32 + lane))

inline_for_extraction noextract
let warp_square_sum (hidden : lseq real hidden_width) (warp : nat) : GTot real =
  if warp < 16 then prefix 32 (square_pair hidden warp) else 0.0R

inline_for_extraction noextract
let rec squares_prefix
  (hidden : lseq real hidden_width)
  (count : nat{count <= hidden_width})
  : Lemma
      (ensures sum_squares hidden count == prefix count (square_cell hidden))
      (decreases count)
  = if count > 0 then begin
      squares_prefix hidden (count - 1);
      prefix_step (count - 1) (square_cell hidden)
    end

inline_for_extraction noextract
let warp_squares_split
  (hidden : lseq real hidden_width)
  (warp : nat{warp < 16})
  : Lemma (warp_square_sum hidden warp ==
      prefix 32 (fun lane -> square_cell hidden (warp * 32 + lane)) +.
      prefix 32 (fun lane -> square_cell hidden (512 + (warp * 32 + lane))))
  = prefix_ext 32 (square_pair hidden warp)
      (fun lane -> square_cell hidden (warp * 32 + lane) +.
        square_cell hidden (512 + (warp * 32 + lane)));
    prefix_add 32
      (fun lane -> square_cell hidden (warp * 32 + lane))
      (fun lane -> square_cell hidden (512 + (warp * 32 + lane)))

inline_for_extraction noextract
let all_squares
  (hidden : lseq real hidden_width)
  : Lemma (prefix 32 (warp_square_sum hidden) == sum_squares hidden hidden_width)
  = let first = (fun (warp : nat) -> prefix 32
      (fun lane -> square_cell hidden (warp * 32 + lane))) in
    let second = (fun (warp : nat) -> prefix 32
      (fun lane -> square_cell hidden (512 + (warp * 32 + lane)))) in
    C.forall_intro (warp_squares_split hidden);
    prefix_ext 16 (warp_square_sum hidden) (fun warp -> first warp +. second warp);
    prefix_add 16 first second;
    prefix_blocks 16 32 (square_cell hidden);
    prefix_blocks 16 32 (fun i -> square_cell hidden (512 + i));
    prefix_split 512 512 (square_cell hidden);
    squares_prefix hidden 1024;
    prefix_split 16 16 (warp_square_sum hidden);
    prefix_ext 16 (fun i -> warp_square_sum hidden (16 + i)) (fun _ -> 0.0R);
    prefix_zero 16

inline_for_extraction noextract
let lanes_squared
  (hidden : lseq bf16 hidden_width)
  (values : lseq f32 32)
  (warp : nat{warp < 16})
  (done : nat{done <= 32})
  : prop
  = forall (model : lseq real hidden_width).
      hidden %~ model ==>
      (forall (i : nat{i < done}).
        S.index values i %~ square_pair model warp i)

inline_for_extraction noextract
let warps_squared
  (hidden : lseq bf16 hidden_width)
  (values : lseq f32 32)
  (done : nat{done <= 16})
  : prop
  = forall (model : lseq real hidden_width).
      hidden %~ model ==>
      (forall (i : nat{i < 32}).
        if i < done then S.index values i %~ warp_square_sum model i
        else S.index values i == zero #f32)

inline_for_extraction noextract
let rms_finish
  (hidden : lseq bf16 hidden_width)
  (argument result : f32)
  : Lemma
      (requires
        (forall (model : lseq real hidden_width).
          hidden %~ model ==> argument %~
            (sum_squares model hidden_width /. 1024.0R +. 0.000001R <: real)) /\
        (forall (x : Sqrt.rpos).
          argument %~ (x <: real) ==>
          result %~ (1.0R /. Sqrt.sqrt x)))
      (ensures forall (model : lseq real hidden_width).
        hidden %~ model ==> result %~ (rms_scale model <: real))
  = let aux (model : lseq real hidden_width{hidden %~ model})
      : Lemma (result %~ (rms_scale model <: real))
      = ()
    in
    C.forall_intro aux

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
fn hidden_rstd
  (hidden : larray bf16 hidden_width)
  (#before : erased (lseq bf16 hidden_width))
  (#fraction : perm)
  preserves gpu ** hidden |-> Frac fraction before
  returns result : f32
  ensures pure (forall (model : lseq real hidden_width).
    before %~ model ==> result %~ (rms_scale model <: real))
{
  assert pure ((zero #f32) %~ 0.0R);
  let mut lanes = [| zero #f32 #_; 32sz |];
  let mut warps = [| zero #f32 #_; 32sz |];
  let mut warp = 0sz;
  while (!warp <^ 16sz)
    invariant exists* (w : sz{w <= 16}) (values : lseq f32 32).
      warp |-> w ** warps |-> values ** pure (warps_squared before values (SZ.v w))
    invariant exists* (values : lseq f32 32). lanes |-> values
    decreases (16 - !warp)
  {
    let w = !warp;
    let mut lane = 0sz;
    while (!lane <^ 32sz)
      invariant exists* (i : sz{i <= 32}) (values : lseq f32 32).
        lane |-> i ** lanes |-> values **
        pure (lanes_squared before values (SZ.v w) (SZ.v i))
      decreases (32 - !lane)
    {
      let i = !lane;
      let a : f32 = Cast.fcast (hidden.(w *^ 32sz +^ i));
      let b : f32 = Cast.fcast (hidden.(w *^ 32sz +^ i +^ 512sz));
      assert pure (forall (model : lseq real hidden_width).
        before %~ model ==>
        a %~ S.index model (SZ.v w * 32 + SZ.v i) /\
        b %~ S.index model (SZ.v w * 32 + SZ.v i + 512));
      let a_squared = F32.fma_rn_ftz a a (zero #f32);
      lift_fma
        (fun (model : lseq real hidden_width) -> before %~ model)
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i))
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i))
        (fun _ -> 0.0R)
        a a (zero #f32) a_squared;
      assert pure (forall (model : lseq real hidden_width).
        before %~ model ==>
        a_squared %~ square_cell model (SZ.v w * 32 + SZ.v i));
      let pair = F32.fma_rn_ftz b b a_squared;
      lift_fma
        (fun (model : lseq real hidden_width) -> before %~ model)
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i + 512))
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i + 512))
        (fun model -> square_cell model (SZ.v w * 32 + SZ.v i))
        b b a_squared pair;
      assert pure (forall (model : lseq real hidden_width).
        before %~ model ==> pair %~
          (S.index model (SZ.v w * 32 + SZ.v i + 512) *.
           S.index model (SZ.v w * 32 + SZ.v i + 512) +.
           square_cell model (SZ.v w * 32 + SZ.v i) <: real));
      assert pure (forall (model : lseq real hidden_width).
        before %~ model ==> pair %~ square_pair model (SZ.v w) (SZ.v i));
      lanes.(i) <- pair;
      lane := i +^ 1sz;
    };
    with lane_values. assert lanes |-> lane_values;
    let warp_total = sum32 lanes;
    lift_sum32
      (fun (model : lseq real hidden_width) -> before %~ model)
      (fun model -> square_pair model (SZ.v w))
      lane_values warp_total;
    warps.(w) <- warp_total;
    warp := w +^ 1sz;
  };
  with warp_values. assert warps |-> warp_values;
  let squared_total = sum32 warps;
  lift_sum32
    (fun (model : lseq real hidden_width) -> before %~ model)
    warp_square_sum warp_values squared_total;
  C.forall_intro all_squares;
  assert pure (forall (model : lseq real hidden_width).
    before %~ model ==> squared_total %~ (sum_squares model hidden_width <: real));
  let factor = inverse_width ();
  let epsilon = epsilon32 ();
  assert pure (factor %~ (1.0R /. 1024.0R));
  assert pure (epsilon %~ 0.000001R);
  let argument = F32.fma_rn_ftz squared_total factor epsilon;
  lift_fma
    (fun (model : lseq real hidden_width) -> before %~ model)
    (fun model -> (sum_squares model hidden_width <: real))
    (fun _ -> 1.0R /. 1024.0R)
    (fun _ -> 0.000001R)
    squared_total factor epsilon argument;
  let result = F32.rsqrt_approx_ftz argument;
  rms_finish before argument result;
  result
}
#pop-options

inline_for_extraction noextract
let interpretation = lseq real weight_elements & lseq real hidden_width

inline_for_extraction noextract
let norm_inputs
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (model : interpretation)
  : prop
  = weights %~ fst model /\ hidden %~ snd model

inline_for_extraction noextract
let activation_cell (model : interpretation) (i : nat) : GTot real =
  if i < hidden_width then normalized_cell (fst model) (snd model) i else 0.0R

inline_for_extraction noextract
let normalized_prefix
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (activation : lseq f32 hidden_width)
  (done : nat{done <= hidden_width})
  : prop
  = forall (model : interpretation).
      norm_inputs weights hidden model ==>
      (forall (i : nat{i < done}).
        S.index activation i %~ activation_cell model i)

inline_for_extraction noextract
fn normalize
  (weights : larray bf16 weight_elements)
  (hidden : larray bf16 hidden_width)
  (activation : larray f32 hidden_width)
  (#wv : erased (lseq bf16 weight_elements))
  (#hv : erased (lseq bf16 hidden_width))
  (#av : erased (lseq f32 hidden_width))
  (#fw #fh : perm)
  preserves gpu ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires activation |-> av
  ensures exists* (after : lseq f32 hidden_width).
    activation |-> after **
    pure (normalized_prefix wv hv after hidden_width)
{
  let scale = hidden_rstd hidden;
  assert pure ((one #f32) %~ 1.0R);
  let mut column = 0sz;
  while (!column <^ 1024sz)
    invariant exists* (i : sz{i <= 1024}) (values : lseq f32 hidden_width).
      column |-> i ** activation |-> values **
      pure (normalized_prefix wv hv values (SZ.v i))
    decreases (1024 - !column)
  {
    let i = !column;
    let h : f32 = Cast.fcast (hidden.(i));
    let w : f32 = Cast.fcast (weights.(SZ.uint32_to_sizet 254279680ul +^ i));
    assert pure (forall (model : interpretation). norm_inputs wv hv model ==>
      h %~ S.index (snd model) (SZ.v i) /\
      w %~ S.index (fst model) (embedding_elements + SZ.v i) /\
      scale %~ (rms_scale (snd model) <: real));
    let scaled = F32.mul_rn_ftz h scale;
    lift_mul #interpretation (norm_inputs wv hv)
      (fun model -> S.index (snd model) (SZ.v i))
      (fun model -> (rms_scale (snd model) <: real))
      h scale scaled;
    let adjusted = F32.add_rn_ftz (one #f32) w;
    lift_add #interpretation (norm_inputs wv hv) (fun _ -> 1.0R)
      (fun model -> S.index (fst model) (embedding_elements + SZ.v i))
      (one #f32) w adjusted;
    let value = F32.mul_rn_ftz scaled adjusted;
    lift_mul #interpretation (norm_inputs wv hv)
      (fun model -> S.index (snd model) (SZ.v i) *. rms_scale (snd model))
      (fun model -> 1.0R +. S.index (fst model) (embedding_elements + SZ.v i))
      scaled adjusted value;
    assert pure (forall (model : interpretation). norm_inputs wv hv model ==>
      value %~ activation_cell model (SZ.v i));
    activation.(i) <- value;
    column := i +^ 1sz;
  }
}

inline_for_extraction noextract
let dot_inputs
  (weights : lseq bf16 weight_elements)
  (activation : lseq f32 hidden_width)
  (model : interpretation)
  : prop
  = weights %~ fst model /\ activation %~ snd model

inline_for_extraction noextract
let dot_term
  (model : interpretation)
  (row : nat{row < vocabulary_size})
  (column : nat)
  : GTot real
  = if column < hidden_width then
      S.index (fst model) (row * hidden_width + column) *.
      S.index (snd model) column
    else 0.0R

inline_for_extraction noextract
let dot_block
  (model : interpretation)
  (row : nat{row < vocabulary_size})
  (lane group : nat)
  : GTot real
  = prefix 8 (fun j -> dot_term model row (group * 256 + (lane * 8 + j)))

inline_for_extraction noextract
let dot_lane
  (model : interpretation)
  (row : nat{row < vocabulary_size})
  (lane : nat)
  : GTot real
  = prefix 4 (dot_block model row lane)

inline_for_extraction noextract
let dot_partial
  (model : interpretation)
  (row : nat{row < vocabulary_size})
  (lane group count : nat)
  : GTot real
  = prefix group (dot_block model row lane) +.
      prefix count (fun j -> dot_term model row (group * 256 + (lane * 8 + j)))

inline_for_extraction noextract
let dot_partial_step
  (row : nat{row < vocabulary_size})
  (lane group count : nat)
  (model : interpretation)
  : Lemma (dot_partial model row lane group (count + 1) ==
      dot_partial model row lane group count +.
      dot_term model row (group * 256 + (lane * 8 + count)))
  = prefix_step count
      (fun j -> dot_term model row (group * 256 + (lane * 8 + j)))

inline_for_extraction noextract
let dot_group_end
  (row : nat{row < vocabulary_size})
  (lane group : nat)
  (model : interpretation)
  : Lemma (dot_partial model row lane group 8 ==
      prefix (group + 1) (dot_block model row lane))
  = prefix_step group (dot_block model row lane)

inline_for_extraction noextract
let dot_group_sum
  (model : interpretation)
  (row : nat{row < vocabulary_size})
  (group : nat)
  : Lemma (
      prefix 32 (fun lane -> dot_block model row lane group) ==
      prefix 256 (fun column -> dot_term model row (group * 256 + column)))
  = prefix_blocks 32 8 (fun column -> dot_term model row (group * 256 + column));
    prefix_ext 32 (fun lane -> dot_block model row lane group)
      (fun lane -> prefix 8
        (fun j -> dot_term model row (group * 256 + (lane * 8 + j))))

inline_for_extraction noextract
let dot_all_columns
  (row : nat{row < vocabulary_size})
  (model : interpretation)
  : Lemma (prefix 32 (dot_lane model row) == prefix 1024 (dot_term model row))
  = let expand_lane (lane : nat)
      : Lemma (dot_lane model row lane ==
          prefix 4 (fun group -> dot_block model row lane group))
      = prefix_ext 4 (dot_block model row lane)
          (fun group -> dot_block model row lane group)
    in
    C.forall_intro expand_lane;
    prefix_ext 32 (dot_lane model row)
      (fun lane -> prefix 4 (fun group -> dot_block model row lane group));
    prefix_swap 32 4 (fun lane group -> dot_block model row lane group);
    C.forall_intro (dot_group_sum model row);
    prefix_ext 4
      (fun group -> prefix 32 (fun lane -> dot_block model row lane group))
      (fun group -> prefix 256
        (fun column -> dot_term model row (group * 256 + column)));
    prefix_blocks 4 256 (dot_term model row)

inline_for_extraction noextract
fn dot_lane_run
  (weights : larray bf16 weight_elements)
  (activation : larray f32 hidden_width)
  (row : sz{row < vocabulary_size})
  (lane : sz{lane < 32})
  (#wv : erased (lseq bf16 weight_elements))
  (#av : erased (lseq f32 hidden_width))
  (#fw #fa : perm)
  preserves gpu ** weights |-> Frac fw wv ** activation |-> Frac fa av
  returns result : f32
  ensures pure (forall (model : interpretation). dot_inputs wv av model ==>
    result %~ dot_lane model (SZ.v row) (SZ.v lane))
{
  assert pure ((zero #f32) %~ 0.0R);
  let mut accumulator : f32 = zero;
  let mut group = 0sz;
  while (!group <^ 4sz)
    invariant exists* (g : sz{g <= 4}) (value : f32).
      group |-> g ** accumulator |-> value **
      pure (forall (model : interpretation). dot_inputs wv av model ==>
        value %~ prefix (SZ.v g) (dot_block model (SZ.v row) (SZ.v lane)))
    decreases (4 - !group)
  {
    let g = !group;
    let mut element = 0sz;
    while (!element <^ 8sz)
      invariant exists* (j : sz{j <= 8}) (value : f32).
        element |-> j ** accumulator |-> value **
        pure (forall (model : interpretation). dot_inputs wv av model ==>
          value %~ dot_partial model (SZ.v row) (SZ.v lane) (SZ.v g) (SZ.v j))
      decreases (8 - !element)
    {
      let j = !element;
      let column = g *^ 256sz +^ lane *^ 8sz +^ j;
      let w : f32 = Cast.fcast (weights.(row *^ 1024sz +^ column));
      let a = activation.(column);
      assert pure (forall (model : interpretation). dot_inputs wv av model ==>
        w %~ S.index (fst model) (SZ.v row * hidden_width + SZ.v column) /\
        a %~ S.index (snd model) (SZ.v column));
      let previous = !accumulator;
      let value = F32.fma_rn_ftz w a previous;
      lift_fma #interpretation (dot_inputs wv av)
        (fun model -> S.index (fst model) (SZ.v row * hidden_width + SZ.v column))
        (fun model -> S.index (snd model) (SZ.v column))
        (fun model -> dot_partial model (SZ.v row) (SZ.v lane) (SZ.v g) (SZ.v j))
        w a previous value;
      C.forall_intro (dot_partial_step (SZ.v row) (SZ.v lane) (SZ.v g) (SZ.v j));
      assert pure (forall (model : interpretation). dot_inputs wv av model ==>
        value %~ dot_partial model (SZ.v row) (SZ.v lane) (SZ.v g) (SZ.v j + 1));
      accumulator := value;
      element := j +^ 1sz;
    };
    C.forall_intro (dot_group_end (SZ.v row) (SZ.v lane) (SZ.v g));
    group := g +^ 1sz;
  };
  !accumulator
}

inline_for_extraction noextract
let dot_lanes_done
  (weights : lseq bf16 weight_elements)
  (activation : lseq f32 hidden_width)
  (partial : lseq f32 32)
  (row : nat{row < vocabulary_size})
  (done : nat{done <= 32})
  : prop
  = forall (model : interpretation). dot_inputs weights activation model ==>
      (forall (lane : nat{lane < done}).
        S.index partial lane %~ dot_lane model row lane)

inline_for_extraction noextract
fn dot_float
  (weights : larray bf16 weight_elements)
  (activation : larray f32 hidden_width)
  (row : sz{row < vocabulary_size})
  (#wv : erased (lseq bf16 weight_elements))
  (#av : erased (lseq f32 hidden_width))
  (#fw #fa : perm)
  preserves gpu ** weights |-> Frac fw wv ** activation |-> Frac fa av
  returns result : f32
  ensures pure (forall (model : interpretation). dot_inputs wv av model ==>
    result %~ prefix 1024 (dot_term model (SZ.v row)))
{
  let mut partial = [| zero #f32 #_; 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i : sz{i <= 32}) (values : lseq f32 32).
      lane |-> i ** partial |-> values **
      pure (dot_lanes_done wv av values (SZ.v row) (SZ.v i))
    decreases (32 - !lane)
  {
    let i = !lane;
    let value = dot_lane_run weights activation row i;
    partial.(i) <- value;
    lane := i +^ 1sz;
  };
  with values. assert partial |-> values;
  let result = sum32 partial;
  lift_sum32 #interpretation (dot_inputs wv av)
    (fun model -> dot_lane model (SZ.v row)) values result;
  C.forall_intro (dot_all_columns (SZ.v row));
  result
}

inline_for_extraction noextract
let normalized_interpretation (model : interpretation) : GTot interpretation =
  fst model, S.init_ghost hidden_width (activation_cell model)

inline_for_extraction noextract
let projection_dot_model
  (row : nat{row < vocabulary_size})
  (model : interpretation)
  : Lemma (
      prefix 1024 (dot_term (normalized_interpretation model) row) ==
      projected_cell (fst model) (snd model) row)
  = let actual =
      S.init_ghost 1024 (dot_term (normalized_interpretation model) row) in
    let expected = S.init_ghost hidden_width (fun column ->
        S.index (fst model) (row * hidden_width + column) *.
          normalized_cell (fst model) (snd model) column) in
    prefix_init 1024 (dot_term (normalized_interpretation model) row);
    assert (S.equal actual expected);
    S.lemma_eq_elim actual expected;
    assert (projected_cell (fst model) (snd model) row == rsum expected)
      by (FStar.Tactics.V2.norm [delta_only ["KPlay.Functional.projected_cell"]])

inline_for_extraction noextract
let projection_from_dot
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (activation : lseq f32 hidden_width)
  (row : nat{row < vocabulary_size})
  (score : f32)
  : Lemma
      (requires normalized_prefix weights hidden activation hidden_width /\
        (forall (model : interpretation). dot_inputs weights activation model ==>
          score %~ prefix 1024 (dot_term model row)))
      (ensures forall (model : interpretation). norm_inputs weights hidden model ==>
        score %~ projected_cell (fst model) (snd model) row)
  = let aux (model : interpretation{norm_inputs weights hidden model})
      : Lemma (score %~ projected_cell (fst model) (snd model) row)
      = let dm = normalized_interpretation model in
        assert (dot_inputs weights activation dm);
        projection_dot_model row model
    in
    C.forall_intro aux

inline_for_extraction noextract
let visited_row (row block warp next : nat) : prop =
  row < block * 485 \/
  (row < (block + 1) * 485 /\
    ((row - block * 485) % 8 < warp \/
     ((row - block * 485) % 8 = warp /\ row < next)))

inline_for_extraction noextract
let projection_progress
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (scores : lseq f32 vocabulary_size)
  (block warp next : nat)
  : prop
  = forall (model : interpretation). norm_inputs weights hidden model ==>
      (forall (row : nat{row < vocabulary_size}).
        visited_row row block warp next ==>
        S.index scores row %~ projected_cell (fst model) (snd model) row)

inline_for_extraction noextract
let projection_write
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (scores : lseq f32 vocabulary_size)
  (block : nat{block < 512})
  (warp : nat{warp < 8})
  (row : nat{block * 485 <= row /\ row < (block + 1) * 485 /\
    (row - block * 485) % 8 = warp})
  (score : f32)
  : Lemma
      (requires projection_progress weights hidden scores block warp row /\
        (forall (model : interpretation). norm_inputs weights hidden model ==>
          score %~ projected_cell (fst model) (snd model) row))
      (ensures projection_progress weights hidden (S.upd scores row score)
        block warp (row + 8))
  = ()

inline_for_extraction noextract
fn project_warp
  (weights : larray bf16 weight_elements)
  (activation : larray f32 hidden_width)
  (scores : larray f32 vocabulary_size)
  (block : sz{block < 512})
  (warp : sz{warp < 32})
  (#wv : erased (lseq bf16 weight_elements))
  (#hv : erased (lseq bf16 hidden_width))
  (#av : erased (lseq f32 hidden_width))
  (#sv : erased (lseq f32 vocabulary_size))
  (#fw #fa : perm)
  preserves gpu ** weights |-> Frac fw wv ** activation |-> Frac fa av
  requires scores |-> sv **
    pure (normalized_prefix wv hv av hidden_width /\
      projection_progress wv hv sv (SZ.v block) (SZ.v warp) 0)
  ensures exists* (after : lseq f32 vocabulary_size).
    scores |-> after **
    pure (projection_progress wv hv after (SZ.v block) (SZ.v warp + 1) 0)
{
  if (warp <^ 8sz) {
    let end_row = (block +^ 1sz) *^ 485sz;
    let mut row = block *^ 485sz +^ warp;
    while (!row <^ end_row)
      invariant exists*
        (r : sz{block * 485 <= r /\ r <= (block + 1) * 485 + 7 /\
          (r - block * 485) % 8 = SZ.v warp})
        (values : lseq f32 vocabulary_size).
        row |-> r ** scores |-> values **
        pure (projection_progress wv hv values (SZ.v block) (SZ.v warp) (SZ.v r))
      decreases (SZ.v end_row + 8 - !row)
    {
      let r = !row;
      with values. assert scores |-> values;
      let score = dot_float weights activation r;
      projection_from_dot wv hv av (SZ.v r) score;
      projection_write wv hv values (SZ.v block) (SZ.v warp) (SZ.v r) score;
      scores.(r) <- score;
      row := r +^ 8sz;
    }
  }
}

inline_for_extraction noextract
let projection_finished
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (scores : lseq f32 vocabulary_size)
  : Lemma
      (requires projection_progress weights hidden scores 512 0 0)
      (ensures projection_post weights hidden scores)
  = let aux
      (rw : lseq real weight_elements{weights %~ rw})
      (rh : lseq real hidden_width{hidden %~ rh})
      : Lemma (forall (row : nat{row < vocabulary_size}).
          S.index scores row %~ projected_cell rw rh row)
      = assert (norm_inputs weights hidden (rw, rh));
        assert (forall (row : nat{row < vocabulary_size}).
          visited_row row 512 0 0)
    in
    C.forall_intro_2 aux;
    assert (projection_post weights hidden scores)
      by (FStar.Tactics.V2.norm [delta_only ["KPlay.Functional.projection_post"]])

inline_for_extraction noextract
fn project_scores
  (weights : larray bf16 weight_elements)
  (hidden : larray bf16 hidden_width)
  (scores : larray f32 vocabulary_size)
  (#wv : erased (lseq bf16 weight_elements))
  (#hv : erased (lseq bf16 hidden_width))
  (#sv : erased (lseq f32 vocabulary_size))
  (#fw #fh : perm)
  preserves gpu ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires scores |-> sv
  ensures exists* (after : lseq f32 vocabulary_size).
    scores |-> after ** pure (projection_post wv hv after)
{
  let mut activation = [| zero #f32 #_; 1024sz |];
  normalize weights hidden activation;
  with av. assert activation |-> av;
  let mut block = 0sz;
  while (!block <^ 512sz)
    invariant exists* (b : sz{b <= 512}) (values : lseq f32 vocabulary_size).
      block |-> b ** scores |-> values **
      pure (projection_progress wv hv values (SZ.v b) 0 0)
    decreases (512 - !block)
  {
    let b = !block;
    let mut warp = 0sz;
    while (!warp <^ 32sz)
      invariant exists* (w : sz{w <= 32}) (values : lseq f32 vocabulary_size).
        warp |-> w ** scores |-> values **
        pure (projection_progress wv hv values (SZ.v b) (SZ.v w) 0)
      decreases (32 - !warp)
    {
      let w = !warp;
      project_warp weights activation scores b w #wv #hv #av;
      warp := w +^ 1sz;
    };
    block := b +^ 1sz;
  };
  with values. assert scores |-> values;
  projection_finished wv hv values;
  ()
}

inline_for_extraction noextract
fn write_history
  (hidden : larray bf16 hidden_width)
  (previous_buffer history : array bf16)
  (position : i64)
  (prompt_length : i64{valid_position (I64.v position) (I64.v prompt_length)})
  (#hv : erased (lseq bf16 hidden_width))
  (#pv : erased (lseq bf16
    (previous_rows (I64.v position) (I64.v prompt_length) * hidden_width)))
  (#old : erased (lseq bf16
    (result_rows (I64.v position) (I64.v prompt_length) * hidden_width)))
  (#fh #fp : perm)
  preserves gpu ** hidden |-> Frac fh hv **
    history_ownership previous_buffer
      (previous_rows (I64.v position) (I64.v prompt_length)) fp pv
  requires history_ownership history
    (result_rows (I64.v position) (I64.v prompt_length)) 1.0R old
  ensures history_ownership history
    (result_rows (I64.v position) (I64.v prompt_length)) 1.0R
    (history_model (I64.v position) (I64.v prompt_length) pv hv)
{
  assert_norm (pow2 32 == 0x100000000);
  let p = SZ.uint32_to_sizet (IC.int64_to_uint32 position);
  let n = SZ.uint32_to_sizet (IC.int64_to_uint32 prompt_length);
  assert pure (SZ.v p == I64.v position /\ SZ.v n == I64.v prompt_length);
  if (p <^ n -^ 1sz) {
    assert pure (result_rows (I64.v position) (I64.v prompt_length) == 0);
    rewrite history_ownership history
      (result_rows (I64.v position) (I64.v prompt_length)) 1.0R old as emp;
    rewrite emp as history_ownership history
      (result_rows (I64.v position) (I64.v prompt_length)) 1.0R
      (history_model (I64.v position) (I64.v prompt_length) pv hv);
  } else {
    assert pure (result_rows (I64.v position) (I64.v prompt_length) ==
      previous_rows (I64.v position) (I64.v prompt_length) + 1);
    rewrite history_ownership history
      (result_rows (I64.v position) (I64.v prompt_length)) 1.0R old as
      (pure (is_global_array history) ** history |-> old);
    pts_to_len history;
    let count = if (p <^ n) { 0sz } else { p -^ n +^ 1sz };
    let offset = count *^ 1024sz;
    assert pure (SZ.v count == previous_rows (I64.v position) (I64.v prompt_length));
    let mut index = 0sz;
    while (!index <^ offset)
      invariant exists* (i : sz{i <= offset})
        (values : lseq bf16
          (result_rows (I64.v position) (I64.v prompt_length) * hidden_width)).
        index |-> i ** history |-> values **
        pure (forall (j : nat{j < SZ.v i}). S.index values j == S.index pv j)
      decreases (offset - !index)
    {
      let i = !index;
      rewrite history_ownership previous_buffer
        (previous_rows (I64.v position) (I64.v prompt_length)) fp pv as
        (pure (is_global_array previous_buffer) ** previous_buffer |-> Frac fp pv);
      pts_to_len previous_buffer;
      history.(i) <- previous_buffer.(i);
      rewrite (pure (is_global_array previous_buffer) ** previous_buffer |-> Frac fp pv) as
        history_ownership previous_buffer
          (previous_rows (I64.v position) (I64.v prompt_length)) fp pv;
      index := i +^ 1sz;
    };
    with prefix_values. assert history |-> prefix_values;
    assert pure (forall (j : nat{j < SZ.v offset}).
      S.index prefix_values j == S.index pv j);
    let mut column = 0sz;
    while (!column <^ 1024sz)
      invariant exists* (i : sz{i <= 1024})
        (values : lseq bf16
          (result_rows (I64.v position) (I64.v prompt_length) * hidden_width)).
        column |-> i ** history |-> values **
        pure ((forall (j : nat{j < SZ.v offset}).
          S.index values j == S.index pv j) /\
          (forall (j : nat{j < SZ.v i}).
            S.index values (SZ.v offset + j) == S.index hv j))
      decreases (1024 - !column)
    {
      let i = !column;
      history.(offset +^ i) <- hidden.(i);
      column := i +^ 1sz;
    };
    with values. assert history |-> values;
    assert pure (S.equal values (S.append pv hv));
    S.lemma_eq_elim values (S.append pv hv);
    rewrite (pure (is_global_array history) ** history |-> values) as
      history_ownership history
        (result_rows (I64.v position) (I64.v prompt_length)) 1.0R
        (history_model (I64.v position) (I64.v prompt_length) pv hv);
  }
}

#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"

inline_for_extraction noextract
let real_values (#n : nat) (values : lseq bf16 n)
  : GTot (lseq real n) =
  S.init_ghost n (fun i -> Approx.to_real (S.index values i))

inline_for_extraction noextract
let real_values_approx (#n : nat) (values : lseq bf16 n)
  : Lemma (values %~ real_values values) =
  let aux (i : nat{i < n})
    : Lemma (S.index values i %~ S.index (real_values values) i) =
    Approx.to_real_ok (S.index values i)
  in
  C.forall_intro aux

inline_for_extraction noextract
let real_values_index
  (#n : nat) (values : lseq bf16 n) (i : nat{i < n})
  : Lemma (
      S.index (real_values values) i == Approx.to_real (S.index values i)) =
  ()

inline_for_extraction noextract
let rec prefix_shared_sum
  (#n : nat)
  (f : nat -> GTot real)
  (g : Shared.vector real n)
  (count : nat{count <= n})
  : Lemma
      (requires forall (i : Shared.index n). f i == g i)
      (ensures prefix count f ==
        Sum.sum #real #Sum.real_add_semigroup 0 count g)
      (decreases count) =
  if count > 0 then begin
    prefix_shared_sum f g (count - 1);
    prefix_step (count - 1) f;
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 count g
  end

inline_for_extraction noextract
let shared_rms_expand (x : Shared.vector real hidden_width)
  : Lemma (
      Shared.rms_argument epsilon x ==
        Shared.real_sum (fun i -> x i *. x i) /.
          R.of_int hidden_width +. epsilon) =
  assert (
    Shared.rms_argument epsilon x ==
      Shared.real_sum (fun i -> x i *. x i) /.
        R.of_int hidden_width +. epsilon) by (
    FStar.Tactics.V2.norm [
      zeta; iota; delta_only ["KPlay.SharedSpec.rms_argument"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let rms_refinement (hidden : lseq bf16 hidden_width)
  : Lemma (
      rms_scale (real_values hidden) ==
        Sqrt.rsqrt (Shared.rms_argument epsilon
          (Shared.real_vector (vector_view hidden)))) =
  let x = Shared.real_vector (vector_view hidden) in
  squares_prefix (real_values hidden) hidden_width;
  prefix_shared_sum
    (square_cell (real_values hidden)) (fun i -> x i *. x i) hidden_width;
  shared_rms_expand x

inline_for_extraction noextract
let normalized_refinement
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (i : Shared.index hidden_width)
  : Lemma (
      normalized_cell (real_values weights) (real_values hidden) i ==
        Shared.normalized_for_logits epsilon
          (Shared.real_vector (vector_view hidden))
          (Shared.real_vector (norm_weight_view weights)) i) =
  rms_refinement hidden

inline_for_extraction noextract
let model_projection_term
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (row : Shared.index vocabulary_size)
  (i : nat)
  : GTot real =
  if i < hidden_width then
    S.index (real_values weights) (row * hidden_width + i) *.
      normalized_cell (real_values weights) (real_values hidden) i
  else 0.0R

inline_for_extraction noextract
let shared_projection_term
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (row : Shared.index vocabulary_size)
  (i : Shared.index hidden_width)
  : GTot real =
  Shared.normalized_for_logits Shared.reference_epsilon
    (Shared.real_vector (vector_view hidden))
    (Shared.real_vector (norm_weight_view weights)) i *.
    Approx.to_real (embedding_view weights row i)

inline_for_extraction noextract
let projection_term_refinement
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (row : Shared.index vocabulary_size)
  (i : Shared.index hidden_width)
  : Lemma (
      model_projection_term weights hidden row i ==
        shared_projection_term weights hidden row i) =
  real_values_index weights (row * hidden_width + i);
  normalized_refinement weights hidden i;
  assert (
    Approx.to_real (embedding_view weights row i) ==
      S.index (real_values weights) (row * hidden_width + i));
  assert (
    model_projection_term weights hidden row i ==
      S.index (real_values weights) (row * hidden_width + i) *.
        normalized_cell (real_values weights) (real_values hidden) i);
  assert (
    shared_projection_term weights hidden row i ==
      normalized_cell (real_values weights) (real_values hidden) i *.
        S.index (real_values weights) (row * hidden_width + i))

inline_for_extraction noextract
let shared_logits_expand
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (row : Shared.index vocabulary_size)
  : Lemma (
      Shared.logical_logits Shared.reference_epsilon
        (vector_view hidden) (norm_weight_view weights) (embedding_view weights) row ==
        Shared.real_sum (shared_projection_term weights hidden row)) =
  assert (
    Shared.logical_logits Shared.reference_epsilon
      (vector_view hidden) (norm_weight_view weights) (embedding_view weights) row ==
      Shared.real_sum (shared_projection_term weights hidden row)) by (
    FStar.Tactics.V2.norm [
      zeta; iota; delta_only [
        "KPlay.Functional.shared_projection_term";
        "KPlay.SharedSpec.logical_logits";
        "KPlay.SharedSpec.project";
        "KPlay.SharedSpec.real_matrix";
        "KPlay.SharedSpec.real_vector"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let model_projection_expand
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (row : Shared.index vocabulary_size)
  : Lemma (
      projected_cell (real_values weights) (real_values hidden) row ==
        prefix hidden_width (model_projection_term weights hidden row)) =
  prefix_init hidden_width (model_projection_term weights hidden row);
  let actual = S.init_ghost hidden_width (model_projection_term weights hidden row) in
  let expected = S.init_ghost hidden_width (fun i ->
    S.index (real_values weights) (row * hidden_width + i) *.
      normalized_cell (real_values weights) (real_values hidden) i) in
  assert (S.equal actual expected);
  S.lemma_eq_elim actual expected;
  assert (
    projected_cell (real_values weights) (real_values hidden) row == rsum expected) by (
    FStar.Tactics.V2.norm [delta_only ["KPlay.Functional.projected_cell"]])

inline_for_extraction noextract
let projected_refinement
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (row : Shared.index vocabulary_size)
  : Lemma (
      projected_cell (real_values weights) (real_values hidden) row ==
        Shared.logical_logits Shared.reference_epsilon
          (vector_view hidden) (norm_weight_view weights)
          (embedding_view weights) row) =
  model_projection_expand weights hidden row;
  C.forall_intro (projection_term_refinement weights hidden row);
  prefix_shared_sum
    (model_projection_term weights hidden row)
    (shared_projection_term weights hidden row) hidden_width;
  shared_logits_expand weights hidden row

inline_for_extraction noextract
let projection_refines_shared
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (scores : lseq f32 vocabulary_size)
  : Lemma
      (requires projection_post weights hidden scores)
      (ensures forall (token : Shared.index vocabulary_size).
        S.index scores token %~
          Shared.logical_logits Shared.reference_epsilon
            (vector_view hidden) (norm_weight_view weights)
            (embedding_view weights) token) =
  real_values_approx weights;
  real_values_approx hidden;
  let aux (token : Shared.index vocabulary_size)
    : Lemma (
        S.index scores token %~
          Shared.logical_logits Shared.reference_epsilon
            (vector_view hidden) (norm_weight_view weights)
            (embedding_view weights) token) =
    projected_refinement weights hidden token
  in
  C.forall_intro aux

#pop-options

inline_for_extraction noextract
instance send_global_array_contents
  (#a : Type0)
  (buffer : array a{is_global_array buffer})
  (#fraction : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to buffer #fraction contents)
  = Kuiper.Array.Core.is_send_pts_to buffer #fraction contents

ghost
fn move_history
  (buffer : array bf16)
  (rows : nat)
  (fraction : perm)
  (contents : lseq bf16 (rows * hidden_width))
  (source : loc_id)
  (destination : loc_id{gpu_of source == gpu_of destination})
  requires on source (history_ownership buffer rows fraction contents)
  ensures on destination (history_ownership buffer rows fraction contents)
{
  if (rows = 0) {
    map_loc source #(history_ownership buffer rows fraction contents) #emp
      fn () { rewrite history_ownership buffer rows fraction contents as emp };
    map_loc destination #emp #(history_ownership buffer rows fraction contents)
      fn () { rewrite emp as history_ownership buffer rows fraction contents };
  } else {
    map_loc source
      #(history_ownership buffer rows fraction contents)
      #(pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      fn () {
        rewrite history_ownership buffer rows fraction contents as
          (pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      };
    is_send_across_elim gpu_of (buffer |-> Frac fraction contents) destination;
    map_loc destination
      #(pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      #(history_ownership buffer rows fraction contents)
      fn () {
        rewrite (pure (is_global_array buffer) ** buffer |-> Frac fraction contents) as
          history_ownership buffer rows fraction contents
      };
  }
}

inline_for_extraction noextract
instance send_history
  (buffer : array bf16)
  (rows : nat)
  (fraction : perm)
  (contents : lseq bf16 (rows * hidden_width))
  : is_send_across gpu_of (history_ownership buffer rows fraction contents)
  = fun source destination -> move_history buffer rows fraction contents source destination

inline_for_extraction noextract
fn family_kernel
  (input0 : larray bf16 weight_elements{is_global_array input0})
  (input1 : larray bf16 hidden_width{is_global_array input1})
  (input2 : array bf16)
  (input3 : i64)
  (input4 : i64{valid_position (I64.v input3) (I64.v input4)})
  (output0 : larray f32 vocabulary_size{is_global_array output0})
  (output1 : array bf16)
  (#weights : erased (lseq bf16 weight_elements))
  (#hidden : erased (lseq bf16 hidden_width))
  (#previous : erased
    (lseq bf16 (previous_rows (I64.v input3) (I64.v input4) * hidden_width)))
  (#old_scores : erased (lseq f32 vocabulary_size))
  (#old_history : erased
    (lseq bf16 (result_rows (I64.v input3) (I64.v input4) * hidden_width)))
  (#fweights #fhidden #fprevious : perm)
  preserves gpu **
    input0 |-> Frac fweights weights **
    input1 |-> Frac fhidden hidden **
    history_ownership input2
      (previous_rows (I64.v input3) (I64.v input4)) fprevious previous
  requires output0 |-> old_scores **
    history_ownership output1
      (result_rows (I64.v input3) (I64.v input4)) 1.0R old_history
  ensures exists* (scores : lseq f32 vocabulary_size)
                  (history : lseq bf16
                    (result_rows (I64.v input3) (I64.v input4) * hidden_width)).
    output0 |-> scores **
    history_ownership output1
      (result_rows (I64.v input3) (I64.v input4)) 1.0R history **
    pure (final_norm_projection_post
      weights hidden (I64.v input3) (I64.v input4) previous scores history)
{
  write_history input1 input2 output1 input3 input4;
  project_scores input0 input1 output0;
  with scores. assert output0 |-> scores;
  let history : erased (lseq bf16
    (result_rows (I64.v input3) (I64.v input4) * hidden_width)) =
    history_model (I64.v input3) (I64.v input4) previous hidden;
  assert pure (final_norm_projection_post
    weights hidden (I64.v input3) (I64.v input4) previous scores history);
}

fn kplay_run
  (#state : Type0)
  (#state_ownership : state -> slprop)
  (#before : erased state)
  (input0 : larray bf16 weight_elements{is_global_array input0})
  (input1 : larray bf16 hidden_width{is_global_array input1})
  (input2 : array bf16)
  (input3 : i64)
  (input4 : i64{valid_position (I64.v input3) (I64.v input4)})
  (output0 : larray f32 vocabulary_size{is_global_array output0})
  (output1 : array bf16)
  (#weights : erased (lseq bf16 weight_elements))
  (#hidden : erased (lseq bf16 hidden_width))
  (#previous : erased
    (lseq bf16 (previous_rows (I64.v input3) (I64.v input4) * hidden_width)))
  (#old_scores : erased (lseq f32 vocabulary_size))
  (#old_history : erased
    (lseq bf16 (result_rows (I64.v input3) (I64.v input4) * hidden_width)))
  (#fweights #fhidden #fprevious : perm)
  preserves cpu
  preserves on gpu_loc (
    input0 |-> Frac fweights weights **
    input1 |-> Frac fhidden hidden **
    history_ownership input2
      (previous_rows (I64.v input3) (I64.v input4)) fprevious previous)
  requires state_ownership before ** on gpu_loc (
    output0 |-> old_scores **
    history_ownership output1
      (result_rows (I64.v input3) (I64.v input4)) 1.0R old_history)
  ensures exists* (scores : lseq f32 vocabulary_size)
                  (history : lseq bf16
                    (result_rows (I64.v input3) (I64.v input4) * hidden_width))
                  (after : state).
    on gpu_loc (
      output0 |-> scores **
      history_ownership output1
        (result_rows (I64.v input3) (I64.v input4)) 1.0R history) **
    state_ownership after **
    pure (semantic_postcondition
      weights hidden input3 input4 previous before after scores history)
{
  launch_kernel_1 (fun () ->
    family_kernel input0 input1 input2 input3 input4 output0 output1
      #weights #hidden #previous #old_scores #old_history #fweights #fhidden #fprevious);
  with scores. assert on gpu_loc (output0 |-> scores);
  with history. assert on gpu_loc (
    history_ownership output1
      (result_rows (I64.v input3) (I64.v input4)) 1.0R history);
  projection_refines_shared weights hidden scores;
  establish_semantic_postcondition
    weights hidden input3 input4 previous before before scores history;
}
