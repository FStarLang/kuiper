module KPlay.Functional

#lang-pulse

#set-options "--fuel 4 --ifuel 2 --z3rlimit 40"

open Kuiper
open FStar.Ghost

module Shared = KPlay.SharedSpec
module Seq = FStar.Seq
module Sum = Kuiper.Sum
module Cast = Kuiper.Float.Casts.Base
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let rsum (n:nat) (f:nat -> GTot real) : GTot real =
  Sum.sum #real #Sum.real_add_semigroup 0 n f

inline_for_extraction noextract
let sum_op (x y:real)
  : Lemma (Sum.op #real #Sum.real_add_semigroup x y == x +. y) =
  assert_norm (Sum.op #real #Sum.real_add_semigroup x y == x +. y)

inline_for_extraction noextract
let rsum_succ (f:nat -> GTot real) (k:nat)
  : Lemma (rsum (k+1) f == rsum k f +. f k) =
  Sum.sum_pop_right #real #Sum.real_add_semigroup 0 (k+1) f

inline_for_extraction noextract
let rec rsum_ext (n:nat) (f g:nat -> GTot real)
  : Lemma
      (requires forall (i:natlt n). f i == g i)
      (ensures rsum n f == rsum n g)
      (decreases n) =
  if n > 0 then (
    rsum_ext (n-1) f g;
    rsum_succ f (n-1);
    rsum_succ g (n-1)
  )

inline_for_extraction noextract
let rec rsum_add (n:nat) (f g:nat -> GTot real)
  : Lemma
      (rsum n (fun i -> f i +. g i) == rsum n f +. rsum n g)
      (decreases n) =
  if n > 0 then (
    rsum_add (n-1) f g;
    rsum_succ (fun i -> f i +. g i) (n-1);
    rsum_succ f (n-1);
    rsum_succ g (n-1)
  )

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
inline_for_extraction noextract
let rec rsum_shift (lo count:nat) (f:nat -> GTot real)
  : Lemma
      (Sum.sum #real #Sum.real_add_semigroup lo (lo+count) f ==
       rsum count (fun i -> f (lo+i)))
      (decreases count) =
  if count > 0 then (
    rsum_shift lo (count-1) f;
    assert (Sum.sum #real #Sum.real_add_semigroup lo (lo+count-1) f ==
      rsum (count-1) (fun i -> f (lo+i)));
    Sum.sum_pop_right #real #Sum.real_add_semigroup lo (lo+count) f;
    sum_op (Sum.sum #real #Sum.real_add_semigroup lo (lo+count-1) f)
      (f (lo+count-1));
    assert (Sum.sum #real #Sum.real_add_semigroup lo (lo+count) f ==
      rsum (count-1) (fun i -> f (lo+i)) +. f (lo+count-1));
    rsum_succ (fun i -> f (lo+i)) (count-1);
    assert (rsum count (fun i -> f (lo+i)) ==
      rsum (count-1) (fun i -> f (lo+i)) +. f (lo+count-1))
  ) else (
    assert_norm (Sum.sum #real #Sum.real_add_semigroup lo lo f == 0.0R);
    assert_norm (rsum 0 (fun i -> f (lo+i)) == 0.0R)
  )

inline_for_extraction noextract
let rec rsum_even_odd (n:nat) (f:nat -> GTot real)
  : Lemma
      (rsum n (fun i -> f (2*i)) +.
       rsum n (fun i -> f (2*i+1)) == rsum (2*n) f)
      (decreases n) =
  if n > 0 then (
    rsum_even_odd (n-1) f;
    rsum_succ (fun i -> f (2*i)) (n-1);
    rsum_succ (fun i -> f (2*i+1)) (n-1);
    rsum_succ f (2*n-1);
    rsum_succ f (2*n-2)
  ) else (
    assert_norm (rsum 0 f == 0.0R);
    assert_norm (rsum 0 (fun i -> f (2*i)) == 0.0R);
    assert_norm (rsum 0 (fun i -> f (2*i+1)) == 0.0R)
  )

inline_for_extraction noextract
let rec rsum_quads (n:nat) (f:nat -> GTot real)
  : Lemma
      (rsum n (fun i ->
        ((f (4*i) +. f (4*i+1)) +. f (4*i+2)) +. f (4*i+3)) ==
       rsum (4*n) f)
      (decreases n) =
  if n > 0 then (
    rsum_quads (n-1) f;
    rsum_succ (fun i ->
      ((f (4*i) +. f (4*i+1)) +. f (4*i+2)) +. f (4*i+3)) (n-1);
    rsum_succ f (4*n-1);
    rsum_succ f (4*n-2);
    rsum_succ f (4*n-3);
    rsum_succ f (4*n-4)
  ) else (
    assert_norm (rsum 0 f == 0.0R);
    assert_norm (rsum 0 (fun i ->
      ((f (4*i) +. f (4*i+1)) +. f (4*i+2)) +. f (4*i+3)) == 0.0R)
  )

inline_for_extraction noextract
let rsum_strided (n lane:nat) (stride:pos) (f:nat -> GTot real)
  : Lemma (
      rsum (2*n) (fun i -> f (lane+i*stride)) ==
      rsum n (fun i -> f (lane+i*(2*stride))) +.
      rsum n (fun i -> f (lane+stride+i*(2*stride)))) =
  let g : nat -> GTot real = fun i -> f (lane+i*stride) in
  rsum_even_odd n g;
  rsum_ext n (fun i -> g (2*i)) (fun i -> f (lane+i*(2*stride)));
  rsum_ext n (fun i -> g (2*i+1)) (fun i -> f (lane+stride+i*(2*stride)))
#pop-options

inline_for_extraction noextract
let square (row:Shared.hidden_row) (d:nat) : GTot real =
  if d < Shared.hidden_size then row d *. row d else 0.0R

inline_for_extraction noextract
let lane_sum (row:Shared.hidden_row) (lane:nat) : GTot real =
  let f = square row in
  (((f (4*lane) +. f (4*lane+512)) +.
    (f (4*lane+1) +. f (4*lane+513))) +.
    (f (4*lane+2) +. f (4*lane+514))) +.
    (f (4*lane+3) +. f (4*lane+515))

#push-options "--fuel 1 --ifuel 0 --z3rlimit 100"
inline_for_extraction noextract
let lanes_sum_to_hidden (row:Shared.hidden_row)
  : Lemma
      (rsum 128 (lane_sum row) == Shared.sum_hidden (square row)) =
  let f = square row in
  let left : nat -> GTot real = fun i ->
    ((f (4*i) +. f (4*i+1)) +. f (4*i+2)) +. f (4*i+3) in
  let right : nat -> GTot real = fun i ->
    ((f (512+4*i) +. f (513+4*i)) +. f (514+4*i)) +. f (515+4*i) in
  rsum_ext 128 (lane_sum row) (fun i -> left i +. right i);
  rsum_add 128 left right;
  rsum_quads 128 f;
  rsum_quads 128 (fun i -> f (512+i));
  rsum_ext 128 right (fun i ->
    ((f (512+4*i) +. f (512+(4*i+1))) +. f (512+(4*i+2))) +.
      f (512+(4*i+3)));
  assert (rsum 128 left == rsum 512 f);
  assert (rsum 128 right == rsum 512 (fun i -> f (512+i)));
  rsum_shift 512 512 f;
  Sum.sum_split #real #Sum.real_add_semigroup 0 1024 f 512;
  sum_op (rsum 512 f)
    (Sum.sum #real #Sum.real_add_semigroup 512 1024 f);
  assert (Shared.sum_hidden (square row) ==
    Sum.sum #real #Sum.real_add_semigroup 0 1024 f)
#pop-options

inline_for_extraction noextract
let valid_width (n:nat) : prop =
  n == 128 \/ n == 64 \/ n == 32 \/ n == 16 \/
  n == 8 \/ n == 4 \/ n == 2 \/ n == 1

inline_for_extraction noextract
type reduction_width = n:pos{valid_width n}

inline_for_extraction noextract
let width_halves (width:reduction_width{width>1})
  : Lemma (valid_width (width/2) /\ 0 < width/2 /\ width/2 < width) = ()

inline_for_extraction noextract
let grouped_sum
  (row:Shared.hidden_row)
  (width:reduction_width)
  (lane:natlt width) : GTot real =
  rsum (128/width) (fun i -> lane_sum row (lane+i*width))

inline_for_extraction noextract
let grouped_def (row:Shared.hidden_row)
  (width:reduction_width) (lane:natlt width)
  : Lemma (grouped_sum row width lane ==
    rsum (128/width) (fun i -> lane_sum row (lane+i*width))) = ()

inline_for_extraction noextract
let grouped_initial (row:Shared.hidden_row) (lane:natlt 128)
  : Lemma (grouped_sum row 128 lane == lane_sum row lane) =
  rsum_succ (fun i -> lane_sum row (lane+i*128)) 0

#push-options "--fuel 1 --ifuel 0 --z3rlimit 100"
inline_for_extraction noextract
let grouped_step
  (row:Shared.hidden_row)
  (width:reduction_width{width>1})
  (lane:natlt (width/2))
  : Lemma
      (grouped_sum row (width/2) lane ==
       grouped_sum row width lane +. grouped_sum row width (lane+width/2)) =
  assert (2*(128/width) == 128/(width/2));
  assert (2*(width/2) == width);
  rsum_strided (128/width) lane (width/2) (lane_sum row);
  rsum_ext (128/width)
    (fun i -> lane_sum row (lane+i*(2*(width/2))))
    (fun i -> lane_sum row (lane+i*width));
  rsum_ext (128/width)
    (fun i -> lane_sum row (lane+width/2+i*(2*(width/2))))
    (fun i -> lane_sum row (lane+width/2+i*width));
  grouped_def row (width/2) lane;
  grouped_def row width lane;
  grouped_def row width (lane+width/2)

inline_for_extraction noextract
let grouped_final (row:Shared.hidden_row)
  : Lemma (grouped_sum row 1 0 == Shared.sum_hidden (square row)) =
  rsum_ext 128 (fun i -> lane_sum row (0+i*1)) (lane_sum row);
  lanes_sum_to_hidden row;
  grouped_def row 1 0
#pop-options

inline_for_extraction noextract
let square_inside (row:Shared.hidden_row) (d:Shared.hidden_index)
  : Lemma (square row d == row d *. row d) = ()

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
let shared_square_sum_definition (row:Shared.hidden_row)
  : Lemma (
    Shared.sum_hidden (square row) ==
    Sum.sum #real #Sum.real_add_semigroup 0 1024 (square row)) = ()

#push-options "--fuel 1 --ifuel 0 --z3rlimit 60"
inline_for_extraction noextract
let shared_mean_from_squares (row:Shared.hidden_row)
  : Lemma (
    Shared.mean_square row ==
    Shared.sum_hidden (square row) *. (1.0R /. 1024.0R)) =
  let rec equal_sums (lo:nat{lo<=1024})
    : Lemma (
        Sum.sum #real #Sum.real_add_semigroup lo 1024 (square row) ==
        Sum.sum #real #Sum.real_add_semigroup lo 1024
          (fun (d:Shared.hidden_index) -> row d *. row d))
        (decreases (1024-lo)) =
    if lo < 1024 then (
      equal_sums (lo+1);
      square_inside row lo;
      Sum.sum_pop_left #real #Sum.real_add_semigroup lo 1024 (square row);
      Sum.sum_pop_left #real #Sum.real_add_semigroup lo 1024
        (fun (d:Shared.hidden_index) -> row d *. row d)
    )
  in
  equal_sums 0;
  shared_mean_definition row;
  shared_square_sum_definition row
#pop-options

inline_for_extraction noextract
let epsilon () : Tot (x:float{x %~ (Shared.rms_epsilon <: real)}) =
  let numerator = of_int #float 1L in
  let denominator = of_int #float 1000000L in
  of_int_approx #float 1L;
  of_int_approx #float 1000000L;
  div_approx numerator denominator 1.0R 1000000.0R;
  div numerator denominator

inline_for_extraction noextract
let inverse_width () : Tot (x:float{x %~ (1.0R /. 1024.0R)}) =
  let numerator = of_int #float 1L in
  let denominator = of_int #float 1024L in
  of_int_approx #float 1L;
  of_int_approx #float 1024L;
  div_approx numerator denominator 1.0R 1024.0R;
  div numerator denominator

inline_for_extraction noextract
fn add_related (r s:erased real) (a b:float)
  preserves gpu
  requires pure (a %~ reveal r /\ b %~ reveal s)
  returns c:float
  ensures pure (c %~ (reveal r +. reveal s))
{
  a_add a b (reveal r) (reveal s);
  add a b
}

inline_for_extraction noextract
fn mul_related (r s:erased real) (a b:float)
  preserves gpu
  requires pure (a %~ reveal r /\ b %~ reveal s)
  returns c:float
  ensures pure (c %~ (reveal r *. reveal s))
{
  a_mul a b (reveal r) (reveal s);
  mul a b
}

inline_for_extraction noextract
fn widen_related (r:erased real) (a:bf16)
  preserves gpu
  requires pure (a %~ reveal r)
  returns c:float
  ensures pure (c %~ reveal r)
{
  Cast.cast_bf16_to_f32_ok a (reveal r);
  Cast.cast_bf16_to_f32 a
}

inline_for_extraction noextract
fn narrow_related (r:erased real) (a:float)
  preserves gpu
  requires pure (a %~ reveal r)
  returns c:bf16
  ensures pure (c %~ reveal r)
{
  Cast.cast_f32_to_bf16_ok a (reveal r);
  Cast.cast_f32_to_bf16 a
}

inline_for_extraction noextract
let flat_hidden (rows:Shared.hidden_rows) (i:natlt hidden_elements)
  : GTot real =
  rows (i/1024) (i%1024)

inline_for_extraction noextract
let flat_representation (values:lseq bf16 hidden_elements)
  (rows:Shared.hidden_rows) : GTot prop =
  forall (i:natlt hidden_elements).
    Seq.index values i %~ flat_hidden rows i

inline_for_extraction noextract
let added_rows (#n:nat) (logical:input_denotation n) : GTot Shared.hidden_rows =
  fun p d -> logical.residual p d +. logical.delta p d

inline_for_extraction noextract
let square_representation (values:lseq float hidden_elements)
  (rows:Shared.hidden_rows) : GTot prop =
  forall (i:natlt hidden_elements).
    Seq.index values i %~ (flat_hidden rows i *. flat_hidden rows i)

inline_for_extraction noextract
fn residual_kernel
  (#n:erased nat)
  (residual delta output:larray bf16 hidden_elements)
  (#before:erased (input_snapshot n))
  (#logical:erased (input_denotation n))
  (#rp #dp:perm)
  (#old:erased (seq bf16))
  preserves gpu **
    residual |-> Frac rp (reveal before).residual_values **
    delta |-> Frac dp (reveal before).delta_values
  requires output |-> old
  requires pure (input_representation (reveal before) (reveal logical))
  ensures exists* (values:lseq bf16 hidden_elements).
    output |-> values **
    pure (flat_representation values (added_rows (reveal logical)))
{
  pts_to_len output;
  let mut i = 0sz;
  while (!i <^ 7168sz)
    invariant exists* (j:sz{j<=7168}) (values:lseq bf16 hidden_elements).
      i |-> j ** output |-> values **
      pure (forall (k:natlt hidden_elements).
        k < j ==> Seq.index values k %~
          flat_hidden (added_rows (reveal logical)) k)
    decreases (7168 - !i)
  {
    let j = !i;
    let r = residual.(j);
    let d = delta.(j);
    residual_add_refines r d
      ((reveal logical).residual (j/1024) (j%1024))
      ((reveal logical).delta (j/1024) (j%1024));
    let result = Cast.cast_f32_to_bf16
      (add (Cast.cast_bf16_to_f32 r) (Cast.cast_bf16_to_f32 d));
    output.(j) <- result;
    i := j +^ 1sz;
  }
}

(* Publishing the squares separately prevents CUDA FMA contraction across
   the source's explicit multiplication/addition rounding boundary. *)
inline_for_extraction noextract
fn square_kernel
  (hidden:larray bf16 hidden_elements)
  (squares:larray float hidden_elements)
  (#values:erased (lseq bf16 hidden_elements))
  (#rows:erased Shared.hidden_rows)
  (#hp:perm)
  (#old:erased (seq float))
  preserves gpu ** hidden |-> Frac hp values
  requires squares |-> old
  requires pure (flat_representation (reveal values) (reveal rows))
  ensures exists* (results:lseq float hidden_elements).
    squares |-> results **
    pure (square_representation results (reveal rows))
{
  pts_to_len squares;
  let mut i = 0sz;
  while (!i <^ 7168sz)
    invariant exists* (j:sz{j<=7168}) (results:lseq float hidden_elements).
      i |-> j ** squares |-> results **
      pure (forall (k:natlt hidden_elements).
        k < j ==> Seq.index results k %~
          (flat_hidden (reveal rows) k *. flat_hidden (reveal rows) k))
    decreases (7168 - !i)
  {
    let j = !i;
    let x = hidden.(j);
    let xr = hide (flat_hidden (reveal rows) j);
    let xf = widen_related xr x;
    let squared = mul_related xr xr xf xf;
    squares.(j) <- squared;
    i := j +^ 1sz;
  }
}

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn lane_reduce
  (squares:larray float hidden_elements)
  (#values:erased (lseq float hidden_elements))
  (#rows:erased Shared.hidden_rows)
  (#sp:perm)
  (p:szlt 7) (lane:szlt 128)
  preserves gpu ** squares |-> Frac sp values
  requires pure (square_representation (reveal values) (reveal rows))
  returns result:float
  ensures pure (result %~ lane_sum ((reveal rows) p) lane)
{
  let base = p *^ 1024sz +^ lane *^ 4sz;
  let a0 = squares.(base);
  let a1 = squares.(base +^ 1sz);
  let a2 = squares.(base +^ 2sz);
  let a3 = squares.(base +^ 3sz);
  let b0 = squares.(base +^ 512sz);
  let b1 = squares.(base +^ 513sz);
  let b2 = squares.(base +^ 514sz);
  let b3 = squares.(base +^ 515sz);
  let x0 = hide (square ((reveal rows) p) (4*lane));
  let x1 = hide (square ((reveal rows) p) (4*lane+1));
  let x2 = hide (square ((reveal rows) p) (4*lane+2));
  let x3 = hide (square ((reveal rows) p) (4*lane+3));
  let y0 = hide (square ((reveal rows) p) (4*lane+512));
  let y1 = hide (square ((reveal rows) p) (4*lane+513));
  let y2 = hide (square ((reveal rows) p) (4*lane+514));
  let y3 = hide (square ((reveal rows) p) (4*lane+515));
  let v0 = add_related (hide 0.0R) x0 (zero #float) a0;
  let v1 = add_related (hide 0.0R) x1 (zero #float) a1;
  let v2 = add_related (hide 0.0R) x2 (zero #float) a2;
  let v3 = add_related (hide 0.0R) x3 (zero #float) a3;
  let t0 = add_related x0 y0 v0 b0;
  let t1 = add_related x1 y1 v1 b1;
  let t2 = add_related x2 y2 v2 b2;
  let t3 = add_related x3 y3 v3 b3;
  let s0 = hide (reveal x0 +. reveal y0);
  let s1 = hide (reveal x1 +. reveal y1);
  let s2 = hide (reveal x2 +. reveal y2);
  let s3 = hide (reveal x3 +. reveal y3);
  let t01 = add_related s0 s1 t0 t1;
  let t012 = add_related (hide (reveal s0 +. reveal s1)) s2 t01 t2;
  let result = add_related
    (hide ((reveal s0 +. reveal s1) +. reveal s2)) s3 t012 t3;
  assert pure (result %~
    (((reveal s0 +. reveal s1) +. reveal s2) +. reveal s3));
  assert pure (lane_sum ((reveal rows) p) lane ==
    (((reveal s0 +. reveal s1) +. reveal s2) +. reveal s3));
  result
}
#pop-options

inline_for_extraction noextract
fn reduce_row
  (squares:larray float hidden_elements)
  (#values:erased (lseq float hidden_elements))
  (#rows:erased Shared.hidden_rows)
  (#sp:perm)
  (p:szlt 7)
  preserves gpu ** squares |-> Frac sp values
  requires pure (square_representation (reveal values) (reveal rows))
  returns result:float
  ensures pure (result %~ (Shared.mean_square ((reveal rows) p) <: real))
{
  let mut partial = [| zero #float #_; 128sz |];
  let mut lane = 0sz;
  while (!lane <^ 128sz)
    invariant exists* (j:sz{j<=128}) (values:lseq float 128).
      lane |-> j ** partial |-> values **
      pure (forall (k:natlt 128). k < j ==>
        Seq.index values k %~ grouped_sum ((reveal rows) p) 128 k)
    decreases (128 - !lane)
  {
    let j = !lane;
    let value = lane_reduce squares #values #rows #sp p j;
    grouped_initial ((reveal rows) p) j;
    partial.(j) <- value;
    lane := j +^ 1sz;
  };
  let mut width = 128sz;
  while (!width >^ 1sz)
    invariant exists* (w:sz{valid_width (SZ.v w)})
      (values:lseq float 128).
      width |-> w ** partial |-> values **
      pure (forall (k:natlt w).
        Seq.index values k %~ grouped_sum ((reveal rows) p) w k)
    decreases (SZ.v !width)
  {
    let w = !width;
    let half = w /^ 2sz;
    width_halves w;
    let mut j = 0sz;
    while (!j <^ half)
      invariant exists* (k:sz{k<=half}) (values:lseq float 128).
        j |-> k ** partial |-> values **
        pure (forall (i:natlt w).
          (if i < k then
            Seq.index values i %~ grouped_sum ((reveal rows) p) half i
          else Seq.index values i %~ grouped_sum ((reveal rows) p) w i))
      decreases (half - !j)
    {
      let k = !j;
      let left = partial.(k);
      let right = partial.(k +^ half);
      let result = add_related
        (hide (grouped_sum ((reveal rows) p) w k))
        (hide (grouped_sum ((reveal rows) p) w (k+half)))
        left right;
      grouped_step ((reveal rows) p) w k;
      partial.(k) <- result;
      j := k +^ 1sz;
    };
    width_halves w;
    width := half;
  };
  let sum = partial.(0sz);
  grouped_final ((reveal rows) p);
  shared_mean_from_squares ((reveal rows) p);
  let factor = inverse_width ();
  mul_related
    (hide (Shared.sum_hidden (square ((reveal rows) p))))
    (hide (1.0R /. 1024.0R)) sum factor
}

inline_for_extraction noextract
let mean_representation (values:lseq float 7)
  (rows:Shared.hidden_rows) : GTot prop =
  forall (p:Shared.prompt_index).
    Seq.index values p %~ (Shared.mean_square (rows p) <: real)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
inline_for_extraction noextract
let prefix_update (#a:Type0) {| scalar a, real_like a |}
  (#n:nat) (values:lseq a n)
  (logical:natlt n -> GTot real) (j:natlt n) (value:a)
  : Lemma
      (requires
        (forall (k:natlt n). k < j ==> Seq.index values k %~ logical k) /\
        value %~ logical j)
      (ensures
        forall (k:natlt n). k < j+1 ==>
          Seq.index (Seq.upd values j value) k %~ logical k) = ()
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
inline_for_extraction noextract
let normalized_cell_definition (x:Shared.inputs)
  (p:Shared.prompt_index) (d:Shared.hidden_index)
  : Lemma (
    flat_hidden (Shared.final_rms_norm x) (p*1024+d) ==
    (x.hidden_in p d *. Shared.normalization_scale (x.hidden_in p)) *.
      x.norm_weight d) =
  Kuiper.Math.div_mod_of_mul_add 1024 p d;
  assert (
    flat_hidden (Shared.final_rms_norm x) (p*1024+d) ==
    (x.hidden_in p d *. Shared.normalization_scale (x.hidden_in p)) *.
      x.norm_weight d) by
    (FStar.Tactics.V2.norm
      [delta_only [`%flat_hidden; `%Shared.final_rms_norm]];
     FStar.Tactics.V2.smt ())
#pop-options

inline_for_extraction noextract
fn mean_kernel
  (squares:larray float hidden_elements)
  (means:larray float 7)
  (#values:erased (lseq float hidden_elements))
  (#rows:erased Shared.hidden_rows)
  (#sp:perm)
  (#old:erased (seq float))
  preserves gpu ** squares |-> Frac sp values
  requires means |-> old
  requires pure (square_representation (reveal values) (reveal rows))
  ensures exists* (results:lseq float 7).
    means |-> results ** pure (mean_representation results (reveal rows))
{
  pts_to_len means;
  let mut p = 0sz;
  while (!p <^ 7sz)
    invariant exists* (j:sz{j<=7}) (results:lseq float 7).
      p |-> j ** means |-> results **
      pure (forall (k:Shared.prompt_index). k < j ==>
        Seq.index results k %~ (Shared.mean_square ((reveal rows) k) <: real))
    decreases (7 - !p)
  {
    let j = !p;
    let mean = reduce_row squares #values #rows #sp j;
    with prior. assert means |-> prior;
    prefix_update prior
      (fun k -> (Shared.mean_square ((reveal rows) k) <: real)) j mean;
    means.(j) <- mean;
    p := j +^ 1sz;
  }
}

#push-options "--fuel 1 --ifuel 1 --z3rlimit 100"

open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }
open Pulse.Lib.Trade

module MMA = Kuiper.TensorCore.MMA
module G = Kuiper.Spec.GEMM
module C = Kuiper.Chest
module T = Kuiper.Tensor
module FA = Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc

inline_for_extraction noextract
instance c1024 : concrete_sz 1024 = { x = 1024sz }

inline_for_extraction noextract
let a_layout = rm 16 1024

inline_for_extraction noextract
let b_layout = cm 1024 8

inline_for_extraction noextract
let c_layout = rm 16 8

inline_for_extraction noextract
instance a_concrete : ctlayout a_layout =
  Kuiper.Tensor.Layout.Alg.c_l2_row_major 16 1024sz

inline_for_extraction noextract
instance b_concrete : ctlayout b_layout =
  Kuiper.Tensor.Layout.Alg.c_l2_col_major 1024sz 8

inline_for_extraction noextract
instance c_concrete : ctlayout c_layout =
  Kuiper.Tensor.Layout.Alg.c_l2_row_major 16 8sz

inline_for_extraction noextract
let rec mma_accum
  (a:chest2 bf16 16 1024)
  (b:chest2 bf16 1024 8)
  (count:nat{count<=64})
  : GTot (chest2 float 16 8) (decreases count) =
  if count = 0 then C.const (16 @| 8 @| INil) (zero #float)
  else MMA.emma (mma_accum a b (count-1))
    (ematrix_subtile a 16 16 0 (count-1))
    (ematrix_subtile b 16 8 (count-1) 0)

inline_for_extraction noextract
let mma_accum_zero (a:chest2 bf16 16 1024) (b:chest2 bf16 1024 8)
  : Lemma (mma_accum a b 0 == C.const (16 @| 8 @| INil) (zero #float)) = ()

inline_for_extraction noextract
let mma_accum_step
  (a:chest2 bf16 16 1024) (b:chest2 bf16 1024 8)
  (k:natlt 64)
  : Lemma (mma_accum a b (k+1) ==
      MMA.emma (mma_accum a b k)
        (ematrix_subtile a 16 16 0 k)
        (ematrix_subtile b 16 8 k 0)) = ()

inline_for_extraction noextract
let real_accum
  (a:chest2 real 16 1024) (b:chest2 real 1024 8)
  (count:nat{count<=64}) : GTot (chest2 real 16 8) =
  G.__gmatmul_single (C.const (16 @| 8 @| INil) 0.0R) G.matmul G.matplus
    (ematrix_tiled a 16 16) (ematrix_tiled b 16 8) 0 0 count

inline_for_extraction noextract
let real_accum_step
  (a:chest2 real 16 1024) (b:chest2 real 1024 8)
  (k:natlt 64)
  : Lemma (real_accum a b (k+1) ==
      G.matplus (real_accum a b k)
        (G.matmul (ematrix_subtile a 16 16 0 k)
          (ematrix_subtile b 16 8 k 0))) =
  G.__gmatmul_single_lemma (C.const (16 @| 8 @| INil) 0.0R) G.matmul G.matplus
    (ematrix_tiled a 16 16) (ematrix_tiled b 16 8) 0 0 (k+1);
  macc_ematrix_tiled a 16 16 0 k;
  macc_ematrix_tiled b 16 8 k 0;
  assert (C.equal (real_accum a b (k+1))
    (G.matplus (real_accum a b k)
      (G.matmul (ematrix_subtile a 16 16 0 k)
        (ematrix_subtile b 16 8 k 0))))

inline_for_extraction noextract
let rec mma_accum_approx
  (a:chest2 bf16 16 1024) (b:chest2 bf16 1024 8)
  (ra:chest2 real 16 1024) (rb:chest2 real 1024 8)
  (count:nat{count<=64})
  : Lemma
      (requires a %~ ra /\ b %~ rb)
      (ensures mma_accum a b count %~ real_accum ra rb count)
      (decreases count) =
  if count = 0 then (
    mma_accum_zero a b;
    assert (C.const (16 @| 8 @| INil) (zero #float) %~
      C.const (16 @| 8 @| INil) 0.0R)
  ) else (
    mma_accum_approx a b ra rb (count-1);
    FA.subtile_approx a ra 16 16 0 (count-1);
    FA.subtile_approx b rb 16 8 (count-1) 0;
    MMA.emma_approx_lemma
      (mma_accum a b (count-1))
      (ematrix_subtile a 16 16 0 (count-1))
      (ematrix_subtile b 16 8 (count-1) 0)
      (real_accum ra rb (count-1))
      (ematrix_subtile ra 16 16 0 (count-1))
      (ematrix_subtile rb 16 8 (count-1) 0);
    mma_accum_step a b (count-1);
    real_accum_step ra rb (count-1)
  )

inline_for_extraction noextract
let real_add_zero (x:real) : squash (add x zero == x /\ add zero x == x) = ()

inline_for_extraction noextract
let real_add_assoc (x y z:real)
  : squash (add x (add y z) == add (add x y) z) = ()

inline_for_extraction noextract
let real_accum_complete
  (a:chest2 real 16 1024) (b:chest2 real 1024 8)
  : Lemma (real_accum a b 64 == G.matmul a b) =
  G.matmul_tiles_lemma real_add_zero real_add_assoc
    16 8 16 (C.const (16 @| 8 @| INil) 0.0R) a b 0 0;
  assert (C.equal (ematrix_subtile a 16 1024 0 0) a);
  assert (C.equal (ematrix_subtile b 1024 8 0 0) b);
  assert (C.equal (G.matplus (C.const (16 @| 8 @| INil) 0.0R) (G.matmul a b))
    (G.matmul a b))

inline_for_extraction noextract
let mma_product_approx
  (a:chest2 bf16 16 1024) (b:chest2 bf16 1024 8)
  (ra:chest2 real 16 1024) (rb:chest2 real 1024 8)
  : Lemma
      (requires a %~ ra /\ b %~ rb)
      (ensures mma_accum a b 64 %~ G.matmul ra rb) =
  mma_accum_approx a b ra rb 64;
  real_accum_complete ra rb

inline_for_extraction noextract
fn mma_kernel
  (a:array2 bf16 a_layout)
  (b:array2 bf16 b_layout)
  (c:array2 float c_layout)
  (#av:chest2 bf16 16 1024)
  (#bv:chest2 bf16 1024 8)
  (#cv:chest2 float 16 8)
  preserves gpu **
    a |-> Frac (1.0R /. 32.0R) av **
    b |-> Frac (1.0R /. 32.0R) bv
  requires c |-> Frac (1.0R /. 32.0R) cv
  ensures c |-> Frac (1.0R /. 32.0R) (mma_accum av bv 64)
{
  let fr = MMA.alloc_fragment ();
  MMA.fill fr (zero #float);
  mma_accum_zero av bv;
  let mut i = 0sz;
  while (!i <^ 64sz)
    invariant exists* (k:sz{k<=64}).
      i |-> k ** fr |-> mma_accum av bv k
    decreases (64 - !i)
  {
    let k = !i;
    let at = array2_extract_tile_ro' a 16 16 0 (SZ.v k);
    let bt = array2_extract_tile_ro' b 16 8 (SZ.v k) 0;
    MMA.mma_sync at bt fr;
    elim_trade
      (at |-> Frac (1.0R /. 32.0R) (ematrix_subtile av 16 16 0 k))
      (a |-> Frac (1.0R /. 32.0R) av);
    elim_trade
      (bt |-> Frac (1.0R /. 32.0R) (ematrix_subtile bv 16 8 k 0))
      (b |-> Frac (1.0R /. 32.0R) bv);
    mma_accum_step av bv k;
    i := k +^ 1sz;
  };
  MMA.store fr c;
  drop_ (fr |-> mma_accum av bv 64);
}

inline_for_extraction noextract
fn mma_thread
  (a:array2 bf16 a_layout) (b:array2 bf16 b_layout) (c:array2 float c_layout)
  (#av:chest2 bf16 16 1024) (#bv:chest2 bf16 1024 8) (#cv:chest2 float 16 8)
  (tid:szlt 32) ()
  norewrite
  requires gpu **
    (a |-> Frac (1.0R /. 32.0R) av **
     b |-> Frac (1.0R /. 32.0R) bv **
     c |-> Frac (1.0R /. 32.0R) cv) **
    thread_id 32 tid
  ensures gpu **
    (a |-> Frac (1.0R /. 32.0R) av **
     b |-> Frac (1.0R /. 32.0R) bv **
     c |-> Frac (1.0R /. 32.0R) (mma_accum av bv 64)) **
    thread_id 32 tid
{
  mma_kernel a b c #av #bv #cv;
}

ghost
fn mma_setup
  (a:array2 bf16 a_layout) (b:array2 bf16 b_layout) (c:array2 float c_layout)
  (#av:chest2 bf16 16 1024) (#bv:chest2 bf16 1024 8) (#cv:chest2 float 16 8)
  ()
  norewrite
  requires a |-> av ** b |-> bv ** c |-> cv
  ensures (forall+ (tid:natlt 32).
    a |-> Frac (1.0R /. 32.0R) av **
    b |-> Frac (1.0R /. 32.0R) bv **
    c |-> Frac (1.0R /. 32.0R) cv) ** emp
{
  T.tensor_share_n a 32;
  T.tensor_share_n b 32;
  T.tensor_share_n c 32;
  forevery_zip3
    (fun (_:natlt 32) -> a |-> Frac (1.0R /. 32.0R) av)
    (fun (_:natlt 32) -> b |-> Frac (1.0R /. 32.0R) bv)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) cv);
}

ghost
fn mma_teardown
  (a:array2 bf16 a_layout) (b:array2 bf16 b_layout) (c:array2 float c_layout)
  (#av:chest2 bf16 16 1024) (#bv:chest2 bf16 1024 8)
  ()
  norewrite
  requires (forall+ (tid:natlt 32).
    a |-> Frac (1.0R /. 32.0R) av **
    b |-> Frac (1.0R /. 32.0R) bv **
    c |-> Frac (1.0R /. 32.0R) (mma_accum av bv 64)) ** emp
  ensures a |-> av ** b |-> bv ** c |-> mma_accum av bv 64
{
  forevery_unzip3
    (fun (_:natlt 32) -> a |-> Frac (1.0R /. 32.0R) av)
    (fun (_:natlt 32) -> b |-> Frac (1.0R /. 32.0R) bv)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) (mma_accum av bv 64));
  T.tensor_gather_n a 32;
  T.tensor_gather_n b 32;
  T.tensor_gather_n c 32;
}

inline_for_extraction noextract
let mma_desc
  (a:array2 bf16 a_layout{T.is_global a})
  (b:array2 bf16 b_layout{T.is_global b})
  (c:array2 float c_layout{T.is_global c})
  (#av:chest2 bf16 16 1024)
  (#bv:chest2 bf16 1024 8)
  (#cv:chest2 float 16 8)
  : kernel_desc_1_n
      (a |-> av ** b |-> bv ** c |-> cv)
      (a |-> av ** b |-> bv ** c |-> mma_accum av bv 64) =
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
      c |-> Frac (1.0R /. 32.0R) (mma_accum av bv 64));
    f = mma_thread a b c #av #bv #cv;
    block_setup = mma_setup a b c #av #bv #cv;
    block_teardown = mma_teardown a b c #av #bv;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve
  }

inline_for_extraction noextract
fn mma_sync_host
  (a:array2 bf16 a_layout{T.is_global a})
  (b:array2 bf16 b_layout{T.is_global b})
  (c:array2 float c_layout{T.is_global c})
  (#av:chest2 bf16 16 1024)
  (#bv:chest2 bf16 1024 8)
  (#cv:chest2 float 16 8)
  preserves cpu **
    on gpu_loc (a |-> av) **
    on gpu_loc (b |-> bv)
  requires on gpu_loc (c |-> cv)
  ensures on gpu_loc (c |-> mma_accum av bv 64)
{
  launch_sync (mma_desc a b c #av #bv #cv);
}

inline_for_extraction noextract
let projection_a (embedding:Shared.tied_embedding) (tile:natlt 9496)
  : GTot (chest2 real 16 1024) =
  C.mk2 (fun r d -> embedding (tile*16+r) d)

inline_for_extraction noextract
let projection_b (hidden:Shared.hidden_rows) : GTot (chest2 real 1024 8) =
  C.mk2 (fun d p -> if p < 7 then hidden p d else 0.0R)

#push-options "--fuel 4 --ifuel 1 --z3rlimit 40"
inline_for_extraction noextract
let projection_a_at (embedding:Shared.tied_embedding) (tile:natlt 9496)
  (r:natlt 16) (d:Shared.hidden_index)
  : Lemma (C.acc2 (projection_a embedding tile) r d ==
      embedding (tile*16+r) d) = ()

inline_for_extraction noextract
let projection_b_at (hidden:Shared.hidden_rows)
  (p:Shared.prompt_index) (d:Shared.hidden_index)
  : Lemma (C.acc2 (projection_b hidden) d p == hidden p d) = ()
#pop-options

inline_for_extraction noextract
let projection_terms (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496) (r:natlt 16) (p:Shared.prompt_index)
  : GTot Shared.hidden_row =
  fun d -> hidden p d *. embedding (tile*16+r) d

inline_for_extraction noextract
let projection_terms_at
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496) (r:natlt 16) (p:Shared.prompt_index) (d:Shared.hidden_index)
  : Lemma (projection_terms embedding hidden tile r p d ==
      hidden p d *. embedding (tile*16+r) d) = ()

inline_for_extraction noextract
let real_madd (a b c:real)
  : Lemma (add a (mul b c) == a +. b *. c) =
  assert_norm (add a (mul b c) == a +. b *. c)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
inline_for_extraction noextract
let rec projection_prefix
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496) (r:natlt 16) (p:Shared.prompt_index)
  (count:nat{count<=1024})
  : Lemma (
      G.__matmul_single (projection_a embedding tile) (projection_b hidden)
        r p count ==
      Sum.sum #real #Sum.real_add_semigroup 0 count
        (projection_terms embedding hidden tile r p))
      (decreases count) =
  let a = projection_a embedding tile in
  let b = projection_b hidden in
  let terms = projection_terms embedding hidden tile r p in
  if count = 0 then (
    G.matmul_zero_lemma a b r p;
    assert_norm (Sum.sum #real #Sum.real_add_semigroup 0 0 terms == 0.0R);
    assert_norm (zero #real == 0.0R)
  ) else (
    projection_prefix embedding hidden tile r p (count-1);
    G.matmul_single_lemma a b r p count;
    projection_a_at embedding tile r (count-1);
    projection_b_at hidden p (count-1);
    real_madd (G.__matmul_single a b r p (count-1))
      (embedding (tile*16+r) (count-1)) (hidden p (count-1));
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 count terms;
    assert_norm (
      Sum.op #real #Sum.real_add_semigroup
        (Sum.sum #real #Sum.real_add_semigroup 0 (count-1) terms)
        (terms (count-1)) ==
      Sum.sum #real #Sum.real_add_semigroup 0 (count-1) terms +. terms (count-1));
    projection_terms_at embedding hidden tile r p (count-1)
  )
#pop-options

inline_for_extraction noextract
let projection_spec_def
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496) (r:natlt 16) (p:Shared.prompt_index)
  : Lemma (
      Shared.tied_projection embedding (hidden p) (tile*16+r) ==
      Sum.sum #real #Sum.real_add_semigroup 0 1024
        (projection_terms embedding hidden tile r p)) =
  assert (
    Shared.tied_projection embedding (hidden p) (tile*16+r) ==
    Sum.sum #real #Sum.real_add_semigroup 0 1024
      (projection_terms embedding hidden tile r p)) by
    (FStar.Tactics.V2.norm
      [delta_only [`%Shared.tied_projection; `%Shared.sum_hidden;
        `%Shared.hidden_size; `%projection_terms]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let projection_matmul_def
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496) (r:natlt 16) (p:Shared.prompt_index)
  : Lemma (
    G.matmul_single (projection_a embedding tile) (projection_b hidden) r p ==
    G.__matmul_single (projection_a embedding tile) (projection_b hidden) r p 1024)
  = ()

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
inline_for_extraction noextract
let projection_cell
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496) (r:natlt 16) (p:Shared.prompt_index)
  : Lemma (
      C.acc2 (G.matmul (projection_a embedding tile) (projection_b hidden)) r p ==
      Shared.tied_projection embedding (hidden p) (tile*16+r)) =
  projection_prefix embedding hidden tile r p 1024;
  G.lemma_matmul_index (projection_a embedding tile) (projection_b hidden) r p;
  projection_spec_def embedding hidden tile r p;
  projection_matmul_def embedding hidden tile r p
#pop-options

inline_for_extraction noextract
let projection_result
  (values:chest2 float 16 8)
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496)
  : GTot prop =
  forall (r:natlt 16) (p:Shared.prompt_index).
    C.acc2 values r p %~ Shared.tied_projection embedding (hidden p) (tile*16+r)

inline_for_extraction noextract
let projection_result_from_mma
  (values:chest2 float 16 8)
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496)
  : Lemma
      (requires values %~ G.matmul (projection_a embedding tile) (projection_b hidden))
      (ensures projection_result values embedding hidden tile) =
  let point (r:natlt 16) (p:Shared.prompt_index)
    : Lemma (C.acc2 values r p %~
      Shared.tied_projection embedding (hidden p) (tile*16+r)) =
    projection_cell embedding hidden tile r p
  in FStar.Classical.forall_intro_2 point

inline_for_extraction noextract
fn copy_projection_tile
  (#n:erased nat{n<=2147483647})
  (weights:larray bf16 n)
  (offset:sz{offset+embedding_elements<=n})
  (a:array2 bf16 a_layout)
  (#wv:erased (lseq bf16 n))
  (#av:chest2 bf16 16 1024)
  (#wp:perm)
  (#embedding:erased Shared.tied_embedding)
  (tile:szlt 9496)
  preserves gpu ** weights |-> Frac wp wv
  requires a |-> av
  requires pure (forall (v:Shared.vocab_index) (d:Shared.hidden_index).
    Seq.index (reveal wv) (offset+v*1024+d) %~ (reveal embedding) v d)
  ensures exists* (values:chest2 bf16 16 1024).
    a |-> values ** pure (values %~ projection_a (reveal embedding) tile)
{
  open Pulse.Lib.Array;
  let mut i = 0sz;
  while (!i <^ 16384sz)
    invariant exists* (j:sz{j<=16384}) (values:chest2 bf16 16 1024).
      i |-> j ** a |-> values **
      pure (forall (r:natlt 16) (d:Shared.hidden_index).
        r*1024+d < j ==> C.acc2 values r d %~
          (reveal embedding) (tile*16+r) d)
    decreases (16384 - !i)
  {
    let j = !i;
    let r = j /^ 1024sz;
    let d = j %^ 1024sz;
    let value = weights.(offset +^ tile *^ 16384sz +^ j);
    T.tensor_write a (r, (d, ())) value;
    i := j +^ 1sz;
  };
}

inline_for_extraction noextract
fn pad_projection_input
  (hidden:larray bf16 hidden_elements)
  (b:array2 bf16 b_layout)
  (#hv:erased (lseq bf16 hidden_elements))
  (#bv:chest2 bf16 1024 8)
  (#hp:perm)
  (#rows:erased Shared.hidden_rows)
  preserves gpu ** hidden |-> Frac hp hv
  requires b |-> bv
  requires pure (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
    Seq.index (reveal hv) (p*1024+d) %~ (reveal rows) p d)
  ensures exists* (values:chest2 bf16 1024 8).
    b |-> values ** pure (values %~ projection_b (reveal rows))
{
  open Pulse.Lib.Array;
  let mut i = 0sz;
  while (!i <^ 8192sz)
    invariant exists* (j:sz{j<=8192}) (values:chest2 bf16 1024 8).
      i |-> j ** b |-> values **
      pure (forall (p:natlt 8) (d:Shared.hidden_index).
        p*1024+d < j ==> C.acc2 values d p %~
          (if p < 7 then (reveal rows) p d else 0.0R))
    decreases (8192 - !i)
  {
    let j = !i;
    let p = j /^ 1024sz;
    let d = j %^ 1024sz;
    if (p <^ 7sz) {
      let value = hidden.(j);
      T.tensor_write b (d, (p, ())) value;
    } else {
      T.tensor_write b (d, (p, ())) (zero #bf16);
    };
    i := j +^ 1sz;
  };
}

inline_for_extraction noextract
let logits_prefix
  (values:lseq bf16 logit_elements)
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (columns:nat{columns<=151936}) : GTot prop =
  forall (p:Shared.prompt_index) (v:Shared.vocab_index). v < columns ==>
    Seq.index values (p*151936+v) %~ Shared.tied_projection embedding (hidden p) v

inline_for_extraction noextract
let logits_tile_progress
  (values:lseq bf16 logit_elements)
  (embedding:Shared.tied_embedding) (hidden:Shared.hidden_rows)
  (tile:natlt 9496) (count:nat{count<=112}) : GTot prop =
  forall (p:Shared.prompt_index) (v:Shared.vocab_index).
    (v < tile*16 \/
     (tile*16 <= v /\ v < tile*16+16 /\ p*16+v-tile*16 < count)) ==>
    Seq.index values (p*151936+v) %~ Shared.tied_projection embedding (hidden p) v

inline_for_extraction noextract
fn write_projection_tile
  (c:array2 float c_layout)
  (output:larray bf16 logit_elements)
  (#cv:chest2 float 16 8)
  (#cp:perm)
  (#old:erased (lseq bf16 logit_elements))
  (#embedding:erased Shared.tied_embedding)
  (#hidden:erased Shared.hidden_rows)
  (tile:szlt 9496)
  preserves gpu ** c |-> Frac cp cv
  requires output |-> old
  requires pure (
    projection_result cv (reveal embedding) (reveal hidden) tile /\
    logits_prefix (reveal old) (reveal embedding) (reveal hidden) (tile*16))
  ensures exists* (values:lseq bf16 logit_elements).
    output |-> values **
    pure (logits_prefix values (reveal embedding) (reveal hidden) ((tile+1)*16))
{
  open Pulse.Lib.Array;
  let mut i = 0sz;
  while (!i <^ 112sz)
    invariant exists* (j:sz{j<=112}) (values:lseq bf16 logit_elements).
      i |-> j ** output |-> values **
      pure (logits_tile_progress values (reveal embedding) (reveal hidden) tile j)
    decreases (112 - !i)
  {
    let j = !i;
    let p = j /^ 16sz;
    let r = j %^ 16sz;
    let v = tile *^ 16sz +^ r;
    let score = T.tensor_read c (r, (p, ()));
    Cast.cast_f32_to_bf16_ok score
      (Shared.tied_projection (reveal embedding) ((reveal hidden) p) v);
    let result = Cast.cast_f32_to_bf16 score;
    let columns = 148sz *^ 1024sz +^ 384sz;
    output.(p *^ columns +^ v) <- result;
    i := j +^ 1sz;
  };
}

instance send_global_array_contents
  (#a:Type0)
  (arr:array a{is_global_array arr})
  (#f:perm)
  (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents) =
  Kuiper.Array.Core.is_send_pts_to arr #f contents

inline_for_extraction noextract
fn projection_host
  (#n:erased nat{n<=2147483647})
  (weights:larray bf16 n{is_global_array weights})
  (offset:sz{offset+embedding_elements<=n})
  (hidden:larray bf16 hidden_elements{is_global_array hidden})
  (output:larray bf16 logit_elements{is_global_array output})
  (#wv:erased (lseq bf16 n))
  (#hv:erased (lseq bf16 hidden_elements))
  (#old:erased (lseq bf16 logit_elements))
  (#wp #hp:perm)
  (#embedding:erased Shared.tied_embedding)
  (#rows:erased Shared.hidden_rows)
  preserves cpu **
    on gpu_loc (weights |-> Frac wp wv) **
    on gpu_loc (hidden |-> Frac hp hv)
  requires on gpu_loc (output |-> old)
  requires pure (
    (forall (v:Shared.vocab_index) (d:Shared.hidden_index).
      Seq.index (reveal wv) (offset+v*1024+d) %~ (reveal embedding) v d) /\
    (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
      Seq.index (reveal hv) (p*1024+d) %~ (reveal rows) p d))
  ensures exists* (values:lseq bf16 logit_elements).
    on gpu_loc (output |-> values) **
    pure (logits_prefix values (reveal embedding) (reveal rows) 151936)
{
  let a = T.alloc0 #bf16 16384sz a_layout;
  let b = T.alloc0 #bf16 8192sz b_layout;
  let c = T.alloc0 #float 128sz c_layout;
  with b0. assert on gpu_loc (b |-> b0);
  launch_kernel_1 (fun () -> pad_projection_input hidden b #hv #b0 #hp #rows);
  with bv. assert on gpu_loc (b |-> bv);
  let mut tile = 0sz;
  while (!tile <^ 9496sz)
    invariant exists* (j:sz{j<=9496})
      (av:chest2 bf16 16 1024) (cv:chest2 float 16 8)
      (values:lseq bf16 logit_elements).
      tile |-> j **
      on gpu_loc (a |-> av) **
      on gpu_loc (c |-> cv) **
      on gpu_loc (output |-> values) **
      pure (logits_prefix values (reveal embedding) (reveal rows) (j*16))
    decreases (9496 - !tile)
  {
    let j = !tile;
    with av. assert on gpu_loc (a |-> av);
    with cv. assert on gpu_loc (c |-> cv);
    launch_kernel_1 (fun () ->
      copy_projection_tile weights offset a #wv #av #wp #embedding j);
    with a1. assert on gpu_loc (a |-> a1);
    mma_sync_host a b c #a1 #bv #cv;
    mma_product_approx a1 bv (projection_a (reveal embedding) j)
      (projection_b (reveal rows));
    projection_result_from_mma (mma_accum a1 bv 64)
      (reveal embedding) (reveal rows) j;
    with ov. assert on gpu_loc (output |-> ov);
    launch_kernel_1 (fun () ->
      write_projection_tile c output #(mma_accum a1 bv 64) #1.0R #ov
        #embedding #rows j);
    tile := j +^ 1sz;
  };
  T.free a;
  T.free b;
  T.free c;
}
#pop-options

inline_for_extraction noextract
fn normalize_kernel
  (#n:erased nat)
  (weights:larray bf16 n)
  (norm_offset:sz{norm_offset+1024 <= n})
  (hidden:larray bf16 hidden_elements)
  (means:larray float 7)
  (output:larray bf16 hidden_elements)
  (#wv:erased (lseq bf16 n))
  (#hv:erased (lseq bf16 hidden_elements))
  (#mv:erased (lseq float 7))
  (#x:erased Shared.inputs)
  (#wp #hp #mp:perm)
  (#old:erased (lseq bf16 hidden_elements))
  preserves gpu **
    weights |-> Frac wp wv **
    hidden |-> Frac hp hv **
    means |-> Frac mp mv
  requires output |-> old
  requires pure (
    flat_representation (reveal hv) (reveal x).hidden_in /\
    mean_representation (reveal mv) (reveal x).hidden_in /\
    (forall (d:Shared.hidden_index).
      Seq.index (reveal wv) (norm_offset+d) %~ (reveal x).norm_weight d))
  ensures exists* (results:lseq bf16 hidden_elements).
    output |-> results **
    pure (flat_representation results (Shared.final_rms_norm (reveal x)))
{
  open Pulse.Lib.Array;
  let mut p = 0sz;
  while (!p <^ 7sz)
    invariant exists* (j:sz{j<=7}) (results:lseq bf16 hidden_elements).
      p |-> j ** output |-> results **
      pure (forall (i:natlt hidden_elements). i < j*1024 ==>
        Seq.index results i %~ flat_hidden (Shared.final_rms_norm (reveal x)) i)
    decreases (7 - !p)
  {
    let j = !p;
    let mean = means.(j);
    let mr = hide (Shared.mean_square ((reveal x).hidden_in j) <: real);
    let e = epsilon ();
    let shifted = add_related mr (hide (Shared.rms_epsilon <: real)) mean e;
    rsqrt_approx shifted
      (Shared.mean_square ((reveal x).hidden_in j) +. Shared.rms_epsilon);
    let scale = rsqrt shifted;
    let sr = hide (Shared.normalization_scale ((reveal x).hidden_in j) <: real);
    let mut col = 0sz;
    while (!col <^ 1024sz)
      invariant exists* (d:sz{d<=1024}) (results:lseq bf16 hidden_elements).
        col |-> d ** output |-> results **
        pure (forall (i:natlt hidden_elements). i < j*1024+d ==>
          Seq.index results i %~
            flat_hidden (Shared.final_rms_norm (reveal x)) i)
      decreases (1024 - !col)
    {
      let d = !col;
      let idx = j *^ 1024sz +^ d;
      assert pure (SZ.v idx == j*1024+d /\ idx < hidden_elements);
      let value = hidden.(idx);
      let gamma = weights.(norm_offset +^ d);
      let xr = hide ((reveal x).hidden_in j d);
      let gr = hide ((reveal x).norm_weight d);
      let xf = widen_related xr value;
      let scaled = mul_related xr sr xf scale;
      let nr = hide (reveal xr *. reveal sr);
      let rounded = narrow_related nr scaled;
      let rounded_f = widen_related nr rounded;
      let gamma_f = widen_related gr gamma;
      let weighted = mul_related nr gr rounded_f gamma_f;
      let result = narrow_related (hide (reveal nr *. reveal gr)) weighted;
      normalized_cell_definition (reveal x) j d;
      assert pure (result %~ flat_hidden (Shared.final_rms_norm (reveal x)) idx);
      with prior. assert output |-> prior;
      prefix_update prior (flat_hidden (Shared.final_rms_norm (reveal x))) idx result;
      output.(idx) <- result;
      col := d +^ 1sz;
    };
    p := j +^ 1sz;
  }
}

module IC = FStar.Int.Cast
module I64 = FStar.Int64
module U64 = FStar.UInt64

inline_for_extraction noextract
let offset_size (offset:I64.t{0<=I64.v offset /\ I64.v offset<=2147483647})
  : Tot (n:sz{SZ.v n == I64.v offset}) =
  SZ.uint32_to_sizet (IC.int64_to_uint32 offset)

inline_for_extraction noextract
let input_views (#n:nat)
  (norm_offset:weight_offset n Shared.hidden_size)
  (projection_offset:weight_offset n embedding_elements)
  (before:input_snapshot n) (logical:input_denotation n)
  : Lemma
      (requires input_representation before logical)
      (ensures (
        let x = map_inputs norm_offset projection_offset logical in
        added_rows logical == x.hidden_in /\
        (forall (d:Shared.hidden_index).
          Seq.index before.packed_values (I64.v norm_offset+d) %~ x.norm_weight d) /\
        (forall (v:Shared.vocab_index) (d:Shared.hidden_index).
          Seq.index before.packed_values (I64.v projection_offset+v*1024+d) %~
            x.embedding v d))) = ()

inline_for_extraction noextract
let flat_hidden_at (rows:Shared.hidden_rows)
  (p:Shared.prompt_index) (d:Shared.hidden_index)
  : Lemma (flat_hidden rows (p*1024+d) == rows p d) =
  Kuiper.Math.div_mod_of_mul_add 1024 p d

inline_for_extraction noextract
let flat_to_rows (values:lseq bf16 hidden_elements) (rows:Shared.hidden_rows)
  : Lemma
      (requires flat_representation values rows)
      (ensures forall (p:Shared.prompt_index) (d:Shared.hidden_index).
        Seq.index values (p*1024+d) %~ rows p d) =
  let point (p:Shared.prompt_index) (d:Shared.hidden_index)
    : Lemma (Seq.index values (p*1024+d) %~ rows p d) =
    flat_hidden_at rows p d
  in FStar.Classical.forall_intro_2 point

inline_for_extraction noextract
let complete_output_representation (x:Shared.inputs)
  (hidden:lseq bf16 hidden_elements) (logits:lseq bf16 logit_elements)
  : Lemma
      (requires
        flat_representation hidden (Shared.final_rms_norm x) /\
        logits_prefix logits x.embedding (Shared.final_rms_norm x) 151936)
      (ensures output_representation hidden logits (expected_outputs x)) =
  flat_to_rows hidden (Shared.final_rms_norm x)

fn kplay_run
  (weights_numel:packed_extent)
  (weights:larray bf16 (U64.v weights_numel){is_global_array weights})
  (residual:larray bf16 hidden_elements{is_global_array residual})
  (delta:larray bf16 hidden_elements{is_global_array delta})
  (norm_offset:weight_offset (U64.v weights_numel) Shared.hidden_size)
  (projection_offset:weight_offset (U64.v weights_numel) embedding_elements)
  (#before:erased (input_snapshot (U64.v weights_numel)))
  (#logical:erased (input_denotation (U64.v weights_numel)))
  (#weights_perm #residual_perm #delta_perm:perm)
  (output_hidden:larray bf16 hidden_elements{is_global_array output_hidden})
  (output_logits:larray bf16 logit_elements{is_global_array output_logits})
  preserves cpu **
    on gpu_loc (weights |-> Frac weights_perm (reveal before).packed_values) **
    on gpu_loc (residual |-> Frac residual_perm (reveal before).residual_values) **
    on gpu_loc (delta |-> Frac delta_perm (reveal before).delta_values)
  requires pure (input_representation (reveal before) (reveal logical))
  requires on gpu_loc (
    exists* (old_hidden:lseq bf16 hidden_elements).
      output_hidden |-> old_hidden)
  requires on gpu_loc (
    exists* (old_logits:lseq bf16 logit_elements).
      output_logits |-> old_logits)
  ensures exists*
    (hidden:lseq bf16 hidden_elements)
    (logits:lseq bf16 logit_elements).
    on gpu_loc (output_hidden |-> hidden ** output_logits |-> logits) **
    pure (postcondition norm_offset projection_offset (reveal logical)
      (reveal before) (reveal before) hidden logits)
{
  let noff = offset_size norm_offset;
  let poff = offset_size projection_offset;
  let x = hide (map_inputs norm_offset projection_offset (reveal logical));
  input_views norm_offset projection_offset (reveal before) (reveal logical);
  let combined = gpu_array_alloc #bf16 7168sz;
  let squares = gpu_array_alloc #float 7168sz;
  let means = gpu_array_alloc #float 7sz;
  with (h0:lseq bf16 hidden_elements). assert on gpu_loc (combined |-> h0);
  with (s0:lseq float hidden_elements). assert on gpu_loc (squares |-> s0);
  with (m0:lseq float 7). assert on gpu_loc (means |-> m0);
  with (oh0:lseq bf16 hidden_elements). assert on gpu_loc (output_hidden |-> oh0);
  with (ol0:lseq bf16 logit_elements). assert on gpu_loc (output_logits |-> ol0);
  launch_kernel_1 (fun () ->
    residual_kernel residual delta combined #before #logical
      #residual_perm #delta_perm #(hide (reveal h0 <: seq bf16)));
  with (hv:lseq bf16 hidden_elements). assert on gpu_loc (combined |-> hv);
  launch_kernel_1 (fun () ->
    square_kernel combined squares
      #(hide (reveal hv <: lseq bf16 hidden_elements))
      #(hide (reveal x).hidden_in) #1.0R
      #(hide (reveal s0 <: seq float)));
  with (sv:lseq float hidden_elements). assert on gpu_loc (squares |-> sv);
  launch_kernel_1 (fun () ->
    mean_kernel squares means
      #(hide (reveal sv <: lseq float hidden_elements))
      #(hide (reveal x).hidden_in) #1.0R #(hide (reveal m0 <: seq float)));
  with (mv:lseq float 7). assert on gpu_loc (means |-> mv);
  launch_kernel_1 (fun () ->
    normalize_kernel weights noff combined means output_hidden
      #(hide (reveal before).packed_values)
      #(hide (reveal hv <: lseq bf16 hidden_elements))
      #(hide (reveal mv <: lseq float 7)) #x
      #weights_perm #1.0R #1.0R
      #(hide (reveal oh0 <: lseq bf16 hidden_elements)));
  with (hidden:lseq bf16 hidden_elements). assert on gpu_loc (output_hidden |-> hidden);
  flat_to_rows hidden (Shared.final_rms_norm (reveal x));
  projection_host weights poff output_hidden output_logits
    #(hide (reveal before).packed_values)
    #(hide (reveal hidden <: lseq bf16 hidden_elements))
    #(hide (reveal ol0 <: lseq bf16 logit_elements))
    #weights_perm #1.0R
    #(hide (reveal x).embedding) #(hide (Shared.final_rms_norm (reveal x)));
  with (logits:lseq bf16 logit_elements). assert on gpu_loc (output_logits |-> logits);
  complete_output_representation (reveal x) hidden logits;
  represented_outputs_establish_postcondition
    norm_offset projection_offset (reveal logical) (reveal before) hidden logits;
  gpu_array_free combined;
  gpu_array_free squares;
  gpu_array_free means;
}
