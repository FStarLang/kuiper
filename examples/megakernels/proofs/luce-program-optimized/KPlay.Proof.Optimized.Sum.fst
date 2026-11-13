module KPlay.Proof.Optimized.Sum

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
module S = FStar.Seq
module SC = Kuiper.Seq.Common
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module C = FStar.Classical

inline_for_extraction noextract
let rec prefix (n:nat) (f:nat -> GTot real)
  : GTot real (decreases n) =
  if n = 0 then 0.0R else prefix (n - 1) f +. f (n - 1)

let prefix_step (n:nat) (f:nat -> GTot real)
  : Lemma (prefix (n + 1) f == prefix n f +. f n) = ()

let rec prefix_ext (n:nat) (f g:nat -> GTot real)
  : Lemma
      (requires forall (i:nat). i < n ==> f i == g i)
      (ensures prefix n f == prefix n g)
      (decreases n) =
  if n > 0 then prefix_ext (n - 1) f g

let rec prefix_add (n:nat) (f g:nat -> GTot real)
  : Lemma
      (ensures prefix n (fun i -> f i +. g i) == prefix n f +. prefix n g)
      (decreases n) =
  if n > 0 then prefix_add (n - 1) f g

let rec prefix_scale (n:nat) (scale:real) (f:nat -> GTot real)
  : Lemma
      (ensures prefix n (fun i -> scale *. f i) == scale *. prefix n f)
      (decreases n) =
  if n > 0 then prefix_scale (n - 1) scale f

let rec prefix_zero (n:nat)
  : Lemma (ensures prefix n (fun _ -> 0.0R) == 0.0R) (decreases n) =
  if n > 0 then prefix_zero (n - 1)

let rec prefix_split (n m:nat) (f:nat -> GTot real)
  : Lemma
      (ensures prefix (n + m) f == prefix n f +. prefix m (fun i -> f (n + i)))
      (decreases m) =
  if m > 0 then prefix_split n (m - 1) f

let rec prefix_swap (n m:nat) (f:nat -> nat -> GTot real)
  : Lemma
      (ensures prefix n (fun i -> prefix m (f i)) ==
        prefix m (fun j -> prefix n (fun i -> f i j)))
      (decreases n) =
  if n = 0 then (
    prefix_zero m;
    prefix_ext m (fun j -> prefix n (fun i -> f i j)) (fun _ -> 0.0R)
  ) else (
    prefix_swap (n - 1) m f;
    prefix_add m (fun j -> prefix (n - 1) (fun i -> f i j)) (f (n - 1));
    prefix_ext m
      (fun j -> prefix n (fun i -> f i j))
      (fun j -> prefix (n - 1) (fun i -> f i j) +. f (n - 1) j)
  )

let rec prefix_blocks (blocks width:nat) (f:nat -> GTot real)
  : Lemma
      (ensures prefix blocks (fun b ->
        prefix width (fun i -> f (b * width + i))) == prefix (blocks * width) f)
      (decreases blocks) =
  if blocks > 0 then (
    prefix_step (blocks - 1) (fun b -> prefix width (fun i -> f (b * width + i)));
    prefix_blocks (blocks - 1) width f;
    prefix_split ((blocks - 1) * width) width f;
    assert (blocks * width == (blocks - 1) * width + width)
  )

inline_for_extraction noextract
let seq_value (s:seq real) (i:nat) : GTot real =
  if i < S.length s then S.index s i else 0.0R

let rec prefix_rsum (s:seq real) (n:nat{n <= S.length s})
  : Lemma
      (ensures prefix n (seq_value s) == rsum (S.slice s 0 n))
      (decreases n) =
  if n = 0 then assert (S.equal (S.slice s 0 0) (S.empty #real))
  else (
    prefix_rsum s (n - 1);
    SC.lemma_seq_fold_left_slice 0.0R (+.) s 0 (n - 1)
  )

let prefix_init (n:nat) (f:nat -> GTot real)
  : Lemma (prefix n f == rsum (S.init_ghost n f)) =
  let s = S.init_ghost n f in
  prefix_ext n f (seq_value s);
  prefix_rsum s n;
  assert (S.equal (S.slice s 0 n) s)

let rec prefix_kuiper (n:nat) (f:nat -> GTot real)
  : Lemma
      (ensures prefix n f == Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 n f)
      (decreases n) =
  if n = 0 then
    assert_norm (Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 0 f == 0.0R)
  else (
    prefix_kuiper (n - 1) f;
    Kuiper.Sum.sum_pop_right #real #Kuiper.Sum.real_add_semigroup 0 n f
  )

let rec bounded_sum_ext (#n:nat)
  (f g:(i:nat{i < n}) -> GTot real) (count:nat{count <= n})
  : Lemma
      (requires forall (i:nat{i < count}). f i == g i)
      (ensures
        Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 count f ==
        Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 count g)
      (decreases count) =
  if count > 0 then (
    bounded_sum_ext #n f g (count - 1);
    Kuiper.Sum.sum_pop_right #real #Kuiper.Sum.real_add_semigroup 0 count f;
    Kuiper.Sum.sum_pop_right #real #Kuiper.Sum.real_add_semigroup 0 count g
  )

inline_for_extraction noextract
let tree_width (n:nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32

inline_for_extraction noextract
let rec tree_value (f:nat -> GTot real) (width:nat{tree_width width}) (lane:nat)
  : GTot real (decreases 32 - width) =
  if width = 32 then f lane
  else tree_value f (2 * width) lane +. tree_value f (2 * width) (lane + width)

let rec tree_total (f:nat -> GTot real) (width:nat{tree_width width})
  : Lemma
      (ensures prefix width (tree_value f width) == prefix 32 f)
      (decreases 32 - width) =
  if width = 32 then prefix_ext 32 (tree_value f 32) f
  else (
    tree_total f (2 * width);
    prefix_ext width (tree_value f width)
      (fun i -> tree_value f (2 * width) i +. tree_value f (2 * width) (i + width));
    prefix_add width (tree_value f (2 * width))
      (fun i -> tree_value f (2 * width) (i + width));
    prefix_split width width (tree_value f (2 * width));
    prefix_ext width
      (fun i -> tree_value f (2 * width) (i + width))
      (fun i -> tree_value f (2 * width) (width + i))
  )

let tree_root (f:nat -> GTot real)
  : Lemma (tree_value f 1 0 == prefix 32 f) =
  tree_total f 1

inline_for_extraction noextract
let tree_invariant (original current:lseq f32 32) (width:nat{tree_width width}) : prop =
  forall (model:lseq real 32). original %~ model ==>
    (forall (i:nat{i < width}).
      S.index current i %~ tree_value (seq_value model) width i)

inline_for_extraction noextract
let tree_progress (original current:lseq f32 32)
  (distance:nat{tree_width distance /\ distance < 32}) (done:nat{done <= distance}) : prop =
  forall (model:lseq real 32). original %~ model ==>
    (forall (i:nat{i < 2 * distance}).
      S.index current i %~
        (if i < done then tree_value (seq_value model) distance i
         else tree_value (seq_value model) (2 * distance) i))

let tree_step (original current:lseq f32 32)
  (distance:nat{tree_width distance /\ distance < 32})
  (i:nat{i < distance}) (value:f32)
  : Lemma
      (requires tree_progress original current distance i /\
        (forall (xr yr:real).
          S.index current i %~ xr /\ S.index current (i + distance) %~ yr ==>
          value %~ (xr +. yr)))
      (ensures tree_progress original (S.upd current i value) distance (i + 1)) = ()

let tree_finished (original current:lseq f32 32)
  : Lemma
      (requires tree_invariant original current 1)
      (ensures forall (model:lseq real 32).
        original %~ model ==> S.index current 0 %~ prefix 32 (seq_value model)) =
  let related (model:lseq real 32)
    : Lemma (tree_value (seq_value model) 1 0 == prefix 32 (seq_value model)) =
    tree_root (seq_value model)
  in C.forall_intro related

noextract
fn reduce_level (partial:larray f32 32)
  (distance:sz{tree_width (SZ.v distance) /\ distance < 32})
  (#original #before:erased (lseq f32 32))
  preserves gpu
  requires partial |-> before ** pure (tree_invariant original before (2 * SZ.v distance))
  ensures exists* (after:lseq f32 32).
    partial |-> after ** pure (tree_invariant original after (SZ.v distance))
{
  let mut lane = 0sz;
  while (!lane <^ distance)
    invariant exists* (i:sz{i <= distance}) (values:lseq f32 32).
      lane |-> i ** partial |-> values **
      pure (tree_progress original values (SZ.v distance) (SZ.v i))
    decreases (distance - !lane)
  {
    let i = !lane;
    with values. assert partial |-> values;
    let a = partial.(i);
    let b = partial.(i +^ distance);
    let value = FP.add_rn_ftz a b;
    tree_step original values (SZ.v distance) (SZ.v i) value;
    partial.(i) <- value;
    lane := i +^ 1sz;
  }
}

noextract
fn sum32 (partial:larray f32 32) (#before:erased (lseq f32 32))
  preserves gpu
  requires partial |-> before
  returns result:f32
  ensures exists* (after:lseq f32 32).
    partial |-> after ** pure (forall (model:lseq real 32).
      before %~ model ==> result %~ prefix 32 (seq_value model))
{
  assert pure (tree_invariant before before 32);
  let mut distance = 16sz;
  while (0sz <^ !distance)
    invariant exists* (d:sz{d == 0sz \/ (tree_width (SZ.v d) /\ d < 32)})
      (values:lseq f32 32).
      distance |-> d ** partial |-> values **
      pure (tree_invariant before values (if d == 0sz then 1 else 2 * SZ.v d))
    decreases (SZ.v !distance)
  {
    let d = !distance;
    reduce_level partial d #before;
    distance := d /^ 2sz;
  };
  with values. assert partial |-> values;
  tree_finished before values;
  partial.(0sz)
}

let lift_sum32 (#parameter:Type)
  (premise:parameter -> prop) (model:parameter -> nat -> GTot real)
  (values:lseq f32 32) (result:f32)
  : Lemma
      (requires
        (forall (p:parameter). premise p ==>
          (forall (i:nat{i < 32}). S.index values i %~ model p i)) /\
        (forall (r:lseq real 32).
          values %~ r ==> result %~ prefix 32 (seq_value r)))
      (ensures forall (p:parameter). premise p ==> result %~ prefix 32 (model p)) =
  let related (p:parameter{premise p}) : Lemma (result %~ prefix 32 (model p)) =
    let r = S.init_ghost 32 (model p) in
    assert (values %~ r);
    prefix_ext 32 (seq_value r) (model p)
  in C.forall_intro related

let lift_fma (#parameter:Type)
  (premise:parameter -> prop) (xm ym zm:parameter -> GTot real)
  (x y z result:f32)
  : Lemma
      (requires
        (forall (p:parameter). premise p ==>
          x %~ xm p /\ y %~ ym p /\ z %~ zm p) /\
        (forall (xr yr zr:real).
          x %~ xr /\ y %~ yr /\ z %~ zr ==> result %~ (xr *. yr +. zr)))
      (ensures forall (p:parameter). premise p ==>
        result %~ (xm p *. ym p +. zm p)) =
  let related (p:parameter{premise p})
    : Lemma (result %~ (xm p *. ym p +. zm p)) = ()
  in C.forall_intro related

let lift_add (#parameter:Type)
  (premise:parameter -> prop) (xm ym:parameter -> GTot real) (x y result:f32)
  : Lemma
      (requires
        (forall (p:parameter). premise p ==> x %~ xm p /\ y %~ ym p) /\
        (forall (xr yr:real). x %~ xr /\ y %~ yr ==> result %~ (xr +. yr)))
      (ensures forall (p:parameter). premise p ==> result %~ (xm p +. ym p)) =
  let related (p:parameter{premise p})
    : Lemma (result %~ (xm p +. ym p)) = ()
  in C.forall_intro related

let lift_mul (#parameter:Type)
  (premise:parameter -> prop) (xm ym:parameter -> GTot real) (x y result:f32)
  : Lemma
      (requires
        (forall (p:parameter). premise p ==> x %~ xm p /\ y %~ ym p) /\
        (forall (xr yr:real). x %~ xr /\ y %~ yr ==> result %~ (xr *. yr)))
      (ensures forall (p:parameter). premise p ==> result %~ (xm p *. ym p)) =
  let related (p:parameter{premise p})
    : Lemma (result %~ (xm p *. ym p)) = ()
  in C.forall_intro related
