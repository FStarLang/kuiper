module KPlay.Functional

#lang-pulse
open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module U64 = FStar.UInt64
module IC = FStar.Int.Cast
module SZ = Kuiper.SizeT
module Cast = Kuiper.Float.Casts.Base
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module C = FStar.Classical
module ML = FStar.Math.Lemmas
module Shared = KPlay.SharedSpec


(* Order proof and implementation. *)
#push-options "--fuel 2 --ifuel 1"
inline_for_extraction noextract unfold let order_nonnan (x : f32) : prop = not (F.is_nan x)

inline_for_extraction noextract
let order_eq_symmetric (x y : f32)
  : Lemma (F.ieee_eq x y <==> F.ieee_eq y x)
  = F.eq_spec x y; F.eq_spec y x

inline_for_extraction noextract
let order_eq_transitive (x y z : f32)
  : Lemma
      (requires F.ieee_eq x y /\ F.ieee_eq y z)
      (ensures F.ieee_eq x z)
  = F.eq_spec x y; F.eq_spec y z; F.eq_spec x z

inline_for_extraction noextract
let order_lt_irreflexive (x : f32)
  : Lemma (not (lt x x))
  = if lt x x then (
      F32.lt_ordered x x;
      F.eq_spec x x;
      F.lte_is_lt_or_eq x x;
      F.negate_lt_is_lte x x)

inline_for_extraction noextract
let order_trichotomy (x y : f32)
  : Lemma
      (requires order_nonnan x /\ order_nonnan y)
      (ensures lt x y \/ F.ieee_eq x y \/ lt y x)
  = F.negate_lt_is_lte x y;
    F.lte_is_lt_or_eq y x;
    order_eq_symmetric x y

inline_for_extraction noextract
let order_lt_not_eq (x y : f32)
  : Lemma
      (requires lt x y)
      (ensures not (F.ieee_eq x y) /\ not (F.ieee_eq y x))
  = F32.lt_ordered x y;
    F.negate_lt_is_lte x y;
    F.lte_is_lt_or_eq y x;
    order_eq_symmetric x y

inline_for_extraction noextract
let order_lt_eq_right (x y z : f32)
  : Lemma
      (requires lt x y /\ F.ieee_eq y z)
      (ensures lt x z)
  = F32.lt_ordered x y;
    F.eq_spec y z;
    order_trichotomy x z;
    if lt z x then (
      F32.lt_transitive z x y;
      order_lt_not_eq z y;
      order_eq_symmetric y z)
    else if F.ieee_eq x z then (
      order_eq_symmetric y z;
      order_eq_transitive x z y;
      order_lt_not_eq x y)

inline_for_extraction noextract
let order_eq_lt_left (x y z : f32)
  : Lemma
      (requires F.ieee_eq x y /\ lt y z)
      (ensures lt x z)
  = F32.lt_ordered y z;
    F.eq_spec x y;
    order_trichotomy x z;
    if lt z x then (
      F32.lt_transitive y z x;
      order_lt_not_eq y x;
      order_eq_symmetric x y)
    else if F.ieee_eq x z then (
      order_eq_symmetric x y;
      order_eq_transitive y x z;
      order_lt_not_eq y z)
#pop-options


(* The interface's visit keys also describe the strict tree's tie order. *)
#push-options "--fuel 2 --ifuel 1"
inline_for_extraction noextract
let visit_r2_bounds (x : nat) : Lemma (visit_r2 x < 4) = ()
inline_for_extraction noextract
let visit_r3_bounds (x : nat) : Lemma (visit_r3 x < 8) = ()
inline_for_extraction noextract
let visit_r5_bounds (x : nat) : Lemma (visit_r5 x < 32) = ()

#push-options "--z3rlimit 100"
inline_for_extraction noextract
let visit_r2_inv (x : nat{x < 4}) : Lemma (visit_r2 (visit_r2 x) == x) = ()
inline_for_extraction noextract
let visit_r3_inv (x : nat{x < 8}) : Lemma (visit_r3 (visit_r3 x) == x) = ()
inline_for_extraction noextract
let visit_r5_inv (x : nat{x < 32}) : Lemma (visit_r5 (visit_r5 x) == x) =
  if x == 0 then () else if x == 1 then () else if x == 2 then ()
  else if x == 3 then () else if x == 4 then () else if x == 5 then ()
  else if x == 6 then () else if x == 7 then () else if x == 8 then ()
  else if x == 9 then () else if x == 10 then () else if x == 11 then ()
  else if x == 12 then () else if x == 13 then () else if x == 14 then ()
  else if x == 15 then () else if x == 16 then () else if x == 17 then ()
  else if x == 18 then () else if x == 19 then () else if x == 20 then ()
  else if x == 21 then () else if x == 22 then () else if x == 23 then ()
  else if x == 24 then () else if x == 25 then () else if x == 26 then ()
  else if x == 27 then () else if x == 28 then () else if x == 29 then ()
  else if x == 30 then () else ()
#pop-options

inline_for_extraction noextract
let visit_cr_bounds (c : nat) : Lemma (0 <= visit_cr c /\ visit_cr c < 256) =
  visit_r3_bounds (c / 32); visit_r5_bounds (c % 32)

inline_for_extraction noextract
let visit_tr_bounds (t : nat) : Lemma (0 <= visit_tr t /\ visit_tr t < 128) =
  visit_r2_bounds (t / 32); visit_r5_bounds (t % 32)

inline_for_extraction noextract
let visit_cr_inv (c : nat{c < 256}) : Lemma (visit_cr (visit_cr c) == c) =
  visit_r3_bounds (c / 32); visit_r5_bounds (c % 32);
  visit_r3_inv (c / 32); visit_r5_inv (c % 32)

inline_for_extraction noextract
let visit_tr_inv (t : nat{t < 128}) : Lemma (visit_tr (visit_tr t) == t) =
  visit_r2_bounds (t / 32); visit_r5_bounds (t % 32);
  visit_r2_inv (t / 32); visit_r5_inv (t % 32)

#push-options "--fuel 0 --ifuel 0"
inline_for_extraction noextract
let visit_split_digits (c : nat) (t : nat{t < 128}) (k : nat{k < 16})
  : Lemma
      ((2048 * c + 16 * t + k) / 2048 == c /\
       (2048 * c + 16 * t + k) % 2048 == 16 * t + k /\
       (16 * t + k) / 16 == t /\ (16 * t + k) % 16 == k /\
       (2048 * c + 16 * t + k) % 16 == k)
  = ML.division_definition (2048 * c + 16 * t + k) 2048 c;
    ML.lemma_div_mod (2048 * c + 16 * t + k) 2048;
    ML.division_definition (16 * t + k) 16 t;
    ML.lemma_div_mod (16 * t + k) 16;
    ML.lemma_mod_plus k (128 * c + t) 16;
    ML.small_mod k 16
#pop-options

inline_for_extraction noextract
let visit_key_digits (j : nat{j < 153600})
  : Lemma
      (visit_key j / 2048 == visit_cr (j / 1600) /\
       (visit_key j % 2048) / 16 == visit_tr ((j % 1600) % 128) /\
       visit_key j % 16 == (j % 1600) / 128 /\
       0 <= visit_key j /\ visit_key j < 524288)
  = visit_cr_bounds (j / 1600); visit_tr_bounds ((j % 1600) % 128);
    visit_split_digits (visit_cr (j / 1600)) (visit_tr ((j % 1600) % 128)) ((j % 1600) / 128)

inline_for_extraction noextract
let visit_key_def (j : nat)
  : Lemma
      (visit_key j == 2048 * visit_cr (j / 1600) +
        16 * visit_tr ((j % 1600) % 128) + (j % 1600) / 128)
  = ()

#push-options "--fuel 0 --ifuel 0"
inline_for_extraction noextract
let visit_key_decode
  (c : nat{c < 256}) (t : nat{t < 128}) (k : nat{k < 16})
  (j : nat{j < 153600})
  : Lemma
      (visit_key j == 2048 * visit_cr c + 16 * visit_tr t + k <==>
       j / 1600 == c /\ (j % 1600) % 128 == t /\
       (j % 1600) / 128 == k)
  = visit_key_digits j; visit_key_def j;
    visit_cr_bounds c; visit_tr_bounds t;
    visit_cr_inv c; visit_cr_inv (j / 1600);
    visit_tr_inv t; visit_tr_inv ((j % 1600) % 128);
    if visit_key j == 2048 * visit_cr c + 16 * visit_tr t + k then (
      visit_split_digits (visit_cr c) (visit_tr t) k;
      assert (visit_key j / 2048 == visit_cr c);
      assert ((visit_key j % 2048) / 16 == visit_tr t);
      assert (visit_key j % 16 == k);
      assert (visit_cr (j / 1600) == visit_cr c);
      assert (visit_tr ((j % 1600) % 128) == visit_tr t))

inline_for_extraction noextract
let visit_key_singleton
  (c : nat{c < 96}) (t : nat{t < 128}) (k : nat{k < 16})
  (j : nat{j < 153600})
  : Lemma
      (visit_key j == 2048 * visit_cr c + 16 * visit_tr t + k <==>
       t + 128 * k < 1600 /\ j == c * 1600 + t + 128 * k)
  = visit_key_decode c t k j;
    ML.lemma_div_mod j 1600;
    ML.lemma_div_mod (j % 1600) 128;
    if t + 128 * k < 1600 && j == c * 1600 + t + 128 * k then (
      ML.division_definition j 1600 c;
      assert (j % 1600 == t + 128 * k);
      ML.division_definition (j % 1600) 128 k)

inline_for_extraction noextract
let visit_thread_range
  (c : nat{c < 96}) (t : nat{t < 128})
  (j : nat{j < 153600})
  : Lemma
      (requires
        2048 * visit_cr c + 16 * visit_tr t <= visit_key j /\
        visit_key j < 2048 * visit_cr c + 16 * visit_tr t + 16)
      (ensures
        j / 1600 == c /\ (j % 1600) % 128 == t /\
        visit_key j == 2048 * visit_cr c + 16 * visit_tr t + (j % 1600) / 128)
  = visit_key_digits j;
    visit_cr_bounds c; visit_tr_bounds t;
    visit_cr_inv c; visit_cr_inv (j / 1600);
    visit_tr_inv t; visit_tr_inv ((j % 1600) % 128);
    ML.division_definition (visit_key j) 2048 (visit_cr c);
    ML.lemma_div_mod (visit_key j) 2048;
    assert (visit_key j / 2048 == visit_cr c);
    ML.division_definition (visit_key j % 2048) 16 (visit_tr t);
    assert ((visit_key j % 2048) / 16 == visit_tr t);
    assert (visit_cr (j / 1600) == visit_cr c);
    assert (visit_tr ((j % 1600) % 128) == visit_tr t)

inline_for_extraction noextract
let visit_absent_chunk (c : nat{96 <= c /\ c < 256}) (j : nat{j < 153600})
  : Lemma
      (not (2048 * visit_cr c <= visit_key j /\ visit_key j < 2048 * visit_cr c + 2048))
  = visit_key_digits j; visit_cr_bounds c;
    visit_cr_inv c; visit_cr_inv (j / 1600);
    if 2048 * visit_cr c <= visit_key j && visit_key j < 2048 * visit_cr c + 2048 then (
      ML.division_definition (visit_key j) 2048 (visit_cr c);
      assert (visit_key j / 2048 == visit_cr c);
      assert (visit_cr (j / 1600) == visit_cr c))
#pop-options

inline_for_extraction noextract let visit_width = w:nat{w == 4 \/ w == 8 \/ w == 32}
inline_for_extraction noextract let visit_active (w : visit_width) =
  a:nat{(a == 2 \/ a == 4 \/ a == 8 \/ a == 16 \/ a == 32) /\ a <= w}

inline_for_extraction noextract let visit_rev (w : visit_width) (i : nat) : nat =
  if w = 4 then visit_r2 i else if w = 8 then visit_r3 i else visit_r5 i

inline_for_extraction noextract
let visit_rev_bounds (w : visit_width) (i : nat) : Lemma (0 <= visit_rev w i /\ visit_rev w i < w) =
  if w == 4 then visit_r2_bounds i else if w == 8 then visit_r3_bounds i else visit_r5_bounds i

#push-options "--z3rlimit 100"
inline_for_extraction noextract
let visit_tree_step (w : visit_width) (a : visit_active w) (i : nat{i < a / 2})
  : Lemma
      (visit_rev w (i + a / 2) == visit_rev w i + w / a /\
       visit_rev w i % (2 * (w / a)) == 0 /\
       w / (a / 2) == 2 * (w / a))
  = if w == 4 then (if a == 2 then () else ())
    else if w == 8 then (if a == 2 then () else if a == 4 then () else ())
    else (if a == 2 then () else if a == 4 then () else if a == 8 then ()
          else if a == 16 then () else ())
#pop-options

#push-options "--fuel 0 --ifuel 0"
inline_for_extraction noextract
let visit_key_injective (j : nat{j < 153600}) (k : nat{k < 153600})
  : Lemma (requires visit_key j == visit_key k) (ensures j == k)
  = visit_key_digits j; visit_key_digits k;
    visit_cr_inv (j / 1600); visit_cr_inv (k / 1600);
    visit_tr_inv ((j % 1600) % 128); visit_tr_inv ((k % 1600) % 128);
    ML.lemma_div_mod j 1600; ML.lemma_div_mod k 1600;
    ML.lemma_div_mod (j % 1600) 128; ML.lemma_div_mod (k % 1600) 128

inline_for_extraction noextract
let visit_key_injective_all (j : nat{j < 153600})
  : Lemma (forall (k : nat{k < 153600}). visit_key k == visit_key j ==> k == j)
  = let point (k : nat{k < 153600})
      : Lemma (visit_key k == visit_key j ==> k == j)
      = if visit_key k = visit_key j then visit_key_injective k j in
    FStar.Classical.forall_intro point

inline_for_extraction noextract
let visit_all_keys_bounded ()
  : Lemma (forall (j : nat{j < 153600}). 0 <= visit_key j /\ visit_key j < 524288)
  = let point (j : nat{j < 153600})
      : Lemma (0 <= visit_key j /\ visit_key j < 524288)
      = visit_key_digits j in
    FStar.Classical.forall_intro point
#pop-options

inline_for_extraction noextract
let visit_rev_zero (w : visit_width) : Lemma (visit_rev w 0 == 0) = ()

inline_for_extraction noextract
let visit_tr_warp (w : nat{w < 4}) (l : nat{l < 32})
  : Lemma (visit_tr (32 * w + l) == 32 * visit_r2 w + visit_r5 l)
  = ()

inline_for_extraction noextract
let visit_cr_warp (w : nat{w < 8}) (l : nat{l < 32})
  : Lemma (visit_cr (32 * w + l) == 32 * visit_r3 w + visit_r5 l)
  = ()

inline_for_extraction noextract
let visit_rev32 (l : nat) : Lemma (visit_rev 32 l == visit_r5 l) = ()
inline_for_extraction noextract
let visit_rev4 (l : nat) : Lemma (visit_rev 4 l == visit_r2 l) = ()
inline_for_extraction noextract
let visit_rev8 (l : nat) : Lemma (visit_rev 8 l == visit_r3 l) = ()
#pop-options


(* Selection proof and implementation. *)
#push-options "--fuel 2 --ifuel 1"
#set-options "--fuel 1 --ifuel 1"

inline_for_extraction noextract type selection_ix = j:nat{j < 151936}
inline_for_extraction noextract type selection_encoding =
  | Selection_Local : chunk:nat{chunk < 96} -> selection_encoding
  | Selection_Packed : selection_encoding

noextract unfold let selection_decode (e : selection_encoding) (w : int) : int =
  match e with
  | Selection_Local c -> c * 1600 + w
  | Selection_Packed -> (w / 4294967296) * 1600 + w % 4294967296

noextract unfold let selection_valid (e : selection_encoding) (w : int) : prop =
  w == -1 \/
  (match e with
   | Selection_Local c ->
       0 <= w /\ w < 1600 /\ c * 1600 + w < 151936
   | Selection_Packed ->
       0 <= w /\ w < 96 * 4294967296 /\ w % 4294967296 < 1600 /\
       selection_decode e w < 151936)

noextract unfold let selection_inside (lo hi : nat) (j : selection_ix) : bool =
  lo <= visit_key j && visit_key j < hi

noextract unfold let selection_dominates (v : f32) (p : nat) (x : f32) (q : nat) : prop =
  lt x v \/ (F.ieee_eq x v /\ p <= q)

inline_for_extraction noextract
unfold let selection_take (av : f32) (ai : int) (bv : f32) (bi : int) : bool =
  lt av bv

inline_for_extraction noextract
let argmax_better (value : f32) (index : i64) (best : f32) (best_index : i64)
  : b:bool{b == selection_take best (I64.v best_index) value (I64.v index)} =
  lt best value

noextract unfold let selection_best
  (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (lo hi : nat) (v : f32) (w : int) : prop =
  selection_valid e w /\
  (if w == -1 then
    v == initial /\
    (forall (j : selection_ix).
      selection_inside lo hi j ==> not (lt initial (score j)))
   else
    let j = selection_decode e w in
    if 0 <= j && j < 151936 then
      selection_inside lo hi j /\ v == score j /\ lt initial v /\
      (forall (k : selection_ix).
        selection_inside lo hi k /\ lt initial (score k) ==>
        selection_dominates v (visit_key j) (score k) (visit_key k))
    else False)

inline_for_extraction noextract
let selection_best_valid
  (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (lo hi : nat) (v : f32) (w : int)
  : Lemma
      (requires selection_best score initial e lo hi v w)
      (ensures selection_valid e w /\ (w == -1 ==> v == initial) /\
        (w <> -1 ==> lt initial v))
  = ()

inline_for_extraction noextract
let selection_valid_decode (e : selection_encoding) (w : int)
  : Lemma
      (requires selection_valid e w /\ w <> -1)
      (ensures 0 <= selection_decode e w /\ selection_decode e w < 151936)
  = match e with Selection_Local _ -> () | Selection_Packed -> ()

inline_for_extraction noextract
let selection_dominates_transitive (a b c : f32) (pa pb pc : nat)
  : Lemma
      (requires selection_dominates b pb a pa /\ selection_dominates c pc b pb)
      (ensures selection_dominates c pc a pa)
  = if lt a b then (
      if lt b c then F32.lt_transitive a b c
      else order_lt_eq_right a b c)
    else if lt b c then order_eq_lt_left a b c
    else order_eq_transitive a b c

inline_for_extraction noextract
let selection_dominates_self (initial v : f32) (p : nat)
  : Lemma
      (requires lt initial v)
      (ensures selection_dominates v p v p)
  = F32.lt_ordered initial v; F.ieee_eq_refl v

inline_for_extraction noextract
let selection_merge
  (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (lo : nat) (mid : nat{lo <= mid}) (hi : nat{mid <= hi})
  (av bv : f32) (ai bi : int)
  : Lemma
      (requires selection_best score initial e lo mid av ai /\
                selection_best score initial e mid hi bv bi)
      (ensures selection_best score initial e lo hi
        (if selection_take av ai bv bi then bv else av)
        (if selection_take av ai bv bi then bi else ai))
  =
  if lt av bv then (
    if bi == -1 then (
      if ai == -1 then order_lt_irreflexive initial
      else (F32.lt_transitive initial av initial; order_lt_irreflexive initial))
    else (
      selection_valid_decode e bi;
      if ai <> -1 then selection_valid_decode e ai;
      let point (j : selection_ix)
        : Lemma
            (selection_inside lo hi j /\ lt initial (score j) ==>
             selection_dominates bv (visit_key (selection_decode e bi)) (score j) (visit_key j))
        = if selection_inside lo hi j && lt initial (score j) then (
            if visit_key j < mid then (
              assert (ai <> -1);
              selection_dominates_transitive (score j) av bv
                (visit_key j) (visit_key (selection_decode e ai)) (visit_key (selection_decode e bi)))) in
      C.forall_intro point))
  else if ai == -1 then (
    if bi <> -1 then selection_valid_decode e bi)
  else (
    selection_valid_decode e ai;
    if bi <> -1 then (
      selection_valid_decode e bi;
      F32.lt_ordered initial av;
      F32.lt_ordered initial bv;
      order_trichotomy av bv;
      order_eq_symmetric av bv;
      assert (selection_dominates av (visit_key (selection_decode e ai))
        bv (visit_key (selection_decode e bi))));
    let point (j : selection_ix)
      : Lemma
          (selection_inside lo hi j /\ lt initial (score j) ==>
           selection_dominates av (visit_key (selection_decode e ai)) (score j) (visit_key j))
      = if selection_inside lo hi j && lt initial (score j) then (
          if mid <= visit_key j then (
            assert (bi <> -1);
            selection_dominates_transitive (score j) bv av
              (visit_key j) (visit_key (selection_decode e bi)) (visit_key (selection_decode e ai)))) in
    C.forall_intro point)

inline_for_extraction noextract
let selection_empty (score : selection_ix -> f32) (initial : f32) (e : selection_encoding) (lo : nat)
  : Lemma (selection_best score initial e lo lo initial (-1))
  = ()

inline_for_extraction noextract
let selection_absent
  (score : selection_ix -> f32) (initial : f32) (e : selection_encoding) (lo hi : nat)
  : Lemma
      (requires forall (j : selection_ix). not (selection_inside lo hi j))
      (ensures selection_best score initial e lo hi initial (-1))
  = ()

inline_for_extraction noextract
let selection_unit_range (p q : nat)
  : Lemma (requires p <= q && q < p + 1) (ensures p == q)
  = ()

inline_for_extraction noextract
let selection_singleton
  (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (j : selection_ix) (w : int)
  : Lemma
      (requires selection_valid e w /\ w <> -1 /\ selection_decode e w == j /\
        0 <= visit_key j /\
        (forall (k : selection_ix). visit_key k == visit_key j ==> k == j))
      (ensures selection_best score initial e (visit_key j) (visit_key j + 1)
        (if lt initial (score j) then score j else initial)
        (if lt initial (score j) then w else -1))
  =
    let point (k : selection_ix) : Lemma
      (selection_inside (visit_key j) (visit_key j + 1) k ==> k == j) =
      if selection_inside (visit_key j) (visit_key j + 1) k then (
        selection_unit_range (visit_key j) (visit_key k);
        assert (k == j)) in
    C.forall_intro point;
    if lt initial (score j) then (
      selection_dominates_self initial (score j) (visit_key j);
      assert (w <> -1);
      assert (selection_decode e w == j))
    else (
      assert (forall (k : selection_ix).
        selection_inside (visit_key j) (visit_key j + 1) k ==>
          not (lt initial (score k))))

inline_for_extraction noextract
let selection_scan_step
  (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (lo : nat) (j : selection_ix) (av : f32) (ai w : int)
  : Lemma
      (requires
        lo <= visit_key j /\ selection_best score initial e lo (visit_key j) av ai /\
        selection_valid e w /\ w <> -1 /\ selection_decode e w == j /\
        (forall (k : selection_ix). visit_key k == visit_key j ==> k == j))
      (ensures selection_best score initial e lo (visit_key j + 1)
        (if selection_take av ai (score j) w then score j else av)
        (if selection_take av ai (score j) w then w else ai))
  =
  selection_singleton score initial e j w;
  selection_merge score initial e lo (visit_key j) (visit_key j + 1) av
    (if lt initial (score j) then score j else initial) ai
    (if lt initial (score j) then w else -1);
  if not (lt initial (score j)) then (
    if lt av (score j) then (
      if ai <> -1 then F32.lt_transitive initial av (score j));
    if lt av initial then (
      if ai == -1 then order_lt_irreflexive initial
      else (F32.lt_transitive initial av initial; order_lt_irreflexive initial)))

inline_for_extraction noextract
let selection_same_range
  (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (lo mid hi : nat) (v : f32) (w : int)
  : Lemma
      (requires selection_best score initial e lo mid v w /\
        (forall (j : selection_ix). selection_inside lo mid j <==> selection_inside lo hi j))
      (ensures selection_best score initial e lo hi v w)
  = ()

inline_for_extraction noextract
let selection_pack
  (score : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (lo hi : nat) (v : f32) (w : int)
  : Lemma
      (requires selection_best score initial (Selection_Local c) lo hi v w)
      (ensures selection_best score initial Selection_Packed lo hi v
        (if w == -1 then -1 else c * 4294967296 + w))
  = if w <> -1 then (
      selection_valid_decode (Selection_Local c) w;
      ML.division_definition (c * 4294967296 + w) 4294967296 c;
      ML.lemma_div_mod (c * 4294967296 + w) 4294967296;
      assert ((c * 4294967296 + w) / 4294967296 == c);
      assert ((c * 4294967296 + w) % 4294967296 == w))
#pop-options


(* Reduction proof and implementation. *)
#push-options "--fuel 2 --ifuel 1"
#set-options "--fuel 0 --ifuel 1"

noextract unfold let reduction_stage (w : visit_width) (a : nat) : prop =
  (a == 1 \/ a == 2 \/ a == 4 \/ a == 8 \/ a == 16 \/ a == 32) /\ a <= w

noextract unfold let reduction_outside
  (#n : nat) (#a : Type0) (before after : lseq a n) (lo hi : nat) : prop =
  forall (j : nat{j < n}).
    (j < lo \/ hi <= j) ==> S.index after j == S.index before j

noextract let reduction_node
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos) (base origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n) (j : nat{base + j < n}) : prop =
  selection_best score initial e
    (origin + visit_rev w j * stride)
    (origin + (visit_rev w j + w / a) * stride)
    (S.index vs (base + j)) (I64.v (S.index is (base + j)))

noextract unfold let reduction_forest
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos) (base : nat{base + w <= n}) (origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n) : prop =
  forall (j : nat{j < a /\ j < w}).
    reduction_node score initial e w a base origin stride vs is j

inline_for_extraction noextract
let reduction_half_bounds (w : visit_width) (a : nat) (l : nat)
  : Lemma
      (requires reduction_stage w a /\ 1 < a /\ l < a / 2)
      (ensures l + a / 2 < a /\ a / 2 > 0 /\ reduction_stage w (a / 2))
  = ()

inline_for_extraction noextract
let reduction_forest_at
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos) (base : nat{base + w <= n}) (origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n) (j : nat{j < a /\ j < w})
  : Lemma
      (requires reduction_forest score initial e w a base origin stride vs is)
      (ensures reduction_node score initial e w a base origin stride vs is j)
  = ()

noextract unfold let reduction_partial
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos{reduction_stage w a /\ 1 < a})
  (base : nat{base + w <= n}) (origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n) (l : nat) : prop =
  forall (j : nat{j < a}).
    reduction_node score initial e w (if j < l then a / 2 else a)
      base origin stride vs is j

#push-options "--fuel 1"
inline_for_extraction noextract
let reduction_open_node
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos) (base origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n) (j : nat{base + j < n})
  : Lemma
      (reduction_node score initial e w a base origin stride vs is j <==>
       selection_best score initial e
         (origin + visit_rev w j * stride)
         (origin + (visit_rev w j + w / a) * stride)
         (S.index vs (base + j)) (I64.v (S.index is (base + j))))
  = ()

inline_for_extraction noextract
let reduction_node_best
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos) (base origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n) (j : nat{base + j < n})
  : Lemma
      (requires reduction_node score initial e w a base origin stride vs is j)
      (ensures selection_best score initial e
        (origin + visit_rev w j * stride)
        (origin + (visit_rev w j + w / a) * stride)
        (S.index vs (base + j)) (I64.v (S.index is (base + j))))
  = ()
#pop-options

#push-options "--z3rlimit 50"
inline_for_extraction noextract
let reduction_root
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (base : nat{base + w <= n}) (origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n)
  : Lemma
      (requires reduction_forest score initial e w 1 base origin stride vs is)
      (ensures selection_best score initial e origin (origin + w * stride)
        (S.index vs base) (I64.v (S.index is base)))
  = visit_rev_zero w;
    reduction_forest_at score initial e w 1 base origin stride vs is 0;
    assert (w / 1 == w);
    assert (origin + visit_rev w 0 * stride == origin);
    assert (origin + (visit_rev w 0 + w / 1) * stride == origin + w * stride);
    reduction_node_best score initial e w 1 base origin stride vs is 0
#pop-options

inline_for_extraction noextract
let reduction_partial_at
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos{reduction_stage w a /\ 1 < a})
  (base : nat{base + w <= n}) (origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n) (l : nat) (j : nat{j < a})
  : Lemma
      (requires reduction_partial score initial e w a base origin stride vs is l)
      (ensures reduction_node score initial e w (if j < l then a / 2 else a)
        base origin stride vs is j)
  = ()

inline_for_extraction noextract
let reduction_node_same
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos) (base origin stride : nat)
  (vs vs' : lseq f32 n) (is is' : lseq i64 n) (j : nat{base + j < n})
  : Lemma
      (requires reduction_node score initial e w a base origin stride vs is j /\
        S.index vs' (base + j) == S.index vs (base + j) /\
        S.index is' (base + j) == S.index is (base + j))
      (ensures reduction_node score initial e w a base origin stride vs' is' j)
  = reduction_open_node score initial e w a base origin stride vs is j;
    reduction_open_node score initial e w a base origin stride vs' is' j

inline_for_extraction noextract
let reduction_lt_next (j l : nat)
  : Lemma (requires j <> l) (ensures (j < l + 1) == (j < l))
  = ()

inline_for_extraction noextract
let reduction_finish_stage
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos{reduction_stage w a /\ 1 < a})
  (base : nat{base + w <= n}) (origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n)
  : Lemma
      (requires reduction_partial score initial e w a base origin stride vs is (a / 2))
      (ensures reduction_forest score initial e w (a / 2) base origin stride vs is)
  = let point (j : nat{j < a / 2 /\ j < w})
      : Lemma (reduction_node score initial e w (a / 2) base origin stride vs is j)
      = reduction_partial_at score initial e w a base origin stride vs is (a / 2) j in
    C.forall_intro point

inline_for_extraction noextract
let reduction_interval_bounds (origin position reduction_step stride : nat)
  : Lemma
      (0 <= origin + position * stride /\
       origin + position * stride <= origin + (position + reduction_step) * stride /\
       origin + (position + reduction_step) * stride <= origin + (position + 2 * reduction_step) * stride)
  = ()

#push-options "--z3rlimit 20"
inline_for_extraction noextract
let reduction_step
  (#n : nat) (score : selection_ix -> f32) (initial : f32) (e : selection_encoding)
  (w : visit_width) (a : pos{reduction_stage w a /\ 1 < a})
  (base : nat{base + w <= n}) (origin stride : nat)
  (vs : lseq f32 n) (is : lseq i64 n) (l : nat{l < a / 2})
  : Lemma
      (requires reduction_partial score initial e w a base origin stride vs is l)
      (ensures (
        let left = base + l in
        let right = left + a / 2 in
        let take = selection_take (S.index vs left) (I64.v (S.index is left))
          (S.index vs right) (I64.v (S.index is right)) in
        let vs' = if take then S.upd vs left (S.index vs right) else vs in
        let is' = if take then S.upd is left (S.index is right) else is in
        reduction_partial score initial e w a base origin stride vs' is' (l + 1)))
  =
    reduction_half_bounds w a l;
    visit_tree_step w a l; visit_rev_bounds w l; visit_rev_bounds w (l + a / 2);
    reduction_partial_at score initial e w a base origin stride vs is l l;
    reduction_partial_at score initial e w a base origin stride vs is l (l + a / 2);
    assert (reduction_node score initial e w a base origin stride vs is l);
    assert (reduction_node score initial e w a base origin stride vs is (l + a / 2));
    reduction_node_best score initial e w a base origin stride vs is l;
    reduction_node_best score initial e w a base origin stride vs is (l + a / 2);
    let left = base + l in
    let right = left + a / 2 in
    let av = S.index vs left in
    let bv = S.index vs right in
    let ai = I64.v (S.index is left) in
    let bi = I64.v (S.index is right) in
    reduction_interval_bounds origin (visit_rev w l) (w / a) stride;
    assert (visit_rev w (l + a / 2) + w / a == visit_rev w l + 2 * (w / a));
    assert (origin + visit_rev w (l + a / 2) * stride ==
      origin + (visit_rev w l + w / a) * stride);
    assert (origin + (visit_rev w (l + a / 2) + w / a) * stride ==
      origin + (visit_rev w l + 2 * (w / a)) * stride);
    selection_merge score initial e
      (origin + visit_rev w l * stride)
      (origin + (visit_rev w l + w / a) * stride)
      (origin + (visit_rev w l + 2 * (w / a)) * stride)
      av bv ai bi;
    let take = selection_take av ai bv bi in
    let vs' = if take then S.upd vs left (S.index vs right) else vs in
    let is' = if take then S.upd is left (S.index is right) else is in
    let point (j : nat{j < a})
      : Lemma
          (reduction_node score initial e w (if j < l + 1 then a / 2 else a)
            base origin stride vs' is' j)
      =
        assert (reduction_node score initial e w (if j < l then a / 2 else a)
          base origin stride vs is j);
        reduction_open_node score initial e w (if j < l then a / 2 else a)
          base origin stride vs is j;
        reduction_open_node score initial e w (if j < l + 1 then a / 2 else a)
          base origin stride vs' is' j;
        if j = l then (
          assert (S.index vs' (base + j) == (if take then bv else av));
          assert (I64.v (S.index is' (base + j)) == (if take then bi else ai)))
        else (
          reduction_lt_next j l;
          assert (S.index vs' (base + j) == S.index vs (base + j));
          assert (S.index is' (base + j) == S.index is (base + j));
          assert ((if j < l + 1 then a / 2 else a) == (if j < l then a / 2 else a));
          reduction_node_same score initial e w (if j < l then a / 2 else a)
            base origin stride vs vs' is is' j) in
    C.forall_intro point
#pop-options

inline_for_extraction noextract
fn reduction_compare_cells
  (#n : nat)
  (values : larray f32 n) (indices : larray i64 n)
  (left right : szlt n)
  (#vs : erased (lseq f32 n)) (#is : erased (lseq i64 n))
  requires values |-> vs ** indices |-> is
  ensures
    values |-> (if selection_take (S.index vs left) (I64.v (S.index is left))
      (S.index vs right) (I64.v (S.index is right))
      then S.upd vs left (S.index vs right) else reveal vs) **
    indices |-> (if selection_take (S.index vs left) (I64.v (S.index is left))
      (S.index vs right) (I64.v (S.index is right))
      then S.upd is left (S.index is right) else reveal is)
{
  let av = values.(left);
  let bv = values.(right);
  let ai = indices.(left);
  let bi = indices.(right);
  if (argmax_better bv bi av ai) {
    values.(left) <- bv;
    indices.(left) <- bi;
  }
}

inline_for_extraction noextract
fn reduction_tree
  (#n : nat)
  (values : larray f32 n) (indices : larray i64 n)
  (width : sz{SZ.v width == 4 \/ SZ.v width == 8 \/ SZ.v width == 32})
  (base : sz{SZ.v base + SZ.v width <= n})
  (initial : f32)
  (#score : erased (selection_ix -> f32))
  (#e : erased selection_encoding)
  (#origin : erased nat) (#stride : erased nat)
  (#v0 : erased (lseq f32 n)) (#i0 : erased (lseq i64 n))
  preserves gpu
  requires values |-> v0 ** indices |-> i0
  requires pure (reduction_forest score initial e width width base origin stride v0 i0)
  ensures exists* (vs : lseq f32 n) (is : lseq i64 n).
    values |-> vs ** indices |-> is **
    pure (reduction_forest score initial e width 1 base origin stride vs is /\
      reduction_outside v0 vs base (base + width) /\ reduction_outside i0 is base (base + width))
{
  let mut active = width;
  while (!active >^ 1sz)
    invariant exists* (a : sz{reduction_stage width (SZ.v a)})
      (vs : lseq f32 n) (is : lseq i64 n).
      active |-> a ** values |-> vs ** indices |-> is **
      pure (reduction_forest score initial e width a base origin stride vs is /\
        reduction_outside v0 vs base (base + width) /\ reduction_outside i0 is base (base + width))
    decreases (SZ.v (!active))
  {
    let a = !active;
    let offset = a /^ 2sz;
    reduction_half_bounds (SZ.v width) (SZ.v a) 0;
    assert (pure (reduction_stage width (SZ.v offset)));
    let mut lane = 0sz;
    while (!lane <^ offset)
      invariant exists* (l : sz{l <= offset})
        (vs : lseq f32 n) (is : lseq i64 n).
        lane |-> l ** values |-> vs ** indices |-> is **
        pure (
          reduction_partial score initial e width a base origin stride vs is l /\
          reduction_outside v0 vs base (base + width) /\ reduction_outside i0 is base (base + width))
      decreases (SZ.v offset - SZ.v (!lane))
    {
      let l = !lane;
      reduction_half_bounds (SZ.v width) (SZ.v a) (SZ.v l);
      with vs is. assert (values |-> vs ** indices |-> is);
      reduction_step score initial e width a base origin stride vs is l;
      let left = base +^ l;
      let right = left +^ offset;
      assert (pure (SZ.v right < n));
      reduction_compare_cells values indices left right;
      lane := !lane +^ 1sz;
    };
    with vs is. assert (values |-> vs ** indices |-> is);
    reduction_finish_stage score initial e width a base origin stride vs is;
    active := !active /^ 2sz;
  }
}
#pop-options


(* Scan proof and implementation. *)
#push-options "--fuel 2 --ifuel 1"
#set-options "--fuel 0 --ifuel 1"

noextract let scan_score
  (#n : nat) (xs : lseq bf16 n) (row_start : nat{row_start + 153600 <= n})
  (j : selection_ix) : f32 =
  Cast.cast_bf16_to_f32 (S.index xs (row_start + j))

noextract unfold let scan_lower (c t : nat) : nat =
  2048 * visit_cr c + 16 * visit_tr t

noextract let scan_scan_inv
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (t : nat{t < 128})
  (i : nat) (v : f32) (w : int) : prop =
  t <= i /\ i <= t + 1664 /\ (i - t) % 128 == 0 /\
  selection_best sc initial (Selection_Local c) (scan_lower c t) (scan_lower c t + (i - t) / 128) v w

inline_for_extraction noextract
let scan_position
  (c : nat{c < 96}) (t : nat{t < 128})
  (i : nat{t <= i /\ i < 1600 /\ c * 1600 + i < 151936 /\ (i - t) % 128 == 0})
  : Lemma
      (visit_key (c * 1600 + i) == scan_lower c t + (i - t) / 128)
  =
    ML.lemma_div_mod (i - t) 128;
    visit_key_singleton c t ((i - t) / 128) (c * 1600 + i)

#push-options "--fuel 1"
inline_for_extraction noextract
let scan_score_at
  (#n : nat) (xs : lseq bf16 n) (row_start : nat{row_start + 153600 <= n})
  (j : selection_ix)
  : Lemma (scan_score xs row_start j == Cast.cast_bf16_to_f32 (S.index xs (row_start + j)))
  = ()

inline_for_extraction noextract
let scan_scan_start
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (t : nat{t < 128})
  : Lemma (scan_scan_inv sc initial c t t initial (-1))
  = visit_cr_bounds c; visit_tr_bounds t;
    selection_empty sc initial (Selection_Local c) (scan_lower c t)

inline_for_extraction noextract
let scan_scan_bounds
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (t : nat{t < 128}) (i : nat) (v : f32) (w : int)
  : Lemma
      (requires scan_scan_inv sc initial c t i v w)
      (ensures t <= i /\ i <= t + 1664 /\ (i - t) % 128 == 0)
  = ()

inline_for_extraction noextract
let scan_scan_step
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (t : nat{t < 128})
  (i : nat{i < 1600 /\ c * 1600 + i < 151936}) (v : f32) (w : int)
  : Lemma
      (requires scan_scan_inv sc initial c t i v w)
      (ensures scan_scan_inv sc initial c t (i + 128)
        (if selection_take v w (sc (c * 1600 + i)) i then sc (c * 1600 + i) else v)
        (if selection_take v w (sc (c * 1600 + i)) i then i else w))
  =
    visit_cr_bounds c; visit_tr_bounds t;
    scan_position c t i;
    visit_key_injective_all (c * 1600 + i);
    selection_scan_step sc initial (Selection_Local c) (scan_lower c t) (c * 1600 + i) v w i;
    ML.lemma_div_plus (i - t) 1 128

inline_for_extraction noextract
let scan_scan_complete
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (t : nat{t < 128})
  (i : nat{1600 <= i \/ 151936 <= c * 1600 + i}) (v : f32) (w : int)
  : Lemma
      (requires scan_scan_inv sc initial c t i v w)
      (ensures selection_best sc initial (Selection_Local c) (scan_lower c t) (scan_lower c t + 16) v w)
  =
    visit_cr_bounds c; visit_tr_bounds t;
    ML.lemma_div_mod (i - t) 128;
    let point (j : selection_ix)
      : Lemma
          (selection_inside (scan_lower c t) (scan_lower c t + (i - t) / 128) j <==>
           selection_inside (scan_lower c t) (scan_lower c t + 16) j)
      =
        if selection_inside (scan_lower c t) (scan_lower c t + 16) j then (
          visit_thread_range c t j;
          ML.lemma_div_mod j 1600;
          ML.lemma_div_mod (j % 1600) 128;
          assert (j % 1600 < i);
          assert ((j % 1600) / 128 < (i - t) / 128)) in
    C.forall_intro point;
    selection_same_range sc initial (Selection_Local c)
      (scan_lower c t) (scan_lower c t + (i - t) / 128) (scan_lower c t + 16) v w
#pop-options

inline_for_extraction noextract
fn scan_thread
  (#n : nat)
  (scores : larray bf16 n)
  (row_start : sz{SZ.v row_start + 153600 <= n /\ n < 0x100000000})
  (chunk : szlt 96) (tid : szlt 128) (initial : f32)
  (#xs : erased (lseq bf16 n)) (#f : perm)
  preserves gpu ** scores |-> Frac f (reveal xs)
  returns result : (f32 & i64)
  ensures pure (selection_best (scan_score #n (reveal xs) row_start) initial (Selection_Local chunk)
    (scan_lower chunk tid) (scan_lower chunk tid + 16) result._1 (I64.v result._2))
{
  visit_cr_bounds chunk; visit_tr_bounds tid;
  scan_scan_start (scan_score #n (reveal xs) row_start) initial chunk tid;
  let mut i : sz = tid;
  let mut value = initial;
  let mut index = -1L;
  let logical_limit = SZ.uint32_to_sizet 151936ul;
  while (!i <^ 1600sz && chunk *^ 1600sz +^ !i <^ logical_limit)
    invariant exists* (at : sz) (v : f32) (w : i64).
      i |-> at ** value |-> v ** index |-> w **
      pure (scan_scan_inv (scan_score #n (reveal xs) row_start) initial chunk tid at v (I64.v w))
    decreases (1728 - SZ.v (!i))
  {
    let at = !i;
    let v = !value;
    let w = !index;
    scan_scan_bounds (scan_score #n (reveal xs) row_start) initial chunk tid at v (I64.v w);
    let j = chunk *^ 1600sz +^ at;
    let raw = scores.(row_start +^ j);
    let next : f32 = Cast.cast_bf16_to_f32 raw;
    scan_score_at #n (reveal xs) row_start j;
    assert (pure (next == scan_score #n (reveal xs) row_start j));
    scan_scan_step (scan_score #n (reveal xs) row_start) initial chunk tid at v (I64.v w);
    let u = SZ.sizet_to_u32 at;
    let at64 = IC.uint32_to_int64 u;
    assert (pure (I64.v at64 == SZ.v at));
    let take = argmax_better next at64 v w;
    value := (if take then next else v);
    index := (if take then at64 else w);
    i := !i +^ 128sz;
  };
  let at = !i;
  let v = !value;
  let w = !index;
  scan_scan_complete (scan_score #n (reveal xs) row_start) initial chunk tid at v (I64.v w);
  (v, w)
}
#pop-options


(* Chunk proof and implementation. *)
#push-options "--fuel 2 --ifuel 1"
#set-options "--fuel 0 --ifuel 1"

noextract let chunk_leaf
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (t : nat{t < 128}) (v : f32) (i : i64) : prop =
  selection_best sc initial (Selection_Local c)
    (scan_lower c t) (scan_lower c t + 16) v (I64.v i)

noextract let chunk_warp_best
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (w : nat{w < 4}) (v : f32) (i : i64) : prop =
  selection_best sc initial (Selection_Local c)
    (2048 * visit_cr c + 512 * visit_r2 w) (2048 * visit_cr c + 512 * visit_r2 w + 512)
    v (I64.v i)

noextract unfold let chunk_scanned
  (sc : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (vs : lseq f32 256) (is : lseq i64 256) (t : nat{t <= 128}) : prop =
  forall (j : nat{j < t}). chunk_leaf sc initial c j (S.index vs j) (S.index is j)

noextract let chunk_warps_state
  (sc : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (vs : lseq f32 256) (is : lseq i64 256) (w : nat{w <= 4}) : prop =
  (forall (j : nat{j < w}).
    chunk_warp_best sc initial c j (S.index vs j) (S.index is j)) /\
  (forall (j : nat{w * 32 <= j /\ j < 128}).
    chunk_leaf sc initial c j (S.index vs j) (S.index is j))

#push-options "--fuel 1"
inline_for_extraction noextract
let chunk_open_warps_state
  (sc : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (vs : lseq f32 256) (is : lseq i64 256) (w : nat{w <= 4})
  : Lemma (chunk_warps_state sc initial c vs is w <==>
    (forall (j : nat{j < w}).
      chunk_warp_best sc initial c j (S.index vs j) (S.index is j)) /\
    (forall (j : nat{w * 32 <= j /\ j < 128}).
      chunk_leaf sc initial c j (S.index vs j) (S.index is j)))
  = ()

inline_for_extraction noextract
let chunk_start_warps
  (sc : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (vs : lseq f32 256) (is : lseq i64 256)
  : Lemma (requires chunk_scanned sc initial c vs is 128)
      (ensures chunk_warps_state sc initial c vs is 0)
  = ()

inline_for_extraction noextract
let chunk_open_leaf
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (t : nat{t < 128}) (v : f32) (i : i64)
  : Lemma (chunk_leaf sc initial c t v i <==>
    selection_best sc initial (Selection_Local c)
      (scan_lower c t) (scan_lower c t + 16) v (I64.v i))
  = ()

inline_for_extraction noextract
let chunk_open_warp
  (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (w : nat{w < 4}) (v : f32) (i : i64)
  : Lemma (chunk_warp_best sc initial c w v i <==>
    selection_best sc initial (Selection_Local c)
      (2048 * visit_cr c + 512 * visit_r2 w) (2048 * visit_cr c + 512 * visit_r2 w + 512)
      v (I64.v i))
  = ()
#pop-options

inline_for_extraction noextract
let chunk_store_scan
  (sc : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (vs : lseq f32 256) (is : lseq i64 256) (t : nat{t < 128})
  (v : f32) (i : i64)
  : Lemma
      (requires chunk_scanned sc initial c vs is t /\ chunk_leaf sc initial c t v i)
      (ensures chunk_scanned sc initial c (S.upd vs t v) (S.upd is t i) (t + 1))
  = let point (j : nat{j < t + 1})
      : Lemma (chunk_leaf sc initial c j
        (S.index (S.upd vs t v) j) (S.index (S.upd is t i) j))
      = if j = t then () else assert (j < t) in
    C.forall_intro point

inline_for_extraction noextract
let chunk_to_warp
  (sc : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (vs : lseq f32 256) (is : lseq i64 256) (w : nat{w < 4})
  : Lemma
      (requires chunk_warps_state sc initial c vs is w)
      (ensures reduction_forest sc initial (Selection_Local c) 32 32 (32 * w)
        (2048 * visit_cr c + 512 * visit_r2 w) 16 vs is)
  =
    chunk_open_warps_state sc initial c vs is w;
    visit_cr_bounds c; visit_rev4 w; visit_rev_bounds 4 w;
    let point (j : nat{j < 32})
      : Lemma (reduction_node sc initial (Selection_Local c) 32 32 (32 * w)
        (2048 * visit_cr c + 512 * visit_r2 w) 16 vs is j)
      =
        visit_tr_warp w j; visit_rev32 j; visit_rev_bounds 32 j;
        chunk_open_leaf sc initial c (32 * w + j)
          (S.index vs (32 * w + j)) (S.index is (32 * w + j));
        reduction_open_node sc initial (Selection_Local c) 32 32 (32 * w)
          (2048 * visit_cr c + 512 * visit_r2 w) 16 vs is j in
    C.forall_intro point

#push-options "--z3rlimit 20"
inline_for_extraction noextract
let chunk_compress_warp
  (sc : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (vs vs' : lseq f32 256) (is is' : lseq i64 256) (w : nat{w < 4})
  : Lemma
      (requires chunk_warps_state sc initial c vs is w /\
        reduction_forest sc initial (Selection_Local c) 32 1 (32 * w)
          (2048 * visit_cr c + 512 * visit_r2 w) 16 vs' is' /\
        reduction_outside vs vs' (32 * w) (32 * w + 32) /\
        reduction_outside is is' (32 * w) (32 * w + 32))
      (ensures chunk_warps_state sc initial c
        (S.upd vs' w (S.index vs' (32 * w)))
        (S.upd is' w (S.index is' (32 * w))) (w + 1))
  =
    chunk_open_warps_state sc initial c vs is w;
    chunk_open_warps_state sc initial c
      (S.upd vs' w (S.index vs' (32 * w)))
      (S.upd is' w (S.index is' (32 * w))) (w + 1);
    visit_cr_bounds c; visit_rev4 w; visit_rev_bounds 4 w; visit_rev_zero 32;
    reduction_root sc initial (Selection_Local c) 32 (32 * w)
      (2048 * visit_cr c + 512 * visit_r2 w) 16 vs' is';
    chunk_open_warp sc initial c w (S.index vs' (32 * w)) (S.index is' (32 * w));
    assert (chunk_warp_best sc initial c w (S.index vs' (32 * w)) (S.index is' (32 * w)));
    let prefix (j : nat{j < w + 1})
      : Lemma (chunk_warp_best sc initial c j
        (S.index (S.upd vs' w (S.index vs' (32 * w))) j)
        (S.index (S.upd is' w (S.index is' (32 * w))) j))
      = if j = w then (
          assert (S.index (S.upd vs' w (S.index vs' (32 * w))) j == S.index vs' (32 * w));
          assert (S.index (S.upd is' w (S.index is' (32 * w))) j == S.index is' (32 * w)))
        else (
          reduction_lt_next j w;
          assert (j < w /\ j < 32 * w);
          assert (chunk_warp_best sc initial c j (S.index vs j) (S.index is j));
          assert (S.index vs' j == S.index vs j);
          assert (S.index is' j == S.index is j)) in
    C.forall_intro prefix;
    let suffix (j : nat{(w + 1) * 32 <= j /\ j < 128})
      : Lemma (chunk_leaf sc initial c j
        (S.index (S.upd vs' w (S.index vs' (32 * w))) j)
        (S.index (S.upd is' w (S.index is' (32 * w))) j))
      = assert (j <> w /\ 32 * w + 32 <= j);
        assert (S.index vs' j == S.index vs j);
        assert (S.index is' j == S.index is j) in
    C.forall_intro suffix
#pop-options

inline_for_extraction noextract
let chunk_to_cross_warp
  (sc : selection_ix -> f32) (initial : f32) (c : nat{c < 96})
  (vs : lseq f32 256) (is : lseq i64 256)
  : Lemma
      (requires chunk_warps_state sc initial c vs is 4)
      (ensures reduction_forest sc initial (Selection_Local c) 4 4 0 (2048 * visit_cr c) 512 vs is)
  =
    chunk_open_warps_state sc initial c vs is 4;
    visit_cr_bounds c;
    let point (j : nat{j < 4})
      : Lemma (reduction_node sc initial (Selection_Local c) 4 4 0 (2048 * visit_cr c) 512 vs is j)
      = visit_rev4 j; visit_rev_bounds 4 j;
        chunk_open_warp sc initial c j (S.index vs j) (S.index is j);
        reduction_open_node sc initial (Selection_Local c) 4 4 0 (2048 * visit_cr c) 512 vs is j in
    C.forall_intro point

inline_for_extraction noextract
fn chunk_run
  (#n : nat)
  (scores : larray bf16 n)
  (row_start : sz{SZ.v row_start + 153600 <= n /\ n < 0x100000000})
  (chunk : szlt 96) (initial : f32)
  (#xs : erased (lseq bf16 n)) (#f : perm)
  preserves gpu ** scores |-> Frac f (reveal xs)
  returns result : (f32 & i64)
  ensures pure (selection_best (scan_score #n (reveal xs) row_start) initial (Selection_Local chunk)
    (2048 * visit_cr chunk) (2048 * visit_cr chunk + 2048) result._1 (I64.v result._2))
{
  let mut values = [| initial; 256sz |];
  let mut indices = [| -1L; 256sz |];
  let mut tid = 0sz;
  while (!tid <^ 128sz)
    invariant exists* (t : sz{t <= 128})
      (vs : lseq f32 256) (is : lseq i64 256).
      tid |-> t ** values |-> vs ** indices |-> is **
      pure (chunk_scanned (scan_score #n (reveal xs) row_start) initial chunk vs is t)
    decreases (128 - SZ.v (!tid))
  {
    let t = !tid;
    let result = scan_thread scores row_start chunk t initial #xs #f;
    chunk_open_leaf (scan_score #n (reveal xs) row_start) initial chunk t result._1 result._2;
    with vs is. assert (values |-> vs ** indices |-> is);
    chunk_store_scan (scan_score #n (reveal xs) row_start) initial chunk vs is t result._1 result._2;
    values.(t) <- result._1;
    indices.(t) <- result._2;
    tid := !tid +^ 1sz;
  };
  with vs is. assert (values |-> vs ** indices |-> is);
  chunk_start_warps (scan_score #n (reveal xs) row_start) initial chunk vs is;
  let mut warp = 0sz;
  while (!warp <^ 4sz)
    invariant exists* (w : sz{w <= 4})
      (vs : lseq f32 256) (is : lseq i64 256).
      warp |-> w ** values |-> vs ** indices |-> is **
      pure (chunk_warps_state (scan_score #n (reveal xs) row_start) initial chunk vs is w)
    decreases (4 - SZ.v (!warp))
  {
    let w = !warp;
    with vs is. assert (values |-> vs ** indices |-> is);
    chunk_to_warp (scan_score #n (reveal xs) row_start) initial chunk vs is w;
    reduction_tree values indices 32sz (w *^ 32sz) initial
      #(hide (scan_score #n (reveal xs) row_start)) #(hide (Selection_Local chunk))
      #(hide (2048 * visit_cr chunk + 512 * visit_r2 w)) #(hide 16);
    with vs' is'. assert (values |-> vs' ** indices |-> is');
    chunk_compress_warp (scan_score #n (reveal xs) row_start) initial chunk vs vs' is is' w;
    let v = values.(w *^ 32sz);
    let i = indices.(w *^ 32sz);
    values.(w) <- v;
    indices.(w) <- i;
    warp := !warp +^ 1sz;
  };
  with vs is. assert (values |-> vs ** indices |-> is);
  chunk_to_cross_warp (scan_score #n (reveal xs) row_start) initial chunk vs is;
  reduction_tree values indices 4sz 0sz initial
    #(hide (scan_score #n (reveal xs) row_start)) #(hide (Selection_Local chunk))
    #(hide (2048 * visit_cr chunk)) #(hide 512);
  with final_values final_indices.
    assert (values |-> final_values ** indices |-> final_indices);
  reduction_root (scan_score #n (reveal xs) row_start) initial (Selection_Local chunk)
    4 0 (2048 * visit_cr chunk) 512 final_values final_indices;
  let v = values.(0sz);
  let i = indices.(0sz);
  (v, i)
}
#pop-options


(* Select proof and implementation. *)
#push-options "--fuel 2 --ifuel 1"
#set-options "--fuel 0 --ifuel 1"

noextract let select_chunk_best (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (v : f32) (i : i64) : prop =
  selection_best sc initial (Selection_Local c) (2048 * visit_cr c) (2048 * visit_cr c + 2048) v (I64.v i)

noextract unfold let select_chunks (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 96) (is : lseq i64 96) (count : nat{count <= 96}) : prop =
  forall (c : nat{c < count}).
    select_chunk_best sc initial c (S.index vs c) (S.index is c)

noextract let select_leaf (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 256}) (v : f32) (i : i64) : prop =
  selection_best sc initial Selection_Packed (2048 * visit_cr c) (2048 * visit_cr c + 2048) v (I64.v i)

noextract let select_warp_best (sc : selection_ix -> f32) (initial : f32)
  (w : nat{w < 8}) (v : f32) (i : i64) : prop =
  selection_best sc initial Selection_Packed (65536 * visit_r3 w) (65536 * visit_r3 w + 65536) v (I64.v i)

noextract unfold let select_leaves (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 256) (is : lseq i64 256) (count : nat{count <= 256}) : prop =
  forall (c : nat{c < count}). select_leaf sc initial c (S.index vs c) (S.index is c)

noextract let select_warps_state (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 256) (is : lseq i64 256) (w : nat{w <= 8}) : prop =
  (forall (j : nat{j < w}). select_warp_best sc initial j (S.index vs j) (S.index is j)) /\
  (forall (j : nat{w * 32 <= j /\ j < 256}).
    select_leaf sc initial j (S.index vs j) (S.index is j))

#push-options "--fuel 1"
inline_for_extraction noextract
let select_open_chunk (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (v : f32) (i : i64)
  : Lemma (select_chunk_best sc initial c v i <==>
    selection_best sc initial (Selection_Local c) (2048 * visit_cr c) (2048 * visit_cr c + 2048) v (I64.v i))
  = ()

inline_for_extraction noextract
let select_open_leaf (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 256}) (v : f32) (i : i64)
  : Lemma (select_leaf sc initial c v i <==>
    selection_best sc initial Selection_Packed (2048 * visit_cr c) (2048 * visit_cr c + 2048) v (I64.v i))
  = ()

inline_for_extraction noextract
let select_open_warp (sc : selection_ix -> f32) (initial : f32)
  (w : nat{w < 8}) (v : f32) (i : i64)
  : Lemma (select_warp_best sc initial w v i <==>
    selection_best sc initial Selection_Packed (65536 * visit_r3 w) (65536 * visit_r3 w + 65536) v (I64.v i))
  = ()

inline_for_extraction noextract
let select_open_state (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 256) (is : lseq i64 256) (w : nat{w <= 8})
  : Lemma (select_warps_state sc initial vs is w <==>
    (forall (j : nat{j < w}). select_warp_best sc initial j (S.index vs j) (S.index is j)) /\
    (forall (j : nat{w * 32 <= j /\ j < 256}).
      select_leaf sc initial j (S.index vs j) (S.index is j)))
  = ()

inline_for_extraction noextract
let select_start_warps (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 256) (is : lseq i64 256)
  : Lemma (requires select_leaves sc initial vs is 256) (ensures select_warps_state sc initial vs is 0)
  = ()
#pop-options

inline_for_extraction noextract
let select_store_chunk (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 96) (is : lseq i64 96) (c : nat{c < 96}) (v : f32) (i : i64)
  : Lemma
      (requires select_chunks sc initial vs is c /\ select_chunk_best sc initial c v i)
      (ensures select_chunks sc initial (S.upd vs c v) (S.upd is c i) (c + 1))
  = let point (j : nat{j < c + 1}) : Lemma (select_chunk_best sc initial j
        (S.index (S.upd vs c v) j) (S.index (S.upd is c i) j))
      = if j = c then () else reduction_lt_next j c in
    C.forall_intro point

inline_for_extraction noextract
let select_store_leaf (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 256) (is : lseq i64 256) (c : nat{c < 256}) (v : f32) (i : i64)
  : Lemma
      (requires select_leaves sc initial vs is c /\ select_leaf sc initial c v i)
      (ensures select_leaves sc initial (S.upd vs c v) (S.upd is c i) (c + 1))
  = let point (j : nat{j < c + 1}) : Lemma (select_leaf sc initial j
        (S.index (S.upd vs c v) j) (S.index (S.upd is c i) j))
      = if j = c then () else reduction_lt_next j c in
    C.forall_intro point

inline_for_extraction noextract
let select_missing_chunk (sc : selection_ix -> f32) (initial : f32) (c : nat{96 <= c /\ c < 256})
  : Lemma (select_leaf sc initial c initial (-1L))
  =
    visit_cr_bounds c;
    let point (j : selection_ix) : Lemma
      (not (selection_inside (2048 * visit_cr c) (2048 * visit_cr c + 2048) j))
      = visit_absent_chunk c j in
    C.forall_intro point;
    selection_absent sc initial Selection_Packed
      (2048 * visit_cr c) (2048 * visit_cr c + 2048);
    select_open_leaf sc initial c initial (-1L)

inline_for_extraction noextract
let select_candidate_test (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (v : f32) (i : i64)
  : Lemma
      (requires select_chunk_best sc initial c v i)
      (ensures (lt initial v <==> I64.v i <> -1) /\
        (I64.v i == -1 ==> v == initial) /\
        (I64.v i <> -1 ==> 0 <= I64.v i /\ I64.v i < 1600))
  = select_open_chunk sc initial c v i;
    selection_best_valid sc initial (Selection_Local c)
      (2048 * visit_cr c) (2048 * visit_cr c + 2048) v (I64.v i);
    order_lt_irreflexive initial

inline_for_extraction noextract
let select_packed_chunk (sc : selection_ix -> f32) (initial : f32)
  (c : nat{c < 96}) (v : f32) (i packed : i64)
  : Lemma
      (requires select_chunk_best sc initial c v i /\
        I64.v packed == (if I64.v i = -1 then -1 else c * 4294967296 + I64.v i))
      (ensures select_leaf sc initial c v packed)
  = visit_cr_bounds c;
    select_open_chunk sc initial c v i;
    selection_pack sc initial c (2048 * visit_cr c) (2048 * visit_cr c + 2048) v (I64.v i);
    select_open_leaf sc initial c v packed

#push-options "--fuel 2"
inline_for_extraction noextract
let pack_local_index (c : szlt 96) (i : i64{0 <= I64.v i /\ I64.v i < 1600})
  : packed:i64{I64.v packed == SZ.v c * 4294967296 + I64.v i} =
  let c64 = IC.uint32_to_int64 (SZ.sizet_to_u32 c) in
  I64.add (I64.mul c64 4294967296L) i
#pop-options

inline_for_extraction noextract
fn select_initialize_leaf
  (c : szlt 256) (initial : f32)
  (chunk_values : larray f32 96) (chunk_indices : larray i64 96)
  (#sc : erased (selection_ix -> f32))
  (#vs : erased (lseq f32 96)) (#is : erased (lseq i64 96))
  preserves chunk_values |-> vs ** chunk_indices |-> is
  requires pure (select_chunks sc initial vs is 96)
  returns result : (f32 & i64)
  ensures pure (select_leaf sc initial c result._1 result._2)
{
  if (c <^ 96sz) {
    let v = chunk_values.(c);
    let i = chunk_indices.(c);
    select_candidate_test sc initial c v i;
    if (lt initial v) {
      let packed = pack_local_index c i;
      select_packed_chunk sc initial c v i packed;
      (v, packed)
    } else {
      select_packed_chunk sc initial c v i (-1L);
      (initial, -1L)
    }
  } else {
    select_missing_chunk sc initial c;
    (initial, -1L)
  }
}

inline_for_extraction noextract
let select_to_warp (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 256) (is : lseq i64 256) (w : nat{w < 8})
  : Lemma
      (requires select_warps_state sc initial vs is w)
      (ensures reduction_forest sc initial Selection_Packed 32 32 (32 * w) (65536 * visit_r3 w) 2048 vs is)
  =
    select_open_state sc initial vs is w;
    visit_rev8 w; visit_rev_bounds 8 w;
    let point (j : nat{j < 32}) : Lemma
      (reduction_node sc initial Selection_Packed 32 32 (32 * w) (65536 * visit_r3 w) 2048 vs is j)
      = visit_cr_warp w j; visit_rev32 j; visit_rev_bounds 32 j;
        select_open_leaf sc initial (32 * w + j) (S.index vs (32 * w + j)) (S.index is (32 * w + j));
        reduction_open_node sc initial Selection_Packed 32 32 (32 * w) (65536 * visit_r3 w) 2048 vs is j in
    C.forall_intro point

inline_for_extraction noextract
let select_compress_warp (sc : selection_ix -> f32) (initial : f32)
  (vs vs' : lseq f32 256) (is is' : lseq i64 256) (w : nat{w < 8})
  : Lemma
      (requires select_warps_state sc initial vs is w /\
        reduction_forest sc initial Selection_Packed 32 1 (32 * w) (65536 * visit_r3 w) 2048 vs' is' /\
        reduction_outside vs vs' (32 * w) (32 * w + 32) /\
        reduction_outside is is' (32 * w) (32 * w + 32))
      (ensures select_warps_state sc initial
        (S.upd vs' w (S.index vs' (32 * w))) (S.upd is' w (S.index is' (32 * w))) (w + 1))
  =
    select_open_state sc initial vs is w;
    select_open_state sc initial
      (S.upd vs' w (S.index vs' (32 * w))) (S.upd is' w (S.index is' (32 * w))) (w + 1);
    visit_rev8 w; visit_rev_bounds 8 w;
    reduction_root sc initial Selection_Packed 32 (32 * w) (65536 * visit_r3 w) 2048 vs' is';
    select_open_warp sc initial w (S.index vs' (32 * w)) (S.index is' (32 * w));
    assert (select_warp_best sc initial w (S.index vs' (32 * w)) (S.index is' (32 * w)));
    let prefix (j : nat{j < w + 1}) : Lemma (select_warp_best sc initial j
        (S.index (S.upd vs' w (S.index vs' (32 * w))) j)
        (S.index (S.upd is' w (S.index is' (32 * w))) j))
      = if j = w then () else (
          reduction_lt_next j w;
          assert (j < w /\ j < 32 * w);
          assert (select_warp_best sc initial j (S.index vs j) (S.index is j));
          assert (S.index vs' j == S.index vs j);
          assert (S.index is' j == S.index is j)) in
    C.forall_intro prefix;
    let suffix (j : nat{(w + 1) * 32 <= j /\ j < 256}) : Lemma (select_leaf sc initial j
        (S.index (S.upd vs' w (S.index vs' (32 * w))) j)
        (S.index (S.upd is' w (S.index is' (32 * w))) j))
      = assert (j <> w /\ 32 * w + 32 <= j);
        assert (select_leaf sc initial j (S.index vs j) (S.index is j));
        assert (S.index vs' j == S.index vs j);
        assert (S.index is' j == S.index is j);
        assert (S.index (S.upd vs' w (S.index vs' (32 * w))) j == S.index vs j);
        assert (S.index (S.upd is' w (S.index is' (32 * w))) j == S.index is j) in
    C.forall_intro suffix

inline_for_extraction noextract
let select_to_cross_warp (sc : selection_ix -> f32) (initial : f32)
  (vs : lseq f32 256) (is : lseq i64 256)
  : Lemma
      (requires select_warps_state sc initial vs is 8)
      (ensures reduction_forest sc initial Selection_Packed 8 8 0 0 65536 vs is)
  =
    select_open_state sc initial vs is 8;
    let point (j : nat{j < 8}) : Lemma (reduction_node sc initial Selection_Packed 8 8 0 0 65536 vs is j)
      = visit_rev8 j; visit_rev_bounds 8 j;
        select_open_warp sc initial j (S.index vs j) (S.index is j);
        reduction_open_node sc initial Selection_Packed 8 8 0 0 65536 vs is j in
    C.forall_intro point

inline_for_extraction noextract
fn select_run
  (#n : nat) (scores : larray bf16 n)
  (row_start : sz{SZ.v row_start + 153600 <= n /\ n < 0x100000000})
  (initial : f32)
  (#xs : erased (lseq bf16 n)) (#f : perm)
  preserves gpu ** scores |-> Frac f (reveal xs)
  returns result : (f32 & i64)
  ensures pure (selection_best (scan_score #n (reveal xs) row_start) initial Selection_Packed
    0 524288 result._1 (I64.v result._2))
{
  let mut chunk_values = [| initial; 96sz |];
  let mut chunk_indices = [| -1L; 96sz |];
  let mut c = 0sz;
  while (!c <^ 96sz)
    invariant exists* (at : sz{at <= 96}) (vs : lseq f32 96) (is : lseq i64 96).
      c |-> at ** chunk_values |-> vs ** chunk_indices |-> is **
      pure (select_chunks (scan_score #n (reveal xs) row_start) initial vs is at)
    decreases (96 - SZ.v (!c))
  {
    let at = !c;
    let result = chunk_run scores row_start at initial #xs #f;
    select_open_chunk (scan_score #n (reveal xs) row_start) initial at result._1 result._2;
    with vs is. assert (chunk_values |-> vs ** chunk_indices |-> is);
    select_store_chunk (scan_score #n (reveal xs) row_start) initial vs is at result._1 result._2;
    chunk_values.(at) <- result._1;
    chunk_indices.(at) <- result._2;
    c := !c +^ 1sz;
  };
  let mut values = [| initial; 256sz |];
  let mut indices = [| -1L; 256sz |];
  let mut tid = 0sz;
  while (!tid <^ 256sz)
    invariant exists* (at : sz{at <= 256}) (vs : lseq f32 256) (is : lseq i64 256).
      tid |-> at ** values |-> vs ** indices |-> is **
      pure (select_leaves (scan_score #n (reveal xs) row_start) initial vs is at)
    decreases (256 - SZ.v (!tid))
  {
    let at = !tid;
    let result = select_initialize_leaf at initial chunk_values chunk_indices
      #(hide (scan_score #n (reveal xs) row_start));
    with vs is. assert (values |-> vs ** indices |-> is);
    select_store_leaf (scan_score #n (reveal xs) row_start) initial vs is at result._1 result._2;
    values.(at) <- result._1;
    indices.(at) <- result._2;
    tid := !tid +^ 1sz;
  };
  with vs is. assert (values |-> vs ** indices |-> is);
  select_start_warps (scan_score #n (reveal xs) row_start) initial vs is;
  let mut warp = 0sz;
  while (!warp <^ 8sz)
    invariant exists* (w : sz{w <= 8}) (vs : lseq f32 256) (is : lseq i64 256).
      warp |-> w ** values |-> vs ** indices |-> is **
      pure (select_warps_state (scan_score #n (reveal xs) row_start) initial vs is w)
    decreases (8 - SZ.v (!warp))
  {
    let w = !warp;
    with vs is. assert (values |-> vs ** indices |-> is);
    select_to_warp (scan_score #n (reveal xs) row_start) initial vs is w;
    reduction_tree values indices 32sz (w *^ 32sz) initial
      #(hide (scan_score #n (reveal xs) row_start)) #(hide Selection_Packed)
      #(hide (65536 * visit_r3 w)) #(hide 2048);
    with vs' is'. assert (values |-> vs' ** indices |-> is');
    select_compress_warp (scan_score #n (reveal xs) row_start) initial vs vs' is is' w;
    let v = values.(w *^ 32sz);
    let i = indices.(w *^ 32sz);
    values.(w) <- v;
    indices.(w) <- i;
    warp := !warp +^ 1sz;
  };
  with vs is. assert (values |-> vs ** indices |-> is);
  select_to_cross_warp (scan_score #n (reveal xs) row_start) initial vs is;
  reduction_tree values indices 8sz 0sz initial
    #(hide (scan_score #n (reveal xs) row_start)) #(hide Selection_Packed) #(hide 0) #(hide 65536);
  with vs is. assert (values |-> vs ** indices |-> is);
  reduction_root (scan_score #n (reveal xs) row_start) initial Selection_Packed 8 0 0 65536 vs is;
  let v = values.(0sz);
  let i = indices.(0sz);
  (v, i)
}
#pop-options


(* Functional proof and implementation. *)
#push-options "--fuel 2 --ifuel 1"
#set-options "--fuel 1 --ifuel 1"

inline_for_extraction noextract
let score_same
  (xs : physical_logits) (row : row_index) (j : token_index)
  : Lemma (score_at xs row j == scan_score xs (row * physical_width) j)
  = scan_score_at xs (row * physical_width) j

noextract
let row_selection_result (xs : physical_logits) (row : row_index) (winner : i64)
  : GTot prop =
  native_winner (row_scores xs row) (I64.v winner)

inline_for_extraction noextract
let finish
  (xs : physical_logits) (row : row_index)
  (initial v : f32) (packed winner : i64)
  : Lemma
      (requires initial == negative_infinity /\
        selection_best (scan_score xs (row * physical_width)) initial
          Selection_Packed 0 524288 v (I64.v packed) /\
        I64.v winner == (if I64.v packed = -1 then -1 else selection_decode Selection_Packed (I64.v packed)))
      (ensures row_selection_result xs row winner)
  =
    if I64.v packed = -1 then (
      let excluded (j : token_index)
        : Lemma (not (lt negative_infinity (score_at xs row j))) =
        visit_key_digits j;
        score_same xs row j in
      C.forall_intro excluded;
      S.init_index vocab_size (score_at xs row))
    else (
      selection_valid_decode Selection_Packed (I64.v packed);
      let w : token_index = I64.v winner in
      score_same xs row w;
      let point (j : token_index)
        : Lemma
            (lt negative_infinity (score_at xs row j) ==>
             (lt (score_at xs row j) (score_at xs row w) \/
              (F.ieee_eq (score_at xs row j) (score_at xs row w) /\
               visit_key w <= visit_key j)))
        =
          visit_key_digits j;
          score_same xs row j;
          if lt negative_infinity (score_at xs row j) then
            assert (selection_dominates (score_at xs row w) (visit_key w)
              (score_at xs row j) (visit_key j)) in
      C.forall_intro point;
      S.init_index vocab_size (score_at xs row))

#push-options "--fuel 2"
inline_for_extraction noextract
let row_offset (row : i64{0 <= I64.v row /\ I64.v row < prompt_length})
  : x:sz{SZ.v x == I64.v row * physical_width}
  =
    let offset = I64.mul row 153600L in
    let offset32 = IC.int64_to_uint32 offset in
    SZ.uint32_to_sizet offset32

inline_for_extraction noextract
fn unpack (packed : i64{selection_valid Selection_Packed (I64.v packed)})
  requires emp
  returns w : i64
  ensures pure (I64.v w ==
    (if I64.v packed = -1 then -1 else selection_decode Selection_Packed (I64.v packed)))
{
  if (I64.eq packed (-1L)) {
    -1L
  } else {
    let chunk = I64.div packed 4294967296L;
    FStar.Math.Lemmas.lemma_div_mod (I64.v packed) 4294967296;
    let local = I64.sub packed (I64.mul chunk 4294967296L);
    I64.add (I64.mul chunk 1600L) local
  }
}
#pop-options

inline_for_extraction noextract
let copy_step
  (ps : physical_prompt) (ys : physical_output) (i : nat{i < 7})
  : Lemma
      (requires forall (j : nat{j < i}). S.index ys j == S.index ps j)
      (ensures forall (j : nat{j < i + 1}).
        S.index (S.upd ys i (S.index ps i)) j == S.index ps j)
  = let point (j : nat{j < i + 1})
      : Lemma (S.index (S.upd ys i (S.index ps i)) j == S.index ps j)
      = if j = i then () else reduction_lt_next j i in
    C.forall_intro point

noextract
let append_selected
  (xs : physical_logits) (ps : physical_prompt) (row : row_index)
  (copied : physical_output) (winner : i64)
  : Lemma
      (requires row_selection_result xs row winner /\
        (forall (j:nat{j < prompt_length}). S.index copied j == S.index ps j))
      (ensures postcondition xs ps row (S.upd copied prompt_length winner))
  =
    establish_postcondition xs ps row (S.upd copied prompt_length winner)

#push-options "--fuel 0"
inline_for_extraction noextract
fn kernel
  (scores : larray bf16 logits_size)
  (prompt : larray i64 prompt_length)
  (row : i64{0 <= I64.v row /\ I64.v row < prompt_length})
  (tokens : larray i64 sequence_length)
  (#xs : erased physical_logits)
  (#ps : erased physical_prompt)
  (#before : erased (seq i64))
  preserves gpu ** scores |-> xs ** prompt |-> ps
  requires tokens |-> (reveal before)
  ensures exists* (ys : physical_output).
    tokens |-> ys **
    pure (postcondition xs ps (I64.v row) ys)
{
  pts_to_len tokens;
  assert (pure (S.length before == sequence_length));
  let initial : f32 = negative_infinity;
  let offset = row_offset row;
  let result = select_run scores offset initial #xs;
  selection_best_valid (scan_score (reveal xs) offset) initial Selection_Packed
    0 524288 result._1 (I64.v result._2);
  let winner = unpack result._2;
  finish xs (I64.v row) initial result._1 result._2 winner;
  let mut i = 0sz;
  while (!i <^ 7sz)
    invariant exists* (at : sz{at <= 7}) (ys : physical_output).
      i |-> at ** tokens |-> ys **
      pure (forall (j : nat{j < at}). S.index ys j == S.index (reveal ps) j)
    decreases (7 - SZ.v (!i))
  {
    let at = !i;
    with ys. assert (tokens |-> ys);
    copy_step ps ys at;
    let id = prompt.(at);
    tokens.(at) <- id;
    i := !i +^ 1sz;
  };
  with copied. assert (tokens |-> copied);
  append_selected xs ps (I64.v row) copied winner;
  tokens.(7sz) <- winner;
}
#pop-options

inline_for_extraction noextract
instance send_global_array_contents
  (#a : Type0) (arr : array a{is_global_array arr}) (#f : perm) (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (logits : larray bf16 logits_size{is_global_array logits})
  (prefix : larray i64 prompt_length{is_global_array prefix})
  (score_row : i64{0 <= I64.v score_row /\ I64.v score_row < prompt_length})
  (output : larray i64 sequence_length{is_global_array output})
  (#input_logits : erased physical_logits)
  (#input_prefix : erased physical_prompt)
  preserves cpu **
    on gpu_loc (logits |-> input_logits) **
    on gpu_loc (prefix |-> input_prefix)
  requires exists* old. on gpu_loc (output |-> old)
  ensures exists* (result : physical_output).
    on gpu_loc (output |-> result) **
    pure (postcondition input_logits input_prefix (I64.v score_row) result)
{
  with old. assert (on gpu_loc (output |-> old));
  launch_kernel_1 (fun () ->
    kernel logits prefix score_row output #input_logits #input_prefix #old);
}
#pop-options
