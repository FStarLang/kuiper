module Kuiper.Bitmask

open Kuiper
module U32 = FStar.UInt32
module SZ = Kuiper.SizeT
module UI = FStar.UInt
#lang-pulse

(* No recursive definition needs unfolding here; with fuel, Z3 eagerly unrolls
   FStar.UInt.to_vec, which makes many queries diverge. *)
#set-options "--fuel 0 --ifuel 1"

let getBit (u : u32) (i : szlt 32sz) : bool
= U32.shift_right u (SZ.sizet_to_u32 i) `U32.logand` 1ul = 1ul

let setBit (u : u32) (i : szlt 32sz) : u32
= u `U32.logor` U32.shift_left 1ul (SZ.sizet_to_u32 i)

let unsetBit (u : u32) (i : szlt 32sz) : u32
= u `U32.logand` U32.lognot (U32.shift_left 1ul (SZ.sizet_to_u32 i))

let lemma_nth_1 ()
: Lemma (requires true) (ensures forall (i : natlt 31). UI.nth #32 1 i == false)
=
  introduce forall (i : natlt 31). UI.nth #32 1 i == false
  with UI.one_to_vec_lemma #32 i

let shift_bit_lemma_true (u : UI.uint_t 32) (i : natlt 32)
: Lemma
    (requires true)
    (ensures UI.nth #32 (UI.shift_right #32 u i `UI.logand` 1) 31 == UI.nth #32 u (31 - i))
=
  UI.shift_right_lemma_2 #32 u i 31;
  UI.one_to_vec_lemma #32 31;
  UI.logand_definition (UI.shift_right #32 u i) 1 31

let shift_bit_lemma_false (u : UI.uint_t 32)
: Lemma
    (requires true)
    (ensures forall (i : natlt 31). UI.nth #32 (u `UI.logand` 1) i == false)
=
  lemma_nth_1 ()

let cmp_1_lemma (u : UI.uint_t 32)
: Lemma
    (requires forall (i : natlt 31). UI.nth u i == false)
    (ensures UI.nth #32 u 31 == (u = 1))
=
  UI.one_to_vec_lemma #32 31;
  if UI.nth #32 u 31
    then (
      lemma_nth_1 ();
      UI.nth_lemma u 1
    )
    else ()

let getBit_lemma (u : u32) (i : szlt 32)
: Lemma
    (requires true)
    (ensures getBit u i == UI.nth #32 u (31 - i))
=
  shift_bit_lemma_true u i;
  shift_bit_lemma_false (UI.shift_right #32 u i);
  cmp_1_lemma (UI.shift_right #32 u i `UI.logand` 1);
  ()

let shift_left_1_lemma_false (i j : szlt 32)
: Lemma
    (requires SZ.v j <> 31 - i)
    (ensures UI.nth #32 (UI.shift_left 1 i) j == false)
=
  if (j >= 32 - i)
    then ()
    else lemma_nth_1 ()

let setBit_lemma_ensures (u : u32) (i : szlt 32)
: Lemma
    (requires true)
    (ensures UI.nth #32 (setBit u i) (31 - i))
=
  let s = U32.shift_left 1ul (SZ.sizet_to_u32 i) in
  assert (U32.v s == UI.shift_left #32 1 (SZ.v i));
  assert (U32.v (setBit u i) == UI.logor #32 (U32.v u) (U32.v s));
  UI.shift_left_lemma_2 #32 1 i (31 - i);
  UI.one_to_vec_lemma #32 31;
  assert (UI.nth #32 (U32.v s) (31 - i));
  UI.logor_definition #32 u (U32.v s) (31 - i)


let setBit_lemma_preserves (u : u32) (i j : szlt 32)
: Lemma
    (requires j <> i)
    (ensures UI.nth #32 (setBit u i) (31 - j) = UI.nth #32 u (31 - j))
=
  shift_left_1_lemma_false i (31sz -^ j)

let unsetBit_lemma_ensures (u : u32) (i : szlt 32)
: Lemma
    (requires true)
    (ensures UI.nth #32 (unsetBit u i) (31 - i) == false)
=
  let s = U32.shift_left 1ul (SZ.sizet_to_u32 i) in
  assert (U32.v s == UI.shift_left #32 1 (SZ.v i));
  assert (U32.v (U32.lognot s) == UI.lognot #32 (U32.v s));
  assert (U32.v (unsetBit u i) == UI.logand #32 (U32.v u) (UI.lognot #32 (U32.v s)));
  UI.shift_left_lemma_2 #32 1 i (31 - i);
  UI.one_to_vec_lemma #32 31;
  assert (UI.nth #32 (U32.v s) (31 - i));
  UI.lognot_definition #32 (U32.v s) (31 - i);
  UI.logand_definition #32 u (UI.lognot #32 (U32.v s)) (31 - i)

let unsetBit_lemma_preserves (u : u32) (i j : szlt 32)
: Lemma
    (requires j <> i)
    (ensures UI.nth #32 (unsetBit u i) (31 - j) = UI.nth #32 u (31 - j))
=
  shift_left_1_lemma_false i (31sz -^ j)

let bitmask_len (n : nat) = (n + 31) / 32
noeq
type bitmask (n : nat) : Type0 =
  | BM of larray u32 (bitmask_len n)

let core #n (BM a : bitmask n) = a


let bitmask_pts_to (#n:_) (b:bitmask n) (p : GSet.set nat)
: slprop
=
  exists* (a : lseq u32 (bitmask_len n)).
    core b |-> a **
    pure (
      forall (i : szlt n).
        (GSet.mem (SZ.v i) p = getBit (a @! (i / 32)) (i %^ 32sz))
    )

let getBit_lemma_zero ()
: Lemma
  (requires true)
  (ensures forall (i : szlt 32). getBit 0ul i == false)
=
  introduce forall (i : szlt 32). getBit 0ul i == false
  with (
    getBit_lemma 0ul i;
    UI.zero_to_vec_lemma #32 (31 - SZ.v i)
  )

fn init_empty (n:nat) (a : larray u32 (bitmask_len n))
  requires
    exists* v_a.
      a |-> v_a **
      pure (forall i. v_a @! i == 0ul)
  returns  b : bitmask n
  ensures  bitmask_pts_to b GSet.empty
{
  let b : bitmask n = BM a;
  assert rewrites_to a (core b);
  Pulse.Lib.Array.pts_to_len a;
  getBit_lemma_zero ();
  fold bitmask_pts_to b GSet.empty;
  b;
}

let getBit_lemma_full_mask ()
: Lemma
  (requires true)
  (ensures forall (i : szlt 32). getBit full_mask i == true)
=
  assert_norm (U32.v full_mask == UI.ones 32);
  introduce forall (i : szlt 32). getBit full_mask i == true
  with (
    getBit_lemma full_mask i;
    UI.ones_to_vec_lemma #32 (31 - SZ.v i)
  )


fn init_full (n:nat) (a : larray u32 (bitmask_len n))
  requires
    exists* v_a.
      a |-> v_a **
      pure (forall i. v_a @! i == (U32.uint_to_t 0xffffffff))
  returns  b : bitmask n
  ensures  bitmask_pts_to b full
{
  let b : bitmask n = BM a;
  assert rewrites_to a (core b);
  Pulse.Lib.Array.pts_to_len a;
  with va. assert a |-> va;

  assert pure (forall (i : szlt n). va @! i / 32 == full_mask);
  getBit_lemma_full_mask ();

  fold bitmask_pts_to b full;
  b
}

fn get (#n:nat) (b:bitmask n) (i:sz) (#p : GSet.set nat)
  preserves bitmask_pts_to b p
  requires  pure (i < n)
  returns   v : bool
  ensures   pure (v = GSet.mem (SZ.v i) p)
{
  unfold bitmask_pts_to;

  open Pulse.Lib.Array;
  let word = (core b).(i /^ 32sz);
  let bit = i %^ 32sz;

  let v : bool = getBit word bit;
  getBit_lemma word bit;

  fold bitmask_pts_to b p;

  v;
}

let bitmask_idx_lemma (n : nat) (i : nat)
: Lemma (requires i < n) (ensures i / 32 < bitmask_len n)
= FStar.Math.Lemmas.lemma_div_le i (n + 31) 32

let rem_lemma_not_eq (i j : nat) (n : pos)
: Lemma (requires i <> j /\ i / n == j / n) (ensures i % n <> j % n)
= ()

let set_lemma_preserves_j (#n : nat) (a : lseq u32 (bitmask_len n)) (i j : szlt n)
: Lemma
  (requires j <> i /\ i / 32 < bitmask_len n /\ j / 32 < bitmask_len n)
  (ensures
    getBit (Seq.upd a (i / 32) (setBit (a @! i / 32) (i %^ 32sz)) @! j / 32) (j %^ 32sz) ==
    getBit (a @! j / 32) (j %^ 32sz))
=
  let word = a @! i / 32 in
  let word' = setBit word (i %^ 32sz) in
  if (i / 32 = j / 32)
    then (
      Seq.lemma_index_upd1 a (i / 32) word';
      assert (Seq.upd a (i / 32) word' @! j / 32 == word');
      assert (a @! j / 32 == word);
      rem_lemma_not_eq i j 32;
      assert (SZ.v (i %^ 32sz) <> SZ.v (j %^ 32sz));
      setBit_lemma_preserves word (i %^ 32sz) (j %^ 32sz);
      getBit_lemma word (j %^ 32sz);
      getBit_lemma word' (j %^ 32sz)
    )
    else Seq.lemma_index_upd2 a (i / 32) word' (j / 32)

let set_lemma_preserves (#n : nat) (a : lseq u32 (bitmask_len n)) (i : szlt n)
: Lemma
  (requires true)
  (ensures forall (j : szlt n). j <> i ==>
    getBit (Seq.upd a (i / 32) (setBit (a @! i / 32) (i %^ 32sz)) @! j / 32) (j %^ 32sz) ==
    getBit (a @! j / 32) (j %^ 32sz)
  )
=
  introduce forall (j : szlt n). j <> i ==>
    getBit (Seq.upd a (i / 32) (setBit (a @! i / 32) (i %^ 32sz)) @! j / 32) (j %^ 32sz) ==
    getBit (a @! j / 32) (j %^ 32sz)
  with (if j <> i then (bitmask_idx_lemma n i; bitmask_idx_lemma n j; set_lemma_preserves_j a i j) else ())

let unset_lemma_preserves_j (#n : nat) (a : lseq u32 (bitmask_len n)) (i j : szlt n)
: Lemma
  (requires j <> i /\ i / 32 < bitmask_len n /\ j / 32 < bitmask_len n)
  (ensures
    getBit (Seq.upd a (i / 32) (unsetBit (a @! i / 32) (i %^ 32sz)) @! j / 32) (j %^ 32sz) ==
    getBit (a @! j / 32) (j %^ 32sz))
=
  let word = a @! i / 32 in
  let word' = unsetBit word (i %^ 32sz) in
  if (i / 32 = j / 32)
    then (
      Seq.lemma_index_upd1 a (i / 32) word';
      assert (Seq.upd a (i / 32) word' @! j / 32 == word');
      assert (a @! j / 32 == word);
      rem_lemma_not_eq i j 32;
      assert (SZ.v (i %^ 32sz) <> SZ.v (j %^ 32sz));
      unsetBit_lemma_preserves word (i %^ 32sz) (j %^ 32sz);
      getBit_lemma word (j %^ 32sz);
      getBit_lemma word' (j %^ 32sz)
    )
    else Seq.lemma_index_upd2 a (i / 32) word' (j / 32)

let unset_lemma_preserves (#n : nat) (a : lseq u32 (bitmask_len n)) (i : szlt n)
: Lemma
  (requires true)
  (ensures forall (j : szlt n). j <> i ==>
    getBit (Seq.upd a (i / 32) (unsetBit (a @! i / 32) (i %^ 32sz)) @! j / 32) (j %^ 32sz) ==
    getBit (a @! j / 32) (j %^ 32sz)
  )
=
  introduce forall (j : szlt n). j <> i ==>
    getBit (Seq.upd a (i / 32) (unsetBit (a @! i / 32) (i %^ 32sz)) @! j / 32) (j %^ 32sz) ==
    getBit (a @! j / 32) (j %^ 32sz)
  with (if j <> i then (bitmask_idx_lemma n i; bitmask_idx_lemma n j; unset_lemma_preserves_j a i j) else ())


let set_full_lemma (#n : nat) (a : lseq u32 (bitmask_len n)) (p : GSet.set nat) (i : szlt n)
: Lemma
  (requires
    i / 32 < bitmask_len n /\
    (forall (j : szlt n). GSet.mem (SZ.v j) p = getBit (a @! (j / 32)) (j %^ 32sz)))
  (ensures (
    let a' = Seq.upd a (i / 32) (setBit (a @! i / 32) (i %^ 32sz)) in
    forall (j : szlt n). GSet.mem (SZ.v j) (add (SZ.v i) p) = getBit (a' @! (j / 32)) (j %^ 32sz)))
=
  let word = a @! i / 32 in
  let word' = setBit word (i %^ 32sz) in
  let a' = Seq.upd a (i / 32) word' in
  introduce forall (j : szlt n). GSet.mem (SZ.v j) (add (SZ.v i) p) = getBit (a' @! (j / 32)) (j %^ 32sz)
  with (
    bitmask_idx_lemma n j;
    if j = i then (
      Seq.lemma_index_upd1 a (i / 32) word';
      setBit_lemma_ensures word (i %^ 32sz);
      getBit_lemma word' (i %^ 32sz)
    ) else
      set_lemma_preserves_j a i j
  )

let unset_full_lemma (#n : nat) (a : lseq u32 (bitmask_len n)) (p : GSet.set nat) (i : szlt n)
: Lemma
  (requires
    i / 32 < bitmask_len n /\
    (forall (j : szlt n). GSet.mem (SZ.v j) p = getBit (a @! (j / 32)) (j %^ 32sz)))
  (ensures (
    let a' = Seq.upd a (i / 32) (unsetBit (a @! i / 32) (i %^ 32sz)) in
    forall (j : szlt n). GSet.mem (SZ.v j) (remove (SZ.v i) p) = getBit (a' @! (j / 32)) (j %^ 32sz)))
=
  let word = a @! i / 32 in
  let word' = unsetBit word (i %^ 32sz) in
  let a' = Seq.upd a (i / 32) word' in
  introduce forall (j : szlt n). GSet.mem (SZ.v j) (remove (SZ.v i) p) = getBit (a' @! (j / 32)) (j %^ 32sz)
  with (
    bitmask_idx_lemma n j;
    if j = i then (
      Seq.lemma_index_upd1 a (i / 32) word';
      unsetBit_lemma_ensures word (i %^ 32sz);
      getBit_lemma word' (i %^ 32sz)
    ) else
      unset_lemma_preserves_j a i j
  )

fn set (#n:nat) (b:bitmask n) (i:sz) (#p : GSet.set nat)
  requires  bitmask_pts_to b p
  requires  pure (i < n)
  ensures   bitmask_pts_to b (add (SZ.v i) p)
{
  unfold bitmask_pts_to;

  with a. assert core b |-> a;

  open Pulse.Lib.Array;
  let word = (core b).(i /^ 32sz);
  let bit = i %^ 32sz;

  let word' = setBit word bit;
  (core b).(i /^ 32sz) <- word';

  bitmask_idx_lemma n (SZ.v i);
  set_full_lemma a p i;

  fold bitmask_pts_to b (add (SZ.v i) p);
}

fn unset (#n:nat) (b:bitmask n) (i:sz) (#p : GSet.set nat)
  requires  bitmask_pts_to b p
  requires  pure (i < n)
  ensures   bitmask_pts_to b (remove (SizeT.v i) p)
{
  unfold bitmask_pts_to;

  with a. assert core b |-> a;

  open Pulse.Lib.Array;
  let word = (core b).(i /^ 32sz);
  let bit = i %^ 32sz;

  let word' = unsetBit word bit;
  (core b).(i /^ 32sz) <- word';

  bitmask_idx_lemma n (SZ.v i);
  unset_full_lemma a p i;

  fold bitmask_pts_to b (remove (SZ.v i) p);
}
