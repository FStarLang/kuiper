module KPlay.Proof.Optimized.Memory

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open FStar.Ghost
module A = Pulse.Lib.Array
module S = FStar.Seq
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let all_cells (_:nat) : prop = True

inline_for_extraction noextract
let outside (start count:nat) (i:nat) : prop = ~(start <= i /\ i < start + count)

inline_for_extraction noextract
let ready (#a:Type0) (#capacity:nat) (before:lseq (option a) capacity)
  (start count:nat{start + count <= capacity}) : prop =
  forall (i:nat{start <= i /\ i < start + count}). Some? (S.index before i)

inline_for_extraction noextract
let present (#a:Type0) (v:option a{Some? v}) : a =
  let Some x = v in x

inline_for_extraction noextract
let values (#a:Type0) (#capacity:nat) (before:lseq (option a) capacity)
  (start count:nat{start + count <= capacity}) (initialized:squash (ready before start count))
  : GTot (lseq a count) =
  S.init_ghost count (fun i -> present (S.index before (start + i)))

inline_for_extraction noextract
let replace (#a:Type0) (#capacity:nat) (#count:nat) (before:lseq (option a) capacity)
  (start:nat{start + count <= capacity}) (patch:lseq a count)
  : GTot (lseq (option a) capacity) =
  S.init_ghost capacity (fun i ->
    if start <= i && i < start + count then Some (S.index patch (i - start))
    else S.index before i)

let restore_readonly (#a:Type0) (#capacity:nat) (before:lseq (option a) capacity)
  (start count:nat{start + count <= capacity}) (initialized:squash (ready before start count))
  : Lemma (replace before start (values before start count initialized) == before) =
  assert (S.equal (replace before start (values before start count initialized)) before);
  S.lemma_eq_elim (replace before start (values before start count initialized)) before

inline_for_extraction noextract
let view (#a:Type0) (#capacity:nat) (buffer:larray a capacity)
  (start count:nat{start + count <= capacity}) : GTot (larray a count) =
  A.gsub buffer start (start + count)

let view_address (#a:Type0) (#capacity:nat) (buffer:larray a capacity)
  (start count:nat{start + count <= capacity})
  : Lemma (
      A.base_of (view buffer start count) == A.base_of buffer /\
      A.offset_of (view buffer start count) == A.offset_of buffer + start) =
  A.base_of_gsub buffer start (start + count);
  A.offset_of_gsub buffer start (start + count)

noextract
fn borrow (#a:Type0) (#capacity:nat) (buffer:larray a capacity)
  (start count:sz{0 < SZ.v count /\ SZ.v start + SZ.v count <= capacity})
  (#before:erased (lseq (option a) capacity)) (#fraction:perm)
  (initialized:squash (ready before (SZ.v start) (SZ.v count)))
  preserves gpu
  requires A.pts_to_mask buffer #fraction before all_cells
  returns part:larray a (SZ.v count)
  ensures rewrites_to part (view buffer (SZ.v start) (SZ.v count))
  ensures A.pts_to_mask buffer #fraction before (outside (SZ.v start) (SZ.v count))
  ensures part |-> Frac fraction (values before (SZ.v start) (SZ.v count) initialized)
{
  let part = A.sub buffer start (SZ.v start + SZ.v count);
  A.mask_mext buffer #fraction #before
    #(fun i -> all_cells i /\ ~(SZ.v start <= i /\ i < SZ.v start + SZ.v count))
    (outside (SZ.v start) (SZ.v count));
  A.from_mask part;
  with actual. assert part |-> Frac fraction actual;
  A.pts_to_len part;
  assert pure (S.equal actual (values before (SZ.v start) (SZ.v count) initialized));
  S.lemma_eq_elim actual (values before (SZ.v start) (SZ.v count) initialized);
  part
}

ghost
fn restore (#a:Type0) (#capacity:nat) (buffer:larray a capacity)
  (start count:nat{start + count <= capacity})
  (#before:erased (lseq (option a) capacity)) (#patch:erased (lseq a count)) (#fraction:perm)
  requires A.pts_to_mask buffer #fraction before (outside start count)
  requires A.pts_to (view buffer start count) #fraction patch
  ensures A.pts_to_mask buffer #fraction (replace before start patch) all_cells
{
  A.to_mask (view buffer start count);
  with masked. assert A.pts_to_mask (view buffer start count) #fraction masked (fun _ -> True);
  let original:erased (S.seq (option a)) = hide #(S.seq (option a)) (reveal #(lseq (option a) capacity) before);
  A.return_sub buffer #fraction #original #masked #(outside start count) #(fun _ -> True) #start #(start + count);
  with after. assert A.pts_to_mask buffer #fraction after
    (fun i -> outside start count i \/ (start <= i /\ i < start + count));
  A.mask_mext buffer #fraction #after
    #(fun i -> outside start count i \/ (start <= i /\ i < start + count)) all_cells;
  assert pure (S.equal after (replace before start patch));
  S.lemma_eq_elim after (replace before start patch);
  rewrite A.pts_to_mask buffer #fraction after all_cells
    as A.pts_to_mask buffer #fraction (replace before start patch) all_cells;
}

inline_for_extraction noextract
let masked (#a:Type0) (#n:nat) (s:lseq a n) : GTot (lseq (option a) n) =
  S.init_ghost n (fun i -> Some (S.index s i))

inline_for_extraction noextract
let overwrite (#a:Type0) (#n #count:nat) (s:lseq a n)
  (start:nat{start + count <= n}) (patch:lseq a count) : GTot (lseq a n) =
  S.init_ghost n (fun i ->
    if start <= i && i < start + count then S.index patch (i - start)
    else S.index s i)

let overwrite_at (#a:Type0) (#n #count:nat) (s:lseq a n)
  (start:nat{start + count <= n}) (patch:lseq a count) (i:nat{i < n})
  : Lemma (
      S.index (overwrite s start patch) i ==
        (if start <= i && i < start + count then S.index patch (i - start)
         else S.index s i)) = ()

let masked_window (#a:Type0) (#n:nat) (s:lseq a n)
  (start count:nat{start + count <= n})
  : Lemma (
      ready (masked s) start count /\
      values (masked s) start count () == S.slice s start (start + count)) =
  assert (S.equal (values (masked s) start count ()) (S.slice s start (start + count)));
  S.lemma_eq_elim (values (masked s) start count ()) (S.slice s start (start + count))

let overwrite_readonly (#a:Type0) (#n:nat) (s:lseq a n)
  (start count:nat{start + count <= n})
  : Lemma (overwrite #a #n #count s start (S.slice s start (start + count)) == s) =
  assert (S.equal (overwrite #a #n #count s start (S.slice s start (start + count))) s);
  S.lemma_eq_elim (overwrite #a #n #count s start (S.slice s start (start + count))) s

noextract
fn take (#a:Type0) (#n:nat) (buffer:larray a n)
  (start count:sz{0 < SZ.v count /\ SZ.v start + SZ.v count <= n})
  (#before:erased (lseq a n)) (#fraction:perm)
  preserves gpu
  requires buffer |-> Frac fraction before
  returns part:larray a (SZ.v count)
  ensures rewrites_to part (view buffer (SZ.v start) (SZ.v count))
  ensures A.pts_to_mask buffer #fraction (masked before) (outside (SZ.v start) (SZ.v count))
  ensures part |-> Frac fraction (S.slice before (SZ.v start) (SZ.v start + SZ.v count))
{
  A.to_mask buffer;
  with snapshot. assert A.pts_to_mask buffer #fraction snapshot all_cells;
  assert pure (S.equal snapshot (masked before));
  S.lemma_eq_elim snapshot (masked before);
  masked_window before (SZ.v start) (SZ.v count);
  borrow buffer start count #(masked before) #fraction ()
}

ghost
fn give (#a:Type0) (#n:nat) (buffer:larray a n)
  (start count:nat{start + count <= n})
  (#before:erased (lseq a n)) (#patch:erased (lseq a count)) (#fraction:perm)
  requires A.pts_to_mask buffer #fraction (masked before) (outside start count)
  requires A.pts_to (view buffer start count) #fraction patch
  ensures buffer |-> Frac fraction (overwrite before start patch)
{
  restore buffer start count #(masked before) #patch #fraction;
  A.from_mask buffer;
  with after. assert buffer |-> Frac fraction after;
  assert pure (S.equal after (overwrite before start patch));
  S.lemma_eq_elim after (overwrite before start patch);
}

noextract
fn take_mask (#a:Type0) (#n:nat) (buffer:larray a n)
  (start count:sz{0 < SZ.v count /\ SZ.v start + SZ.v count <= n})
  (#before:erased (lseq (option a) n)) (#fraction:perm)
  preserves gpu
  requires A.pts_to_mask buffer #fraction before all_cells
  returns part:larray a (SZ.v count)
  ensures rewrites_to part (view buffer (SZ.v start) (SZ.v count))
  ensures A.pts_to_mask buffer #fraction before (outside (SZ.v start) (SZ.v count))
  ensures A.pts_to_mask part #fraction
    (S.slice before (SZ.v start) (SZ.v start + SZ.v count)) all_cells
{
  let part = A.sub buffer start (SZ.v start + SZ.v count);
  A.mask_mext buffer #fraction #before
    #(fun i -> all_cells i /\ ~(SZ.v start <= i /\ i < SZ.v start + SZ.v count))
    (outside (SZ.v start) (SZ.v count));
  part
}
