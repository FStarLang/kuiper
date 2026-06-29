module Kuiper.Kernel.RowBroadcast

#lang-pulse

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT

let tid_to_cell (m n : nat) (tid : natlt (m * n))
  : abs (m @| n @| INil) =
  idx2 (tid / n) (tid % n)

unfold
let kpre
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (m n : szp)
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 ta la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 tb lb)
  (#fA : perm)
  (#sa : chest1 ta m)
  (#sb : chest2 tb m n)
  (tid : natlt (m *^ n))
  : slprop
  = a |-> Frac (fA /. (m *^ n)) sa **
    Cell b (tid_to_cell m n tid) |-> acc2 sb (tid / n) (tid % n)

unfold
let kpost
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (m n : szp)
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 ta la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 tb lb)
  (#fA : perm)
  (#sa : chest1 ta m)
  (#sb : chest2 tb m n)
  (tid : natlt (m *^ n))
  : slprop
  = a |-> Frac (fA /. (m *^ n)) sa **
    Cell b (tid_to_cell m n tid)
      |-> acc2 (s_row_broadcast f sa sb) (tid / n) (tid % n)

ghost
fn setup
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (m n : szp)
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 ta la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 tb lb)
  (#fA : perm)
  (#sa : chest1 ta m)
  (#sb : chest2 tb m n)
  ()
  norewrite
  requires
    a |-> Frac fA sa **
    b |-> sb
  ensures
    (forall+ (tid : natlt (m *^ n)).
      kpre f m n #la a #lb b #fA #sa #sb tid) **
    pure (SZ.fits (tlayout_ulen lb))
{
  admit();
  // Array2.pts_to_ref b;
  // Array1.share_n a (m *^ n);
  // Array2.explode b;
  // forevery_rw_type (Array2.ait m n) (natlt m & natlt n) _;
  // forevery_unflatten' _;
  // forevery_unfactor' (m *^ n) m n (fun r c ->
  //   Cell b (r, c) |-> macc sb r c);
  // forevery_zip #(natlt (m *^ n))
  //   (fun _ -> a |-> Frac (fA /. (m *^ n)) sa)
  //   (fun tid -> Cell b (tid_to_cell m n tid) |-> macc sb (tid / n) (tid % n));
  // forevery_ext #(natlt (m *^ n)) _ (kpre #t f m n #la a #lb b #fA #sa #sb);
  // ()
}

ghost
fn teardown
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (m n : szp)
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 ta la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 tb lb)
  (#fA : perm)
  (#sa : chest1 ta m)
  (#sb : chest2 tb m n)
  ()
  norewrite
  requires
    (forall+ (tid : natlt (m *^ n)).
      kpost f m n #la a #lb b #fA #sa #sb tid) **
    pure (SZ.fits (tlayout_ulen lb))
  ensures
    a |-> Frac fA sa **
    b |-> s_row_broadcast f sa sb
{
  admit();
  // forevery_ext #(natlt (m *^ n))
  //   (kpost #t f m n #la a #lb b #fA #sa #sb)
  //   (fun tid ->
  //     a |-> Frac (fA /. (m *^ n)) sa **
  //     Cell b (tid_to_cell m n tid)
  //       |-> macc (s_row_broadcast f sa sb) (tid / n) (tid % n));
  // forevery_unzip #(natlt (m *^ n)) _ _;
  // Array1.gather_n a (m *^ n);
  // forevery_factor' (m *^ n) m n (fun r c ->
  //   Cell b (r, c) |-> macc (s_row_broadcast f sa sb) r c);
  // forevery_flatten' (fun (ij : natlt m & natlt n) ->
  //   Cell b ij |-> macc (s_row_broadcast f sa sb) (fst ij) (snd ij));
  // forevery_rw_type (natlt m & natlt n) (Array2.ait m n) _;
  // Array2.implode b;
  ()
}

inline_for_extraction noextract
fn kf
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (m n : szp)
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 ta la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 tb lb)
  (#fA : perm)
  (#sa : chest1 ta m)
  (#sb : chest2 tb m n)
  (tid : szlt (m *^ n))
  ()
  requires
    gpu **
    kpre f m n #la a #lb b #fA #sa #sb tid
  ensures
    gpu **
    kpost f m n #la a #lb b #fA #sa #sb tid
{
  let row : sz = tid /^ n; assert rewrites_to row (tid /^ n);
  let col : sz = tid %^ n; assert rewrites_to col (tid %^ n);
  let va = tensor_read a ((row <: szlt m), ());
  let vb = tensor_read_cell b ((row <: szlt m), ((col <: szlt n), ()));
  tensor_write_cell b ((row <: szlt m), ((col <: szlt n), ())) (f va vb);
  ()
}

inline_for_extraction noextract
let kdesc
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 ta la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 tb lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#fA : perm)
  (#sa : chest1 ta m)
  (#sb : chest2 tb m n)
  : kernel_desc (requires a |-> Frac fA sa ** b |-> sb)
                (ensures  a |-> Frac fA sa ** b |-> s_row_broadcast f sa sb)
  = {
    nthr = m *^ n;
    f = kf f m n a b #fA #sa #sb;
    frame = pure (SZ.fits (tlayout_ulen lb));
    teardown = teardown f m n a b #fA #sa #sb;
    setup    = setup    f m n a b #fA #sa #sb;
    kpre  = kpre f m n #la a #lb b #fA #sa #sb;
    kpost = kpost f m n #la a #lb b #fA #sa #sb;
    kpre_sendable = solve;
    kpost_sendable = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn row_broadcast
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 ta la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 tb lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#fA : perm)
  (#sa : chest1 ta m)
  (#sb : chest2 tb m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sa)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_row_broadcast f sa sb)
{
  launch_sync (kdesc f m n a b);
}
