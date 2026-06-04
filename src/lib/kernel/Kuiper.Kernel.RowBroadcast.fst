module Kuiper.Kernel.RowBroadcast

#lang-pulse

open Kuiper
module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2
module SZ = Kuiper.SizeT
open Kuiper.EMatrix
open Kuiper.Seq.Common
open Kuiper.Tensor

let tid_to_cell (m n : nat) (tid : natlt (m * n)) : (natlt m & natlt n) =
  (tid / n, tid % n)

unfold
let kpre
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (m n : szp)
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  (tid : natlt (m *^ n))
  : slprop
  = a |-> Frac (fA /. (m *^ n)) sa **
    Cell b (tid_to_cell m n tid) |-> macc sb (tid / n) (tid % n)

unfold
let kpost
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (m n : szp)
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  (tid : natlt (m *^ n))
  : slprop
  = a |-> Frac (fA /. (m *^ n)) sa **
    Cell b (tid_to_cell m n tid)
      |-> macc (s_row_broadcast f sa sb) (tid / n) (tid % n)

ghost
fn setup
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (m n : szp)
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  ()
  norewrite
  requires
    a |-> Frac fA sa **
    b |-> sb
  ensures
    (forall+ (tid : natlt (m *^ n)).
      kpre #t f m n #la a #lb b #fA #sa #sb tid) **
    pure (SZ.fits (Array2.layout_size lb))
{
  Array2.pts_to_ref b;
  Array1.share_n a (m *^ n);
  Array2.explode b;
  forevery_rw_type (Array2.ait m n) (natlt m & natlt n) _;
  forevery_unflatten' _;
  forevery_unfactor' (m *^ n) m n (fun r c ->
    Cell b (r, c) |-> macc sb r c);
  forevery_zip #(natlt (m *^ n))
    (fun _ -> a |-> Frac (fA /. (m *^ n)) sa)
    (fun tid -> Cell b (tid_to_cell m n tid) |-> macc sb (tid / n) (tid % n));
  forevery_ext #(natlt (m *^ n)) _ (kpre #t f m n #la a #lb b #fA #sa #sb);
  ()
}

ghost
fn teardown
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (m n : szp)
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  ()
  norewrite
  requires
    (forall+ (tid : natlt (m *^ n)).
      kpost #t f m n #la a #lb b #fA #sa #sb tid) **
    pure (SZ.fits (Array2.layout_size lb))
  ensures
    a |-> Frac fA sa **
    b |-> s_row_broadcast f sa sb
{
  forevery_ext #(natlt (m *^ n))
    (kpost #t f m n #la a #lb b #fA #sa #sb)
    (fun tid ->
      a |-> Frac (fA /. (m *^ n)) sa **
      Cell b (tid_to_cell m n tid)
        |-> macc (s_row_broadcast f sa sb) (tid / n) (tid % n));
  forevery_unzip #(natlt (m *^ n)) _ _;
  Array1.gather_n a (m *^ n);
  forevery_factor' (m *^ n) m n (fun r c ->
    Cell b (r, c) |-> macc (s_row_broadcast f sa sb) r c);
  forevery_flatten' (fun (ij : natlt m & natlt n) ->
    Cell b ij |-> macc (s_row_broadcast f sa sb) (fst ij) (snd ij));
  forevery_rw_type (natlt m & natlt n) (Array2.ait m n) _;
  Array2.implode b;
  ()
}

inline_for_extraction noextract
fn kf
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (m n : szp)
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  (tid : szlt (m *^ n))
  ()
  requires
    gpu **
    kpre #t f m n #la a #lb b #fA #sa #sb tid
  ensures
    gpu **
    kpost #t f m n #la a #lb b #fA #sa #sb tid
{
  let row : sz = tid /^ n; assert rewrites_to row (tid /^ n);
  let col : sz = tid %^ n; assert rewrites_to col (tid %^ n);
  let x = Array2.read_cell b (row, col);
  let v = Array1.read a row;
  Array2.write_cell b (row, col) (f x v);
}

inline_for_extraction noextract
let kdesc
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array2.is_global b))
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  : kernel_desc (requires a |-> Frac fA sa ** b |-> sb)
                (ensures  a |-> Frac fA sa ** b |-> s_row_broadcast f sa sb)
  = {
    nthr = m *^ n;
    f = kf f m n a b #fA #sa #sb;
    frame = pure (SZ.fits (Array2.layout_size lb));
    teardown = teardown f m n a b #fA #sa #sb;
    setup    = setup    f m n a b #fA #sa #sb;
    kpre  = kpre #t f m n #la a #lb b #fA #sa #sb;
    kpost = kpost #t f m n #la a #lb b #fA #sa #sb;
    kpre_sendable = solve;
    kpost_sendable = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn row_broadcast
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array2.is_global b))
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
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
