module ZScratch2
#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.EMatrix
open Kuiper.Bijection
module EM = Kuiper.EMatrix
module CH = Kuiper.Chest
module SZ = Kuiper.SizeT
module A1 = Kuiper.Array1
open ZScratch

#push-options "--split_queries always --fuel 4 --ifuel 4 --z3rlimit 60"

ghost
fn test_relayout
  (#et:Type)
  (n : szp)
  (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  (g1 : tensor et (l1_forward (n *^ h *^ l)))
  (#fp : perm) (#s : CH.t ((n *^ h *^ l) @| INil) et)
  requires (g1 |-> Frac fp s)
  returns g : tensor et (l3_batched_row_major n h l)
  ensures (exists* (s3 : CH.t (n @| h @| l @| INil) et). g |-> Frac fp s3)
{
  FStar.Classical.forall_intro (imap_hyp_l3 n h l);
  relayout_via (l3_batched_row_major n h l) (fold_bij_l3 n h l) () g1;
  let g : tensor et (l3_batched_row_major n h l) = from_array (l3_batched_row_major n h l) (core g1);
  assert rewrites_to g (from_array (l3_batched_row_major n h l) (core g1));
  g
}
