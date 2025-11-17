module Kuiper.View

#lang-pulse

open Kuiper
open Kuiper.Len
open Kuiper.Container { container }
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.IView
module F = FStar.FunctionalExtensionality
module SZ = Kuiper.SizeT

[@@erasable]
noeq
type aview (et : Type) (st : Type0) = {
  (* An indexing view. *)
  iview : aiview;

  (* The high-level spec type is a container, roughly a ghost function ait -> et *)
  ctn  : container st iview.ait et;
}

unfold
instance has_len_aview (et st : Type) : has_len (aview et st) = {
  len = (fun vw -> vw.iview.len);
}

let _helper (et st : Type) (vw : aview et st)
  : Lemma (len vw == vw.iview.len)
          [SMTPat (len vw)]
  = ()

unfold
instance is_container_view_igm (#et:Type) (#st:Type0)
  (vw : aview et st)
  : container st vw.iview.ait et
  = vw.ctn

(* Nothing fancy here. *)
let raw_view (#et:Type) (#len:nat) : aview et (lseq et len) = {
  iview  = IView.raw_view #len;
  ctn    = solve;
}

(* Viewing as a (ghost) function from indices to elements. Forget about sequences. *)
let raw_function_view (#et:Type) (#len:nat) : aview et (natlt len ^->> et) = {
  iview  = IView.raw_view #len;
  ctn    = solve;
}

(* What it means for a view to be concretizable, i.e. executable.
Note how we say **nothing** about the high-level spec type. All that
matters at runtime is the indexing structure. *)
inline_for_extraction noextract
unfold
let cview (#et : Type0) (#st : Type0)
  (avw : aview et st) = IView.ciview avw.iview

noextract
let bij_reindex_ghost_efun (it it' et : Type)
  (bij : it =~ it')
  : ((it ^->> et) =~ (it' ^->> et))
= Mkbijection #(it ^->> et) #(it' ^->> et)
  (fun f -> F.on_g _ <| fun it' -> f (bij.gg it'))
  (fun g -> F.on_g _ <| fun it  -> g (bij.ff it))
  (fun f -> assert (F.feq_g (F.on_g _ <| fun it' -> f (bij.ff (bij.gg it')))
                            f))
  (fun f -> assert (F.feq_g (F.on_g _ <| fun it  -> f (bij.gg (bij.ff it)))
                            f))

let container_reindex (#mt #it #et : Type) (ctn : container mt it et)
  (#it': Type) (bij : it =~ it')
   : container mt it' et =
{
  acc = (fun (v : mt) (i' : it') -> ctn.acc v (bij.gg i'));
  upd = (fun (v : mt) (i' : it') (x : et) -> ctn.upd v (bij.gg i') x);
  l1 = (fun c i v -> ctn.l1 c (bij.gg i) v);
  l2 = (fun c i1 i2 v -> ctn.l2 c (bij.gg i1) (bij.gg i2) v);
  ext = (fun c1 c2 _ ->
    assert (forall (i : it).
      ctn.acc c1 i == ctn.acc c1 (bij.gg (bij.ff i)));
    ctn.ext c1 c2 ());
  from_fun = (fun f -> ctn.from_fun (fun i -> f (bij.ff i)));
  from_fun_ok = (fun f i -> ctn.from_fun_ok (fun i' -> f (bij.ff i')) (bij.gg i));
}

let reindex_view (#et : Type0) (#st : Type0)
  (vw : aview et st)
  (#ait' : Type)
  (bij : vw.iview.ait =~ ait')
  : aview et st = {
  iview  = IView.reindex_view vw.iview bij;
  ctn    = container_reindex vw.ctn bij;
}

let container_review (#mt #it #et : Type) (ctn : container mt it et)
  (#mt': Type) (bij : mt =~ mt')
   : container mt' it et =
{
  acc = (fun (v : mt') (i : it) -> ctn.acc (bij.gg v) i);
  upd = (fun (v : mt') (i : it) (x : et) -> bij.ff (ctn.upd (bij.gg v) i x));
  l1 = (fun c i v -> ctn.l1 (bij.gg c) i v; bij.gg_ff (ctn.upd (bij.gg c) i v));
  l2 = (fun c i1 i2 v -> ctn.l2 (bij.gg c) i1 i2 v);
  ext = (fun c1 c2 _ -> ctn.ext (bij.gg c1) (bij.gg c2) ());
  from_fun = (fun f -> bij.ff (ctn.from_fun f));
  from_fun_ok = (fun f i -> ctn.from_fun_ok f i);
}

let review_view (#et : Type0) (#st : Type0)
  (vw : aview et st)
  (#st' : Type)
  (bij : st =~ st')
  : aview et st' = {
  iview    = vw.iview;
  ctn      = container_review vw.ctn bij;
}

unfold
inline_for_extraction noextract
let concrete_raw_view (#et:Type) (#len:nat{SZ.fits len}) : cview (raw_view #et #len) =
  IView.concrete_raw_view #len

unfold
inline_for_extraction noextract
let concrete_raw_function_view (#et:Type) (#len:nat{SZ.fits len}) : cview (raw_function_view #et #len) =
  IView.concrete_raw_view #len

(* Redefining these *)
let it_to_nat
  (#a:Type) (#st:Type0)
  (vw : aview a st)
  (i : vw.iview.ait)
  : GTot (natlt (len vw))
  = IView.it_to_nat vw.iview i

let it_of_nat
  (#a:Type) (#st:Type0)
  (vw : aview a st)
  (i: natlt (len vw){FStar.Functions.in_image vw.iview.step.imap.f i})
  : GTot vw.iview.ait
  = IView.it_of_nat vw.iview i

val it_nat_rel #a #st (vw : aview a st) (i : vw.iview.ait)
  (j : natlt (len vw){FStar.Functions.in_image vw.iview.step.imap.f j})
  : Lemma (it_to_nat vw i == j <==> i == it_of_nat vw j)
          [SMTPat (it_to_nat vw i); SMTPat (it_of_nat vw j)]

let ci_to_ai
  (#et:Type) (#st : Type0)
  (vw : aview et st)
  {| cw : IView.ciview vw.iview |}
  (i : cw.sch.cit)
  : GTot vw.iview.ait
  = IView.ci_to_ai vw.iview i

let ai_to_ci
  (#et:Type) (#st : Type0)
  (vw : aview et st)
  {| cw : IView.ciview vw.iview |}
  (i : vw.iview.ait)
  : GTot cw.sch.cit
  = IView.ai_to_ci vw.iview i


(* Operating over the abstract view. *)

let is_full_view #et #st (vw : aview et st) : prop =
  IView.is_full_view vw.iview

let to_seq
  (#a:Type) (#st : Type0)
  (vw : aview a st { is_full_view vw })
  (v : st)
  : GTot (lseq a (len vw))
  = Seq.init_ghost (len vw) fun (i : natlt (len vw)) ->
      reveal (vw.ctn.acc v (it_of_nat vw i))

let from_seq
  (#a:Type) (#st:Type)
  (vw : aview a st)
  (s : lseq a (len vw))
  : GTot st
  = vw.ctn.from_fun (fun i -> s @! it_to_nat vw i)

val to_from (#a:Type) (#st:Type)
  (vw : aview a st { is_full_view vw })
  (s : lseq a (len vw))
  : Lemma (ensures to_seq vw (from_seq vw s) == s)
          [SMTPat (to_seq vw (from_seq vw s))]

val from_to (#a:Type) (#st:Type)
  (vw : aview a st { is_full_view vw })
  (v : st)
  : Lemma (ensures from_seq vw (to_seq vw v) == v)
          [SMTPat (from_seq vw (to_seq vw v))]

val to_seq_upd (#a:Type) (#st:Type)
  (vw : aview a st { is_full_view vw })
  (v : st)
  (i : vw.iview.ait)
  (x : a)
  : Lemma (ensures to_seq vw (vw.ctn.upd v i x) == Seq.upd (to_seq vw v) (it_to_nat vw i) x)
          [SMTPat (to_seq vw (vw.ctn.upd v i x))]

let sum_aview
  (#et : Type) (#st1 #st2 : Type)
  (vw1 : aview et st1)
  (vw2 : aview et st2)
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
  : aview et (st1 & st2) =
{
  iview = sum_aiview vw1.iview vw2.iview;
  ctn   = solve;
}

let no_overlap_fam
  (#et : Type0) (#st : Type)
  (n : nat)
  (vw : natlt n -> aview et st)
  : prop
  = IView.no_overlap_fam n (fun i -> (vw i).iview)

let sum_aview_fam
  (#et : Type) (#st : Type)
  (n : pos)
  (vws : natlt n -> aview et st)
  (#_ : squash (no_overlap_fam n vws))
  : aview et (natlt n ^->> st) =
{
  iview = sum_aiview_fam n (fun i -> (vws i).iview);
  ctn   = solve;
}
