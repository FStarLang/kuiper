module Kuiper.View

#lang-pulse

open Kuiper
open Kuiper.Len
open Kuiper.GhostMap { is_ghost_map }
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.IView
module F = FStar.FunctionalExtensionality
module SZ = FStar.SizeT

[@@erasable]
noeq
type aview (et : Type) (st : Type0) = {
  (* An indexing view. *)
  iview : aiview;

  (* The high-level spec type is a container, roughly a ghost function ait -> et *)
  igm  : is_ghost_map st iview.sch.ait et;
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
instance enumerable_view_ait (#et:Type) (#st:Type0)
  (vw : aview et st)
  : Enumerable.enumerable vw.iview.sch.ait
= vw.iview.sch.ait_enum

unfold
instance is_ghost_map_view_igm (#et:Type) (#st:Type0)
  (vw : aview et st)
  : is_ghost_map st vw.iview.sch.ait et
  = vw.igm

(* Nothing fancy here. *)
let raw_view (#et:Type) (#len:nat) : aview et (lseq et len) = {
  iview  = IView.raw_view #len;
  igm    = solve;
}

(* Viewing as a (ghost) function from indices to elements. Forget about sequences. *)
let raw_function_view (#et:Type) (#len:nat) : aview et (natlt len ^->> et) = {
  iview  = IView.raw_view #len;
  igm    = solve;
}

(* What it means for a view to be concretizable, i.e. executable.
Note how we say **nothing** about the high-level spec type. All that
matters at runtime is the indexing structure.

This is typeclass already, there should be no need to mark it as a class here,
but alas it does not quite work. *)
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

let igm_reindex (#mt #it #et : Type) (igm : is_ghost_map mt it et)
  (#it': Type) (bij : it =~ it')
   : is_ghost_map mt it' et =
{
  acc = (fun (v : mt) (i' : it') ->
    igm.acc v (bij.gg i'));
  upd = (fun (v : mt) (i' : it') (x : et) ->
    igm.upd v (bij.gg i') x);
  bij = igm.bij `bij_comp` bij_reindex_ghost_efun _ _ _ bij;
  l1 = ez;
  l2 = ez;
}

let reindex_view (#et : Type0) (#st : Type0)
  (vw : aview et st)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.iview.sch.ait =~ ait')
  : aview et st = {
  iview  = IView.reindex_view vw.iview bij;
  igm    = igm_reindex vw.igm bij;
}

let igm_review (#mt #it #et : Type) (igm : is_ghost_map mt it et)
  (#mt': Type) (bij : mt =~ mt')
   : is_ghost_map mt' it et =
{
  acc = (fun (v : mt') (i : it) -> igm.acc (bij.gg v) i);
  upd = (fun (v : mt') (i : it) (x : et) -> bij.ff (igm.upd (bij.gg v) i x));
  bij = bij_erase (bij_sym bij) `bij_comp` igm.bij;
  l1 = ez;
  l2 = ez;
}

let review_view (#et : Type0) (#st : Type0)
  (vw : aview et st)
  (#st' : Type)
  (bij : st =~ st')
  : aview et st' = {
  iview    = vw.iview;
  igm      = igm_review vw.igm bij;
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
  (i : vw.iview.sch.ait)
  : GTot (natlt (len vw))
  = IView.it_to_nat vw.iview i

let it_of_nat
  (#a:Type) (#st:Type0)
  (vw : aview a st)
  (i: natlt (len vw){FStar.Functions.in_image vw.iview.step.imap.f i})
  : GTot vw.iview.sch.ait
  = IView.it_of_nat vw.iview i

let ci_to_ai
  (#et:Type) (#st : Type0)
  (vw : aview et st)
  {| cw : IView.ciview vw.iview |}
  (i : cw.sch.cit)
  : GTot vw.iview.sch.ait
  = IView.ci_to_ai vw.iview i

let ai_to_ci
  (#et:Type) (#st : Type0)
  (vw : aview et st)
  {| cw : IView.ciview vw.iview |}
  (i : vw.iview.sch.ait)
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
      reveal (vw.igm.acc v (it_of_nat vw i))

let from_seq
  (#a:Type) (#st:Type)
  (vw : aview a st)
  (s : lseq a (len vw))
  : GTot st
  = vw.igm.bij.gg (F.on_g vw.iview.sch.ait <| fun i -> s @! it_to_nat vw i)

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
  (i : vw.iview.sch.ait)
  (x : a)
  : Lemma (ensures to_seq vw (vw.igm.upd v i x) == Seq.upd (to_seq vw v) (it_to_nat vw i) x)
          [SMTPat (to_seq vw (vw.igm.upd v i x))]

let sum_aview
  (#et : Type) (#st1 #st2 : Type)
  (vw1 : aview et st1)
  (vw2 : aview et st2)
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
  : aview et (st1 & st2) =
{
  iview = sum_aiview vw1.iview vw2.iview;
  igm   = solve;
}
