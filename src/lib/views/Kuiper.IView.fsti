module Kuiper.IView

(* Indexing views *)

#lang-pulse

open Kuiper
open Kuiper.Len
open Kuiper.Bijection
open Kuiper.Injection
open FStar.Tactics.Typeclasses { no_method }
module SZ = Kuiper.SizeT

[@@erasable]
noeq
inline_for_extraction noextract
type aiview_schema = {
  (* abstract index type *)
  ait      : Type0;

  (* The index type must be enumerable. This is mostly so we can
     use forall+, the bijection inside here is not used much elsewhere. *)
  ait_enum : Enumerable.enumerable ait;
}

[@@erasable]
noeq
inline_for_extraction noextract
type aiview_step (from_ait to_ait : Type0) = {
  (* Index translation *)
  imap : from_ait @~> to_ait;
}

let raw_aiview_schema (len : nat) : aiview_schema = {
  ait      = natlt len;
  ait_enum = solve;
}

[@@erasable]
noeq
inline_for_extraction noextract
type aiview = {
  len  : nat;
  sch  : aiview_schema;
  step : aiview_step sch.ait (natlt len);
}

unfold
instance has_len_aiview : has_len aiview = {
  len = (fun vw -> vw.len);
}

unfold
instance enumerable_aiview_ait (vw : aiview)
  : Enumerable.enumerable vw.sch.ait
= vw.sch.ait_enum

let is_full_view (avw : aiview) : prop =
  is_surj avw.step.imap.f

let is_full_view_lempat (avw : aiview { is_full_view avw })
     (i : natlt avw.len)
  : Lemma (in_image avw.step.imap.f i)
          [SMTPat (in_image avw.step.imap.f i)]
  = ()

val full_iff_cardinal
  (vw : aiview)
  : Lemma (is_full_view vw <==> vw.sch.ait_enum._cardinal == vw.len)
          [SMTPat (is_full_view vw)]

(* Nothing fancy here. *)
inline_for_extraction noextract
let raw_view (#len : erased nat) : aiview = {
  len  = len;
  sch  = raw_aiview_schema len;
  step = { imap = inj_id; };
}

[@@erasable]
noeq
inline_for_extraction noextract
type ciview_schema (asch : aiview_schema) = {
  (* The concrete index type *)
  [@@@no_method]
  cit   : Type0;

  (* A bijection from the abstract indices to the concrete indices
     Need not be executable. NOTE: Do not mark this erased, this
     worsens SMT performance significantly. This is already an erasable type. *)
  [@@@no_method]
  bij   : asch.ait =~ cit;
}

inline_for_extraction noextract
let raw_ciview_schema (len : erased nat{SZ.fits len}) : ciview_schema (raw_aiview_schema len) = {
  cit = szlt len;
  bij = fin_size_t_bij _;
}

(* A step in a concrete indexing view. *)

inline_for_extraction noextract
class ciview_step
  (#asch1 #asch2 : aiview_schema)
  (csch1 : ciview_schema asch1)
  (csch2 : ciview_schema asch2)
  (step  : aiview_step asch1.ait asch2.ait)
=
{
  (* Concrete index translation *)
  [@@@no_method]
  cimap : csch1.cit @~>> csch2.cit;

  (* The mappings are compatible. I.e. the following diagram commutes:

     ait1    --  imap -->   ait2
       |                      |
       |                      |
   csch1.bij              csch2.bij
       |                      |
       |                      |
     cit1    --  cimap -->  cit2
   *)
  [@@@no_method]
  compat :
    ai : asch1.ait ->
      squash (cimap.cf (csch1.bij.ff ai) == csch2.bij.ff (step.imap.f ai));
}

(* What it means for an indexing view to be concretizable, i.e. executable. *)
inline_for_extraction noextract
class ciview (avw : aiview) =
{
  [@@@no_method]
  clen : (clen : SZ.t {SZ.v clen == avw.len});

  [@@@no_method]
  sch  : ciview_schema avw.sch;

  [@@@no_method]
  step : ciview_step sch (raw_ciview_schema avw.len) avw.step;
}

inline_for_extraction noextract
instance concrete_raw_view (#len : nat{SZ.fits len}) : ciview (raw_view #len) = {
  clen = SZ.uint_to_t len; // weird
  sch  = raw_ciview_schema len;
  step = {
    cimap  = cinj_id;
    compat = ez;
  };
}

let inj_bij (#a #b : Type) (bij : a =~ b) : (a @~> b) =
{
  f = bij.ff;
  is_inj = ez;
}

let inj_bij' (#a #b : Type) (bij : a =~ b) : (b @~> a) =
{
  f = bij.gg;
  is_inj = ez;
}

let reindex_view
  (vw : aiview)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.sch.ait =~ ait')
  : aiview = {
  len = vw.len;
  sch = {
    ait      = ait';
    ait_enum = solve;
  };

  step = {
    imap = inj_bij' bij `inj_comp` vw.step.imap;
  }
}

let it_to_nat
  (vw : aiview)
  (i : vw.sch.ait)
  : GTot (natlt vw.len)
  = i |~> vw.step.imap

let it_of_nat
  (vw : aiview)
  (i: natlt vw.len{FStar.Functions.in_image vw.step.imap.f i})
  : GTot vw.sch.ait
  = i <~| vw.step.imap

val it_nat_rel (vw : aiview) (i : vw.sch.ait)
  (j : natlt vw.len{FStar.Functions.in_image vw.step.imap.f j})
  : Lemma (it_to_nat vw i == j <==> i == it_of_nat vw j)
          [SMTPat (it_to_nat vw i); SMTPat (it_of_nat vw j)]

let ci_to_ai
  (vw : aiview) {| cw : ciview vw |}
  (i : cw.sch.cit)
  : GTot vw.sch.ait
  = let open Kuiper.Bijection in
    i <~| cw.sch.bij

let ai_to_ci
  (vw : aiview) {| cw : ciview vw |}
  (i : vw.sch.ait)
  : GTot cw.sch.cit
  = let open Kuiper.Bijection in
    i |~> cw.sch.bij

let sum_aiview
  (vw1 vw2 : aiview) // { vw1.len == vw2.len })
  (#_ : squash (no_overlap vw1.step.imap.f vw2.step.imap.f))
  : aiview = {
  len = max vw1.len vw2.len;
  sch = {
    ait      = either vw1.sch.ait vw2.sch.ait;
    ait_enum = solve;
  };
  step = {
    imap     = {
      f      = merge_either vw1.step.imap.f vw2.step.imap.f;
      is_inj = ez;
    };
  };
}

let compose_astep (#sch1 #sch2 #sch3 : Type0)
  (step12 : aiview_step sch1 sch2)
  (step23 : aiview_step sch2 sch3)
  : aiview_step sch1 sch3 = {
  imap = step12.imap `inj_comp` step23.imap;
}

inline_for_extraction noextract
let compose_cstep
  (#asch1 #asch2 #asch3 : aiview_schema)
  (#csch1 : ciview_schema asch1)
  (#csch2 : ciview_schema asch2)
  (#csch3 : ciview_schema asch3)
  (#step12 : aiview_step asch1.ait asch2.ait)
  (#step23 : aiview_step asch2.ait asch3.ait)
  (c1 : ciview_step csch1 csch2 step12)
  (c2 : ciview_step csch2 csch3 step23)
  : ciview_step csch1 csch3 (compose_astep step12 step23) =
{
  cimap = c1.cimap `cinj_comp` c2.cimap;

  compat = (fun ai ->
    c1.compat ai;
    c2.compat (step12.imap.f ai)
  );
}

val full_view_bij (avw : aiview { is_full_view avw })
  : Ghost (avw.sch.ait =~ natlt avw.len)
          (requires True)
          (ensures fun b -> forall x. b.ff x == it_to_nat avw x)
