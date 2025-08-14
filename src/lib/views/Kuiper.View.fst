module Kuiper.View

#lang-pulse

open Kuiper
open Kuiper.Bijection
module F = FStar.FunctionalExtensionality

let to_from (#a:Type) (#st:Type)
  (vw : aview a st { is_full_view vw })
  (s : lseq a (len vw))
  : Lemma (ensures to_seq vw (from_seq vw s) == s)
          [SMTPat (to_seq vw (from_seq vw s))]
  = let _ = vw.igm.bij.gg (F.on_g vw.iview.sch.ait <| fun i -> s @! it_to_nat vw i) in
    (* funny, mentioning the term above (= from_seq vw s) makes the proof work. *)
    let aux (i : natlt (len vw))
      : Lemma (to_seq vw (from_seq vw s) @! i == s @! i) =
      calc (==) {
        to_seq vw (from_seq vw s) @! i;
        == {}
        Seq.init_ghost (len vw) (fun (j : natlt (len vw)) -> reveal (vw.igm.acc (from_seq vw s) (it_of_nat vw j))) @! i;
        == {}
        reveal (vw.igm.acc (from_seq vw s) (it_of_nat vw i));
        == {}
        vw.igm.bij.ff (from_seq vw s) (it_of_nat vw i);
        == {}
        vw.igm.bij.ff (vw.igm.bij.gg (F.on_g vw.iview.sch.ait <| fun i -> s @! it_to_nat vw i))
          (it_of_nat vw i);
        == {}
        (F.on_g vw.iview.sch.ait <| fun i -> s @! it_to_nat vw i)
          (it_of_nat vw i);
        == {}
        s @! it_to_nat vw (it_of_nat vw i);
        == { assert_norm (it_to_nat vw (it_of_nat vw i) == i) }
        s @! i;
      }
    in
    Classical.forall_intro aux;
    assert (Seq.equal s (to_seq vw (from_seq vw s)))

let from_to (#a:Type) (#st:Type)
  (vw : aview a st { is_full_view vw })
  (v : st)
  : Lemma (ensures from_seq vw (to_seq vw v) == v)
          [SMTPat (from_seq vw (to_seq vw v))]
=
  let lhs = from_seq vw (to_seq vw v) in
  let rhs = v in
  let lf = vw.igm.bij.ff lhs in
  let rf = vw.igm.bij.ff rhs in
  let aux (i : vw.iview.sch.ait)
    : Lemma (lf i == rf i) =
    calc (==) {
      lf i;
      == {}
      vw.igm.bij.ff (from_seq vw (to_seq vw v)) i;
      == {}
      vw.igm.bij.ff (vw.igm.bij.gg (F.on_g vw.iview.sch.ait <| fun i -> to_seq vw v @! it_to_nat vw i)) i;
      == {}
      (F.on_g vw.iview.sch.ait <| fun i -> to_seq vw v @! it_to_nat vw i) i;
      == {}
      to_seq vw v @! it_to_nat vw i;
      == {}
      rf i;
    }
  in
  Classical.forall_intro aux;
  assert (F.feq_g lf rf);
  calc (==) {
    lhs;
    == { vw.igm.bij.gg_ff lhs }
    vw.igm.bij.gg lf <: st;
    == {}
    vw.igm.bij.gg rf <: st;
    == { vw.igm.bij.gg_ff rhs }
    rhs;
  }

#push-options "--retry 5" // flaky, but fast when it works
let to_seq_upd (#a:Type) (#st:Type)
  (vw : aview a st { is_full_view vw })
  (v : st)
  (i : vw.iview.sch.ait)
  (x : a)
  : Lemma (ensures to_seq vw (vw.igm.upd v i x) == Seq.upd (to_seq vw v) (it_to_nat vw i) x)
          [SMTPat (to_seq vw (vw.igm.upd v i x))]
=
  let aux (idx : natlt (len vw))
    : Lemma (to_seq vw (vw.igm.upd v i x) @! idx == Seq.upd (to_seq vw v) (it_to_nat vw i) x @! idx) =
    vw.igm.l2 (it_of_nat vw idx) v x;
    // calc (==) {
    //   to_seq vw (vw.igm.upd v i x) @! idx;
    //   == {}
    //   // Seq.init_ghost (len vw) (fun (j : natlt (len vw)) -> reveal (vw.igm.acc (vw.igm.upd v i x) (it_of_nat vw j))) @! idx;
    //   // == {}
    //   // reveal (vw.igm.acc (vw.igm.upd v i x) (it_of_nat vw idx));
    //   // == { vw.igm.l2 (it_of_nat vw idx) v x }
    //   // if FStar.StrongExcludedMiddle.strong_excluded_middle (it_of_nat vw idx == i)
    //   // then x
    //   // else reveal (vw.igm.acc v (it_of_nat vw idx));
    //   // == {}
    //   Seq.upd (to_seq vw v) (it_to_nat vw i) x @! idx;
    // };
    ()
  in
  Classical.forall_intro aux;
  assert (to_seq vw (vw.igm.upd v i x) `Seq.equal` Seq.upd (to_seq vw v) (it_to_nat vw i) x) ;
  ()
#pop-options
