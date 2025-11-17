module Kuiper.View

#lang-pulse

open Kuiper
open Kuiper.Bijection

let it_nat_rel #a #st (vw : aview a st) (i : vw.iview.ait)
  (j : natlt (len vw){FStar.Functions.in_image vw.iview.step.imap.f j})
  : Lemma (it_to_nat vw i == j <==> i == it_of_nat vw j)
  = IView.it_nat_rel vw.iview i j

let to_from (#a:Type) (#st:Type)
  (vw : aview a st { is_full_view vw })
  (s : lseq a (len vw))
  : Lemma (ensures to_seq vw (from_seq vw s) == s)
          [SMTPat (to_seq vw (from_seq vw s))]
  = let aux (i : natlt (len vw))
      : Lemma (to_seq vw (from_seq vw s) @! i == s @! i) =
      calc (==) {
        to_seq vw (from_seq vw s) @! i;
        == {}
        Seq.init_ghost (len vw) (fun (j : natlt (len vw)) -> (vw.ctn.acc (from_seq vw s) (it_of_nat vw j))) @! i;
        == {}
        vw.ctn.acc (from_seq vw s) (it_of_nat vw i);
        == {}
        vw.ctn.acc (vw.ctn.from_fun (fun i -> s @! it_to_nat vw i)) (it_of_nat vw i);
        == { vw.ctn.from_fun_ok (fun i -> s @! it_to_nat vw i) (it_of_nat vw i) }
        s @! it_to_nat vw (it_of_nat vw i);
        == {}
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
  let aux (i : vw.iview.ait)
    : Lemma (vw.ctn.acc (from_seq vw (to_seq vw v)) i == vw.ctn.acc v i) =
    calc (==) {
      vw.ctn.acc (from_seq vw (to_seq vw v)) i;
      == { _ by (Tactics.compute ()) } // Odd.
      vw.ctn.acc (vw.ctn.from_fun (fun i -> to_seq vw v @! it_to_nat vw i)) i;
      == { vw.ctn.from_fun_ok (fun i -> to_seq vw v @! it_to_nat vw i) i }
      to_seq vw v @! it_to_nat vw i;
      == {}
      Seq.init_ghost (len vw) (fun (j : natlt (len vw)) -> reveal (vw.ctn.acc v (it_of_nat vw j))) @! it_to_nat vw i;
      == {}
      reveal (vw.ctn.acc v (it_of_nat vw (it_to_nat vw i)));
      == { it_nat_rel vw i (it_to_nat vw i) }
      reveal (vw.ctn.acc v i);
    }
  in
  Classical.forall_intro aux;
  vw.ctn.ext (from_seq vw (to_seq vw v)) v ();
  ()

let to_seq_upd (#a:Type) (#st:Type)
  (vw : aview a st { is_full_view vw })
  (v : st)
  (i : vw.iview.ait)
  (x : a)
  : Lemma (ensures to_seq vw (vw.ctn.upd v i x) == Seq.upd (to_seq vw v) (it_to_nat vw i) x)
          [SMTPat (to_seq vw (vw.ctn.upd v i x))]
=
  let aux (idx : natlt (len vw))
    : Lemma (to_seq vw (vw.ctn.upd v i x) @! idx == Seq.upd (to_seq vw v) (it_to_nat vw i) x @! idx) =
    calc (==) {
      to_seq vw (vw.ctn.upd v i x) @! idx;
      == {}
      Seq.init_ghost (len vw) (fun (j : natlt (len vw)) -> reveal (vw.ctn.acc (vw.ctn.upd v i x) (it_of_nat vw j))) @! idx;
      == {}
      reveal (vw.ctn.acc (vw.ctn.upd v i x) (it_of_nat vw idx));
      == {
        if FStar.StrongExcludedMiddle.strong_excluded_middle (it_of_nat vw idx == i)
        then vw.ctn.l1 v (it_of_nat vw idx) x
        else vw.ctn.l2 v (it_of_nat vw idx) i x
       }
      Seq.upd (to_seq vw v) (it_to_nat vw i) x @! idx;
    };
    ()
  in
  Classical.forall_intro aux;
  assert (to_seq vw (vw.ctn.upd v i x) `Seq.equal` Seq.upd (to_seq vw v) (it_to_nat vw i) x) ;
  ()
