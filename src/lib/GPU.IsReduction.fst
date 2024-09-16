module GPU.IsReduction

open GPU.Functions
open FStar.Seq

let is_permutation (#a:Type) (s1 s2 :seq a) : prop = magic()

let lemma_Singl z f r =
  let pf : is_reduction z f seq![r] r = Singl r in
  Squash.return_squash pf

let rec __ac_eq_foldl
  (#a:Type) (z : a) (f : a -> a -> a) (s : seq a) (r : a)
  (pf : is_reduction z f s r)
  : Lemma (requires is_comm_semigroup z f)
          (ensures r == GPU.Seq.Common.seq_fold_left f z s)
          (decreases pf)
= match pf with
  | Emp -> ()
  | Singl r -> ()
  | Split s1 s2 r1 r2 pf1 pf2 ->
    let _ = __ac_eq_foldl z f s1 r1 pf1 in
    let _ = __ac_eq_foldl z f s2 r2 pf2 in
    (* Prove *)
    assume (f (GPU.Seq.Common.seq_fold_left f z s1) (GPU.Seq.Common.seq_fold_left f z s2)
             ==
           GPU.Seq.Common.seq_fold_left f z s);
    ()
  | Perm s1 s2 r perm pf' ->
    admit();
    let _ = __ac_eq_foldl z f s1 r pf' in
    ()

(* We should really make this easier in F*, it's just grabbing the proof of is_reduction and
making it concrete to call the lemma above. *)
let ac_eq_foldl
  (#a:Type) (z : a) (f : a -> a -> a) (s : seq a) (r : a)
  : Lemma (requires is_comm_semigroup z f /\ is_reduction z f s r)
          (ensures r == GPU.Seq.Common.seq_fold_left f z s)
          [SMTPat (is_reduction z f s r)]
= let pf : squash (is_reduction z f s r) = () in
  Squash.bind_squash pf (fun pf -> __ac_eq_foldl z f s r pf)
    <: squash (r == GPU.Seq.Common.seq_fold_left f z s)

let assoc_uniq_reduction 
  (#a:Type) (z:a) (f : a -> a -> a) (xs : seq a) (r1 r2 : a)
: Lemma (requires is_comm_semigroup z f /\ is_reduction z f xs r1 /\ is_reduction z f xs r2)
        (ensures r1 == r2)
= ac_eq_foldl z f xs r1;
  ac_eq_foldl z f xs r2;
  ()

(* Again, quite terrible to write. *)
let op_is_reduction
  (#a:Type) (z:a) (f : a -> a -> a)
  (s1 : seq a) (r1 : a)
  (s2 : seq a) (r2 : a)
: Lemma (requires is_reduction z f s1 r1 /\ is_reduction z f s2 r2)
        (ensures is_reduction z f (s1 `Seq.append` s2) (f r1 r2))
= Squash.bind_squash () (fun pf1 ->
  Squash.bind_squash () (fun pf2 ->
  let pf = Split s1 s2 r1 r2 pf1 pf2 in
  Squash.return_squash pf))
    <: squash (is_reduction z f (s1 `Seq.append` s2) (f r1 r2))
