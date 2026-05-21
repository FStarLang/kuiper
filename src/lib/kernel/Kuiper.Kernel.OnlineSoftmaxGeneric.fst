module Kuiper.Kernel.OnlineSoftmaxGeneric

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
module SZ = FStar.SizeT
open Kuiper.Array1
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Softmax

let rmax_seq (s: Seq.seq real { Seq.length s > 0 }) : real = seq_fold_left rmax (s @! 0) (seq_drop 1 s)

let online_softmax_invariant (l m: real) (z: seq real {Seq.length z > 0}) (k: natle (Seq.length z)): prop =
  (k == 0 ==> m == (z @! 0)) /\
  (k > 0 ==> (m == rmax_seq (seq_take k z))) /\
  (l == rsum (seq_map (fun zi -> rexp (zi -. m)) z))

let online_softmax_upd 
  (l m: real)
  (z: seq real {Seq.length z > 0}) (i: natlt (Seq.length z)): 
  (real & real & real & real) = 
  let m' = rmax m (z @! i) in
  let adj = rexp (m -. m') in
  let si = rexp ((z @! i) -. m') in
  let l' = adj *. l +. si in
  (l', m', si, adj)

let lem_online_softmax_upd
  (l m: real)
  (z: seq real { Seq.length z > 0 })
  (k : natlt ( Seq.length z - 1 )) 
  (f: real -> real -> real) :
  Lemma (requires online_softmax_invariant l m z k)
        (ensures (let (l', m', _, _) = online_softmax_upd l m z k in
          online_softmax_invariant l' m' z (k + 1))) = admit ()

let online_softmax_init
  (z: seq real {Seq.length z > 0}) :
  (real & real) =
  (0.R, z @! 0)

let lem_online_softmax_init
  (z : seq real {Seq.length z > 0}) :
  Lemma (ensures (let (l,m) = online_softmax_init z in 
    online_softmax_invariant l m z 0)) = admit ()

let lem_online_softmax_elem
  (l m: real)
  (z: seq real {Seq.length z > 0})
  (k: natle (Seq.length z) { k > 0 })
  (i: natlt (Seq.length z) { i < k }) : 
  Lemma (requires online_softmax_invariant l m z k)
        (ensures (rexp ((z @! i) -. m)) /. (l) == (softmax_real (seq_take k z)) @! i) = admit ()

let lem_online_softmax_adj
  (l m: real)
  (z: seq real {Seq.length z > 0})
  (f: real -> natlt (Seq.length z) -> real)
  (op: real -> real -> real {forall (a b r: real). (r *. a) `op` (r *. b) == r *. (a `op` b)})
  (init: real)
  (* TODO: get rid of the extra constraint here. The lemma still holds when k = 0, but we just can't call softmax_real on an empty sequence
    (seq_take 0 z) because that would entail dividing by 0. *)
  (* maybe it just needs to be a "softmax_real_safe" that returns an empty sequence given an empty sequence as input.
    Because it would also not be possible to have the invariant below hold before the loop (when k = 0) on the client side code. *)
  (k: natlt (Seq.length z) { k > 0 }): 
  Lemma (requires online_softmax_invariant l m z k)
        (ensures 
          (let (l', m', si, adj) = online_softmax_upd l m z k in
          (seq_fold_left op init (seq_mapi (softmax_real (seq_take k z)) f)) *. l *. adj `op` (f si k) ==
          (seq_fold_left op init (seq_mapi (softmax_real (seq_take (k+1) z)) f)) *. l')) = admit ()
          (* in client side code, we would divide by l to get the final thing *)


(* this is what the client code for dotproduct would have as the loop invariant for the running dotproduct sum *)
let dotprod_invariant_example 
  (sum: ref real)
  (l: real) (* managed by online_softmax_upd *)
  (a: seq real {Seq.length a > 0})
  (b: seq real {Seq.length b == Seq.length a})
  (* TODO: once again remove side condition for i > 0 *)
  (i: natle (Seq.length a) { i > 0 }): slprop =
    sum |-> ((seq_fold_left (+.) 0.0R (seq_mapi (softmax_real (seq_take i a)) (fun s i -> s *. (b @! i)))) *. l)

(*

previous attempts :

inline_for_extraction noextract
fn online_softmax_generic_one
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lena #leno : szp)
  (#la : Kuiper.Array1.layout lena) {| ctlayout la |}
  (#lo : Kuiper.Array1.layout leno) {| ctlayout lo |}
  (a : array1 et la)
  (o : array1 et lo)
  (lr mr : ref et)
  (frame: slprop) (* TODO: what if fpre and fpost need different `frame` ? problem is we can't request both since they may not be disjoint..  *)
  (fpre : fn (i:szlt lena)
            preserves frame
            returns r : et)
  (fpost : fn (a : array1 et la) (i:szlt leno)
            preserves frame ** (exists* (va: (lseq et lena)). a |-> va)
            returns r : et)
(* LATER: values: 
    (#va : erased (lseq et lenab))
    (ra : erased (lseq real lenab) { va %~ ra }) 
  (#_: squash (seq_forallb not_nan va)) *)
  ()
  preserves
    frame **
    (exists* (va: (lseq et lena)). a |-> va) **
    (exists* (vo: (lseq et leno)). o |-> vo) **
    (exists* (vlr: et). lr |-> vlr) **
    (exists* (vmr: et). mr |-> vmr)
{
  let mut l: et = neg infinity; let mut m: et = zero;
  let mut i = 0sz;

  // Offline softmax portion
  // Generate elements with fpre and find max
  while (!i <^ lena) 
    invariant live i ** 
      (live l ** live m) **
      (exists* (va: (lseq et lena)). a |-> va) **
      frame
    decreases (lena - !i) {
    
    let x = fpre !i;
    write a !i x;
    m := fmax !m x;

    i := !i `SZ.add` 1sz;
  };

  i := 0sz;
  // Get sum of vector
  while (!i <^ lena)
    invariant live i **
      (exists* (va: (lseq et lena)). a |-> va) **
      (live l ** live m)
    decreases (lena - !i) {

    let x = read a !i;
    let s = exp (x `sub` !m);
    write a !i s;
    l := !l `add` s;

    i := !i `SZ.add` 1sz;
  };

  let m_n = fmax !mr !m;
  let l_n = !lr `mul` (exp (!mr `sub` m_n)) `add` !l `mul` (exp (!m `sub` m_n));

  i := 0sz;
  while (!i <^ leno) 
    invariant live i ** frame **
      (exists* (va: (lseq et lena)). a |-> va) **
      (exists* (vo: (lseq et leno)). o |-> vo)
    decreases (leno - !i) {

    let x = fpost a !i;
    let o_p = read o !i;
    let o_n: et = (o_p `mul` (!lr `div` l_n) `mul` (exp (!mr `sub` m_n))) `add` (x `mul` (exp (!m `sub` m_n)));
    write o !i o_n;

    i := !i `SZ.add` 1sz;
  };

  mr := m_n;
  lr := l_n;
  ()
} 

*)