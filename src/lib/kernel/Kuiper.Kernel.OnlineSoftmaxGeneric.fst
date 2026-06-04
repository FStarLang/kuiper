(* An attempt at generalizing online softmax so it can be fused with other kernels. *)
module Kuiper.Kernel.OnlineSoftmaxGeneric

#lang-pulse
open Kuiper
open Kuiper.DotProd
open Kuiper.Spec.Softmax
open Kuiper.Seq.Common

let exp_sum (s : seq real{len s > 0}) : r:real{r >. 0.0R} =
  rsum (seq_map rexp s)

let adjust_factor (s : seq real{len s > 0}) (x : real) : real =
  exp_sum s
  /.
  (exp_sum s +. rexp x)

let rec seq_max (s : seq real{len s > 0})
  : Tot real (decreases len s)
  =
  let SCons h t = view_seq s in
  match view_seq t with
  | SNil -> h
  | SCons _ _ -> rmax h (seq_max t)

let rec seq_max_cons_lem (s : seq real{len s > 0}) (x : real)
  : Lemma (ensures seq_max (s @+ seq![x]) == rmax x (seq_max s))
          (decreases len s)
          [SMTPat (seq_max (s @+ seq![x]))]
  = let SCons h t = view_seq s in
    assert Seq.equal (s @+ seq![x]) (Seq.cons h (t @+ seq![x]));
    match view_seq t with
    | SNil ->
      lem_rmax_comm x (seq_max s)
    | SCons _ _ ->
      calc (==) {
        seq_max (s @+ seq![x]);
        == {}
        rmax h (seq_max (t @+ seq![x]));
        == { seq_max_cons_lem t x }
        rmax h (rmax x (seq_max t));
        == { lem_rmax_assoc h x (seq_max t)}
        rmax (rmax h x) (seq_max t);
        == {}
        rmax x (seq_max s);
      }

(*
desired client code sketch:

va vb: float seq N
ra rb: real seq N { va =~ ra, vb =~ rb }
a b: array N

let state: osmx_st a = online_softmax_init a ();
let res: float = 0.0f;
while (i < N)
  invariant osmx_state_ok state a i **
    !res =~ state.dr *. osmx_reduce (fun fxi i -> fxi `mul` vb @! i) (fun fxi i -> fxi *. rb @! i) ...
  {

  state', fxi, adj = osmx_step a i ();
  res := !res * adj + fxi * (b @! i);
  state' := state;

  osmx_adj_lem res a ...;
}
// smt should automatically be able to show then !res =~ osmx_reduce ... , but maybe it would fail to cancel out the / state.d.
// we can also just force people to use the divides in the invariant.
res := !res / state.d;
client_side_dotprod_ok_proof ();

*)

let get = Some?.v

[@@erasable]
noeq
type osmx_rst : Type0 = {
  s : seq real;
  d : real;
  m : option real;
  #m_empty_ok : Seq.length s > 0 <==> Some? m;
}

let ok_rst (rs: osmx_rst): prop =
  (Seq.length rs.s > 0 ==> get rs.m == seq_max rs.s) /\
  (Seq.length rs.s > 0 ==> rs.d >. 0.0R /\ rs.d == exp_sum rs.s /. rexp (get rs.m))

noeq
type osmx_st (et: Type0): Type0 = {
  d : et;
  m : et;
}

instance st_approx (et:Type) {| scalar et, floating et, real_like et |}
  : can_approximate (osmx_st et) osmx_rst = {
    approximates = (fun (l : osmx_st et) (r : osmx_rst) ->
      // (Seq.length r.s == 0 ==> l.d == zero /\ l.m == neg infinity) /\
      Seq.length r.s > 0 ==>
        l.d %~ r.d /\ l.m %~ get r.m);
  }

instance tup3_approx (a0 a1 a2 b0 b1 b2: Type) {| can_approximate a0 b0, can_approximate a1 b1, can_approximate a2 b2 |}
  : can_approximate (a0 & a1 & a2) (b0 & b1 & b2) = {
    approximates = (fun (at : (a0 & a1 & a2)) (bt: (b0 & b1 & b2)) ->
      let (at0,at1,at2) = at in
      let (bt0,bt1,bt2) = bt in
      (at0 %~ bt0) /\ (at1 %~ bt1) /\ (at2 %~ bt2));
  }

// either put in postcondition "osmx_fst s.t. s == []" or unfold def'n
let osmx_rinit () : osmx_rst = { s = seq![]; d = 0.0R; m = None; }

let osmx_rstep0 (s : osmx_rst) (x : real) : osmx_rst =
  let m' = (match s.m with | None -> x | Some m -> rmax m x) in
  {
    s = s.s @+ seq![x];
    m = Some m';
    d = (match s.m with
         | None   ->                          rexp (x -. m') // equal to one...
         | Some m -> s.d *. rexp (m -. m') +. rexp (x -. m'))
  }

let osmx_rstep (s : osmx_rst) (x : real) : GTot (osmx_rst & real & real) =
  let s' = osmx_rstep0 s x in
  (s', rexp (x -. get s'.m), (if None? s.m then 1.0R else rexp (get s.m -. get s'.m)))

let lemma_osmx_rstep (s : osmx_rst) (x : real)
  : Lemma (requires ok_rst s)
          (ensures  ok_rst (osmx_rstep0 s x))
          [SMTPat (ok_rst (osmx_rstep0 s x))]
  = admit()

(*
let online_softmax_step (xi: real) (m d: real):
  (real & real & real & real) =
  let m' = rmax m xi in
  let fxi = rexp (xi -. m') in
  let adj = rexp (m -. m') in
  let d' = d *. adj +. fxi in
  (m', d', fxi, adj)

val lem_online_softmax_step_correct (#n: pos) (x : lseq real n) (i: pos{i < n})
  (m d: real):
  Lemma (requires online_softmax_invariant x i m d)
        (ensures
          (let (m',d',fxi,_) = online_softmax_step (x @! i) m d in
          online_softmax_invariant x (i+1) m' d' /\
          fxi /. d' == softmax_real (seq_take (i+1) x) @! i))
*)

// let lem_osmx_rstep_correct

inline_for_extraction noextract
fn osmx_init #et {| scalar et, floating et,real_like et |}
  ()
  returns (s: osmx_st et {s %~ osmx_rinit ()})
{
  ({ d = zero #et; m = neg #et infinity; })
}

inline_for_extraction noextract
fn osmx_step #et {| scalar et, floating et, real_like et |}
  (s : osmx_st et)
  (rs: osmx_rst {s %~ rs})
  (x : et)
  (rx : real)
  requires
    pure (x %~ rx)
  returns
    res : (res : (osmx_st et & et & et) { res %~ osmx_rstep rs rx })
{
  let Mkosmx_st d m = s;
  let m' = fmax m x;
  let fx = exp (x `sub` m');
  let adj = exp (m `sub` m');
  let d' = (d `mul` adj) `add` fx;
  let s' = {d=d'; m=m'};
  assume pure False;
  (s', fx, adj)
}

// (* k is a continuation: what to do with e^xi where xi is an element of the softmax'd vector *)
// let fused_k_ty (n: pos) = real -> natlt n -> real
// let fused_k_comm_div_ty (#n: pos) (k: fused_k_ty n) =
//   (nr: real) -> (dr: real { dr =!= 0.0R } ) -> (j: natlt n) -> (k nr j) /. dr == k (nr /. dr) j

// (* how to reduce each result of applying k *)
// let fused_op_ty = real -> real -> real
// let fused_op_distrib_mul_ty (op: fused_op_ty) =
//   (a: real) -> (b: real) -> (r: real) -> (a `op` b) *. r == (a *. r) `op` (b *. r)

// unfold
// let online_softmax_fuse_reduce (#n: pos)
//   (x: lseq real n)
//   (k: fused_k_ty n)  (* continuation: what to do with e^xi *)
//   (op: fused_op_ty)
//   (i: natle n) =
//     if i == 0 then 0.0R
//     else seq_fold_left op 0.0R (seq_mapi (softmax_real (seq_take i x)) k)

// val lem_online_softmax_fuse_by_adj (#n : pos)
//   (x : lseq real n)
//   (i: pos {i < n})
//   (m d: real { online_softmax_invariant x i m d })
//   (k: fused_k_ty n)
//   (#k_comm_div: fused_k_comm_div_ty k)
//   (op: fused_op_ty)
//   (#op_distrib_mul: fused_op_distrib_mul_ty op)
//   (r : real):
//   Lemma
//     (requires r /. d == online_softmax_fuse_reduce x k op i)
//     (ensures
//       (let (m',d',fxi,adj) = online_softmax_step (x @! i) m d in
//       ((r *. adj) `op` (k fxi i)) /. d' == online_softmax_fuse_reduce x k op (i+1)))

let rst_d (r : osmx_rst) : real = r.d

let cancel (a b c : real) :
  Lemma (requires a == b *. c /\ b =!= 0.0R)
        (ensures  a /. b == c)
  = ()

#set-options "--split_queries always"

// (a,b) -> softmax(a) * b
inline_for_extraction noextract
fn osmx_dotprod
  (#et : Type0) {| scalar et, floating et, real_like et, floating_real_like et |}
  (len : szp)
  (a b : larray et len)
  (va vb : erased (lseq et   len))
  (ra rb : erased (lseq real len))
  preserves a |-> va
  preserves b |-> vb
  requires  pure (va %~ ra /\ vb %~ rb)
  returns v : et
  ensures pure (v %~ seq_dotprod (softmax_real ra) rb)
{
  open Pulse.Lib.Array;
  Pulse.Lib.Array.pts_to_len a;
  Pulse.Lib.Array.pts_to_len b;

  let mut st  : osmx_st et = osmx_init #et ();
  let mut rst : osmx_rst = osmx_rinit ();
  let mut i : szle len = 0sz;
  let mut r : et = zero #et;
  let mut rr : real = 0.0R;
  while (!i <^ len)
    invariant live i
    invariant live st ** live rst ** pure (!st %~ !rst)
    invariant live r ** live rr **
      pure (!r %~ !rr /\ ok_rst !rst /\
            (!rst).s == seq_take !i ra /\
            !rr == rst_d !rst *. seq_dotprod #(!i) (softmax_real (seq_take !i ra)) (seq_take !i rb))
    decreases (len - !i)
  {
    let rst0 = !rst;

    let stv, fai, adj = osmx_step !st !rst a.(!i) (Seq.index ra !i);

    assert pure (stv %~ (osmx_rstep !rst (Seq.index ra !i))._1);
    assert pure (stv %~ osmx_rstep0 !rst (Seq.index ra !i));

    assert pure (fai %~ (osmx_rstep !rst (Seq.index ra !i))._2);

    let rst' = osmx_rstep0 !rst (Seq.index ra !i);

    st  := stv;
    rst := rst';
    assert pure (!st %~ !rst);
    r  := add (!r `mul` adj) (fai `mul` b.(!i));
    assume pure (Some? (rst0 <: osmx_rst).m);
    rr := !rr *. (rexp (get (rst0 <: osmx_rst).m) -. rexp (get rst'.m)) +.
           rexp (Seq.index ra !i -. get (rst'.m)) *. Seq.index rb !i;
    assume pure (!r %~ !rr);

    i := !i +^ 1sz;

    assert pure (!st %~ !rst);
    assert pure (!r %~ !rr);
    assert pure (ok_rst !rst);
    assume pure ((!rst).s == seq_take !i ra);
    assume pure (!rr == rst_d !rst *. seq_dotprod #(!i) (softmax_real (seq_take !i ra)) (seq_take !i rb));

    ()
  };
  assert pure (!i == len);
  assert pure (seq_take len ra == ra);
  assert pure (seq_take len rb == rb);
  assert pure (seq_dotprod #len (softmax_real (seq_take len ra)) (seq_take len rb) == seq_dotprod (softmax_real ra) rb);
  assert pure (!rr == rst_d !rst *. seq_dotprod (softmax_real ra) rb);
  assert pure (rst_d !rst =!= 0.0R);
  cancel !rr (rst_d !rst) (seq_dotprod (softmax_real ra) rb);
  assert pure (!rr /. rst_d !rst == seq_dotprod (softmax_real ra) rb);
  let res = !r `div` (!st).d;
  // assert pure (!r %~ !rr);
  // assert pure (!st %~ !rst);
  // assert pure ((!st).d %~ rst_d !rst);
  // div_approx !r (!st).d !rr (rst_d !rst);
  // assert pure ((div !r (!st).d) %~ (!rr /. rst_d !rst));
  assert pure (res %~ seq_dotprod (softmax_real ra) rb);
  res
}

let osmx_dotprod_f32 len = osmx_dotprod #f32 len
