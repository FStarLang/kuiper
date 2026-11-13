module KPlay.Proof.Optimized.DeltaMath

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module C = FStar.Classical
module Strided = KPlay.Proof.Optimized.Strided
module Algebra = KPlay.Proof.Optimized.Algebra

inline_for_extraction noextract
let vector = lseq real 128

inline_for_extraction noextract
let dot_term (a b:vector) (i:nat) : GTot real =
  if i < 128 then S.index a i *. S.index b i else 0.0R

inline_for_extraction noextract
let dot (a b:vector) : GTot real =
  rsum (S.init_ghost 128 (dot_term a b))

let dot_prefix (a b:vector) : Lemma (dot a b == prefix 128 (dot_term a b)) =
  prefix_init 128 (dot_term a b)

let strided_model (a b:vector)
  : Lemma (rsum (S.init_ghost 128 (Strided.term #128 (a,b))) == dot a b) =
  S.lemma_eq_elim (S.init_ghost 128 (Strided.term #128 (a,b)))
    (S.init_ghost 128 (dot_term a b))

inline_for_extraction noextract
let error (old key:vector) (value beta decay:real) : GTot real =
  (value -. decay *. dot old key) *. beta

inline_for_extraction noextract
let updated (old key:vector) (value beta decay:real) : GTot vector =
  S.init_ghost 128 (fun i ->
    S.index old i *. decay +. S.index key i *. error old key value beta decay)

let updated_cell (old key:vector) (value beta decay:real) (i:nat{i < 128})
  : Lemma (S.index (updated old key value beta decay) i ==
    S.index old i *. decay +. S.index key i *. error old key value beta decay) = ()

let distribute (decay old change key query:real)
  : Lemma (
      (old *. decay +. key *. change) *. query ==
      decay *. (old *. query) +. change *. (key *. query)) =
  Algebra.delta_readout decay old change key query

let updated_dot_term (old key query:vector) (value beta decay:real) (i:nat{i < 128})
  : Lemma (
      dot_term (updated old key value beta decay) query i ==
      decay *. dot_term old query i +. error old key value beta decay *. dot_term key query i) =
  updated_cell old key value beta decay i;
  distribute decay (S.index old i) (error old key value beta decay)
    (S.index key i) (S.index query i)

let updated_dot (old key query:vector) (value beta decay:real)
  : Lemma (
      dot (updated old key value beta decay) query ==
      decay *. dot old query +. error old key value beta decay *. dot key query) =
  let e = error old key value beta decay in
  let next = updated old key value beta decay in
  dot_prefix next query;
  dot_prefix old query;
  dot_prefix key query;
  C.forall_intro (updated_dot_term old key query value beta decay);
  prefix_ext 128 (dot_term next query)
    (fun i -> decay *. dot_term old query i +. e *. dot_term key query i);
  prefix_add 128 (fun i -> decay *. dot_term old query i) (fun i -> e *. dot_term key query i);
  prefix_scale 128 decay (dot_term old query);
  prefix_scale 128 e (dot_term key query)

noextract
noeq type interpretation = {
  old: vector;
  key: vector;
  query: vector;
  value: real;
  beta: real;
  decay: real;
}

inline_for_extraction noextract
let result (m:interpretation) : GTot vector =
  updated m.old m.key m.value m.beta m.decay

inline_for_extraction noextract
let change (m:interpretation) : GTot real =
  error m.old m.key m.value m.beta m.decay

inline_for_extraction noextract
let inputs (old key query:lseq f32 128) (value beta decay kq:f32) (m:interpretation) : prop =
  old %~ m.old /\ key %~ m.key /\ query %~ m.query /\
  value %~ m.value /\ beta %~ m.beta /\ decay %~ m.decay /\ kq %~ dot m.key m.query

let dots (old key query:lseq f32 128) (value beta decay kq stk sqv:f32)
  : Lemma
      (requires
        (forall (m:Strided.interpretation 128). Strided.inputs old key m ==>
          stk %~ rsum (S.init_ghost 128 (Strided.term m))) /\
        (forall (m:Strided.interpretation 128). Strided.inputs old query m ==>
          sqv %~ rsum (S.init_ghost 128 (Strided.term m))))
      (ensures forall (m:interpretation). inputs old key query value beta decay kq m ==>
        stk %~ dot m.old m.key /\ sqv %~ dot m.old m.query) =
  let related (m:interpretation{inputs old key query value beta decay kq m})
    : Lemma (stk %~ dot m.old m.key /\ sqv %~ dot m.old m.query) =
    strided_model m.old m.key;
    strided_model m.old m.query;
    assert (Strided.inputs old key (m.old,m.key));
    assert (Strided.inputs old query (m.old,m.query))
  in C.forall_intro related

let readout (m:interpretation)
  : Lemma (
      dot (result m) m.query ==
      m.decay *. dot m.old m.query +. change m *. dot m.key m.query) =
  updated_dot m.old m.key m.query m.value m.beta m.decay

let result_cell (i:nat{i < 128}) (m:interpretation)
  : Lemma (S.index (result m) i == S.index m.old i *. m.decay +. S.index m.key i *. change m) =
  updated_cell m.old m.key m.value m.beta m.decay i
