module Kuiper.MatMulTile.Layout4

open FStar.Mul
open FStar.Pervasives.Native
open NuSeq
open Pulse.Lib.BigStar
open Kuiper.MatMulTile.Layout

#push-options "--fuel 8 --ifuel 8"

let lemma_multiply4 (f: natlt 4 -> pos)
    : Lemma (multiply (mk_seq 4 f) == f 0 * f 1 * f 2 * f 3)
        [SMTPat (mk_seq 4 f)] = admit() // Why doesn't this work??

let lemma_multiply4_manual (i: seq pos): Lemma
    (requires len i == 4)
    (ensures multiply i == i.[0] * i.[1] * i.[2] * i.[3]) = ()

let permute_middle #a (i: seq a { len i == 4 }): (o: seq a { len o == 4 /\
    i.[0] == o.[0] /\ i.[1] == o.[2] /\
    i.[2] == o.[1] /\ i.[3] == o.[3]
}) = seq4 i.[0] i.[2] i.[1] i.[3]

let lemma_permute_middle_inverse #a (i: seq a { len i == 4 }): Lemma (permute_middle (permute_middle i) == i)
    = ()//lemma_eq_intro (permute_middle (permute_middle i)) i

let lemma_permute_preserves_multiply (i: seq pos { len i == 4 }): Lemma (multiply i == multiply (permute_middle i)) =
    // Why doesn't this work??
    assert (i.[0] * i.[1] * i.[2] * i.[3] == i.[0] * i.[2] * i.[1] * i.[3]); ()

let lemma_permute_preserves_lt (i: seq nat { len i == 4 }) (j: seq pos { len j == 4 }):
    // Why doesn't this work??
    Lemma (requires elementwise_smaller i j) (ensures elementwise_smaller (permute_middle i) (permute_middle j)) = admit()

let thread_id_to_idx (tcols trows bcols brows: pos)
    (tid: nat { tid < tcols * trows * bcols * brows }):
    (idx: nat { idx < tcols * trows * bcols * brows }) =
    let dims_in = seq4 tcols trows bcols brows in
    let coords_in = split_to_dims dims_in tid in
    lemma_permute_preserves_multiply dims_in;
    lemma_permute_preserves_lt coords_in dims_in;
    let dims_out = permute_middle dims_in in
    let coords_out = permute_middle coords_in in
    join_from_dims dims_out coords_out

let idx_to_thread_id (tcols trows bcols brows: pos)
    (idx: nat { idx < tcols * trows * bcols * brows }):
    (tid: nat { tid < tcols * trows * bcols * brows }) =
    let dims_in = seq4 tcols bcols trows brows in
    assert (tcols * trows * bcols * brows == tcols * bcols * trows * brows);
    let coords_in = split_to_dims dims_in idx in
    lemma_permute_preserves_multiply dims_in;
    lemma_permute_preserves_lt coords_in dims_in;
    let dims_out = permute_middle dims_in in
    let coords_out = permute_middle coords_in in
    join_from_dims dims_out coords_out

let lemma_permute (x y z w: int): Lemma (x * y * z * w == x * z * y * w) = ()
let elementwise_smaller_4 (outs: seq nat) (dims: seq pos { len dims == len outs }):
    Lemma //(requires len dims == 4 /\ outs.[0] < dims.[0] /\ outs.[1] < dims.[1]
          //                                      /\ outs.[2] < dims.[2] /\ outs.[3] < dims.[3])
          (ensures  elementwise_smaller outs dims) = admit()

// brittle and seems to fail in 4.13.3
#push-options "--z3cliopt 'smt.arith.nl=false' --z3version 4.8.5"

instance titi_permutation (tcols trows bcols brows: pos) : permutation (i: nat { i < tcols * trows * bcols * brows }) = {
    f = thread_id_to_idx tcols trows bcols brows;
    g = idx_to_thread_id tcols trows bcols brows;
    // proof : (x: a) -> (y: a) -> squash (f x == y <==> g y == x);
    proof = fun (x y: (i: nat { i < tcols * trows * bcols * brows })) -> (
        let f = thread_id_to_idx tcols trows bcols brows in
        let g = idx_to_thread_id tcols trows bcols brows in
        lemma_permute tcols trows bcols brows;
        calc (==>) {
            f x == y;
            ==> { calc (==) {
                g (f x);
                == {
                    let dims_in = seq4 tcols trows bcols brows in
                    lemma_multiply4_manual dims_in;
                    let coords_in = split_to_dims dims_in x in
                    lemma_permute_preserves_multiply dims_in;
                    lemma_permute_preserves_lt coords_in dims_in;
                    let dims_out = permute_middle dims_in in
                    let coords_out = permute_middle coords_in in
                    inverse_bwd dims_out coords_out;
                    lemma_permute_middle_inverse coords_in;
                    inverse_fwd dims_in x
                }
                x;
            } }
            g y == x;
        };
        calc (==>) {
            g y == x;
            ==> { calc (==) {
                f (g y);
                == {
                    let dims_in = seq4 tcols bcols trows brows in
                    lemma_multiply4_manual dims_in;
                    let coords_in = split_to_dims dims_in y in
                    lemma_permute_preserves_multiply dims_in;
                    lemma_permute_preserves_lt coords_in dims_in;
                    let dims_out = permute_middle dims_in in
                    let coords_out = permute_middle coords_in in
                    inverse_bwd dims_out coords_out;
                    lemma_permute_middle_inverse coords_in;
                    inverse_fwd dims_in y
                }
                y;
            } }
            f x == y;
        }
    )
}

#pop-options
#pop-options
