module Kuiper.MatMulTileF32.Layout

// open Pulse.Lib.Pervasives
open FStar.Mul
open FStar.Pervasives.Native
open NuSeq

unfold let multiply (dims: seq pos): pos = fold_r #pos dims 1 (fun x y -> x * y)

let elementwise_smaller (outs: seq nat) (dims: seq pos { len dims == len outs }): bool
    = fold_r (map (zip outs dims) (fun (x, y) -> x < y)) true (fun x y -> x && y)

/// Properties we don't get inside of `smt.arith.nl=false`
let lemma_div_eq (x y: int) (z: nonzero): Lemma (requires x == y) (ensures x / z == y / z) = ()
let lemma_div_sub1 (x: int) (z: pos): Lemma ((z * x - 1) / z == x - 1) = ()
let lemma_multiply_defn (dims: seq pos):
    Lemma (multiply dims == (if len dims = 0 then 1 else dims.[0] * multiply (pop_l dims)._2)) = ()

#push-options "--z3rlimit 20"
let lemma_ews_defn (outs: seq nat) (dims: seq pos { len dims == len outs }):
    Lemma (elementwise_smaller outs dims == (if len dims = 0 then true else
        outs.[0] < dims.[0] && elementwise_smaller (pop_l outs)._2 (pop_l dims)._2))
        [SMTPat (elementwise_smaller outs dims)]
= ()
#pop-options

let push_ews
    (dims: seq pos{len dims > 0})
    (out: nat{out < dims.[0]})
    (tail_outs: seq nat {len tail_outs == len dims - 1 /\ elementwise_smaller tail_outs (pop_l dims)._2})
    : (outs: seq nat {len outs == len dims /\ elementwise_smaller outs dims})
    =
    let outs = push_l out tail_outs in
    assert (len outs == len dims /\ (pop_l outs)._2 == tail_outs); // TODO: why is this needed
    outs

/// Nick suggested trying this to avoid NL arith non-termination, I don't think it helped that much since I still ran into
/// (probably some other kind of) non-termination in some cases.
#push-options "--z3cliopt 'smt.arith.nl=false'"

let lemma_div_idx (dims: seq pos { len dims <> 0 }) (idx: nat { idx < multiply dims }):
    Lemma (let (dim, dim_tl) = pop_l dims in idx / dim >= 0 /\ idx / dim < multiply dim_tl) =
        let dim = dims.[0] in
        FStar.Math.Lemmas.nat_over_pos_is_nat idx dim;
        calc (<=) {
            idx / dim;
            <= { FStar.Math.Lemmas.lemma_div_le idx (multiply dims - 1) dim }
            (multiply dims - 1) / dim;
            == { lemma_multiply_defn dims }
            (dim * multiply (pop_l dims)._2 - 1) / dim;
            == { lemma_div_sub1 (multiply (pop_l dims)._2) dim }
            multiply (pop_l dims)._2 - 1;
        }

/// 1st / 2 core functions, splits an index into dimensions.
/// Notice that the last dimension (i.e. when `length dims = 1`) isn't really used
/// since in that case `idx % dims.[0] == idx` and `idx / dims.[0] == 0`,
/// it is important in the `elementwise_smaller` postcondition though.
let rec split_to_dims (dims: seq pos) (idx: nat { idx < multiply dims }):
    Tot (outs: seq nat { len dims == len outs /\ elementwise_smaller outs dims }) (decreases len dims) =
    if len dims = 0 then seq0 else
        let (dim, tail_dims) = pop_l dims in
        let new_idx = idx / dim in
        lemma_div_idx dims idx;
        let tail_outs = (split_to_dims tail_dims new_idx) in
        push_ews dims (idx % dim) tail_outs

let lemma_mult_idx (dims: seq pos { len dims <> 0 })
    (out: nat { out < dims.[0] }) (new_idx: nat { new_idx < multiply (pop_l dims)._2 }):
    Lemma (out + dims.[0] * new_idx >= 0 /\ out + dims.[0] * new_idx < multiply dims) =
        let (dim, tail_dims) = pop_l dims in
        FStar.Math.Lemmas.nat_times_nat_is_nat dim new_idx;
        calc (<=) {
            out + dim * new_idx;
            <= { FStar.Math.Lemmas.lemma_mult_le_left dim new_idx (multiply tail_dims - 1) }
            out + dim * (multiply tail_dims - 1);
            <= {}
            dim - 1 + dim * (multiply tail_dims - 1);
            == { FStar.Math.Lemmas.distributivity_sub_right dim (multiply tail_dims) 1 }
            dim - 1 + dim * multiply tail_dims - dim;
            == { lemma_multiply_defn dims }
            multiply dims - 1;
        }

/// 2nd / 2 core functions, combines dimensions into an index.
/// Notice again that the last dimension isn't really used since in that
/// case `new_idx == 0`, though again it gets us the `idx < multiply dims` post.
#push-options "--z3rlimit 10 --retry 5" // flaky
let rec join_from_dims (dims: seq pos) (outs: seq nat { len dims == len outs /\ elementwise_smaller outs dims }):
    Tot (idx: nat { idx < multiply dims }) (decreases len dims) =
    if len dims = 0 then 0 else
        let (dim, tail_dims) = pop_l dims in
        let (out, tail_outs) = pop_l outs in
        let new_idx = join_from_dims tail_dims tail_outs in
        lemma_mult_idx dims out new_idx;
        new_idx * dim + out
#pop-options

/// The inductive proof revolves around the fact that `a = (a / b) * b + a % b`
let rec inverse_fwd (dims: seq pos) (idx: nat { idx < multiply dims }):
    Lemma (ensures join_from_dims dims (split_to_dims dims idx) == idx) (decreases len dims) =
    if len dims = 0 then () else
        let (dim, tail_dims) = pop_l dims in
        calc (==) {
            join_from_dims dims (split_to_dims dims idx);
            == {}
            (let (out, tail_outs) = pop_l (split_to_dims dims idx) in
                let new_idx = join_from_dims tail_dims tail_outs in
                lemma_mult_idx dims out new_idx;
                new_idx * dim + out);
            == { admit() } // TODO
            (lemma_div_idx dims idx;
                let tail_outs = split_to_dims tail_dims (idx / dim) in
                let new_idx = join_from_dims tail_dims tail_outs in
                lemma_mult_idx dims (idx % dim) new_idx;
                new_idx * dim + idx % dim);
            == { inverse_fwd tail_dims (idx / dim) }
            (lemma_div_idx dims idx;
                lemma_mult_idx dims (idx % dim) (idx / dim);
                (idx / dim) * dim + idx % dim);
            == { FStar.Math.Lemmas.euclidean_division_definition idx dim }
            idx;
        }

/// The inductive proof revolves around the fact that `(a + b * c) % c = a` and `(a + b * c) / c = b` (when `a < c`)
/// i.e. that `/` and `%` can pull out the respective components.
let rec inverse_bwd (dims: seq pos) (outs: seq nat { len dims == len outs /\ elementwise_smaller outs dims }):
    Lemma (ensures split_to_dims dims (join_from_dims dims outs) == outs) (decreases len dims) =
    if len dims = 0 then () else
        let (dim, tail_dims) = pop_l dims in
        let (out, _) = pop_l outs in
        calc (==) {
            split_to_dims dims (join_from_dims dims outs);
            == {}
            (let new_idx = join_from_dims tail_dims (pop_l outs)._2 in
                lemma_mult_idx dims out new_idx;
                let idx = new_idx * dim + out in
                lemma_div_idx dims idx;
                let tail_outs = split_to_dims tail_dims (idx / dim) in
                push_ews dims (idx % dim) tail_outs);
            == {
                let new_idx = join_from_dims tail_dims (pop_l outs)._2 in
                assume (new_idx * dim + out == out + dim * new_idx);
                FStar.Math.Lemmas.lemma_mod_plus out new_idx dim;
                FStar.Math.Lemmas.small_mod out dim;
                FStar.Math.Lemmas.lemma_div_plus out new_idx dim;
                FStar.Math.Lemmas.small_div out dim;
                admit() // TODO
            }
            (let new_idx = join_from_dims tail_dims (pop_l outs)._2 in
                lemma_mult_idx dims out new_idx;
                lemma_div_idx dims (new_idx * dim + out);
                let tail_outs = split_to_dims tail_dims new_idx in
                push_ews dims out tail_outs);
            == { inverse_bwd tail_dims (pop_l outs)._2 }
            push_ews dims out (pop_l outs)._2;
            == { admit() } // TODO
            outs;
        }

/// End of general section
#pop-options
