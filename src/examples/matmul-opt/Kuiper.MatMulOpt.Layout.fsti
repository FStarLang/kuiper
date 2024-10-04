module Kuiper.MatMulOpt.Layout

open Pulse.Lib.Pervasives
open FStar.Mul
open FStar.Pervasives.Native
module SZ = FStar.SizeT
// open Kuiper.MatMulOpt.Kernel
open Pulse.Lib.BigStar

// let block_id (rows columns: nat) (tid: nat { tid < rows * columns }) : GTot nat = tid / (SZ.v tpb)
// let thread_id (rows columns: nat) (tid: nat { tid < rows * columns }) : GTot nat = tid % (SZ.v tpb)

// let lemma_div_lt (a b:int) (d:pos { b % d == 0 }):
//   Lemma (requires (a < b))
//         (ensures  (a / d < b / d)) = ()

// let lemma_mod_mult (a b: int) (c d: pos):
//   Lemma (requires (a % c == 0 /\ b % d == 0))
//         (ensures  ((a * b) % (c * d) == 0)) = admit()

// let lemma_div_mult_lt (a b: int) (d: pos):
//   Lemma (requires (a < b * d))
//         (ensures  (a / d < b)) = ()

// let lemma_mult_le (a b: int) (d: nat):
//   Lemma (requires (a <= b))
//         (ensures  (a * d <= b * d)) = ()

// let lemma_mult_distr (a b: int) (d: nat):
//   Lemma (ensures  (a * d + b * d == (a + b) * d)) = ()

// let thread_id_to_idx (rows columns: (i: pos { i % blocksize == 0 })) (tid: nat { tid < rows * columns }): GTot (r: nat { r < rows * columns })
//   = admit(); lemma_mod_mult rows columns (blocksize) (blocksize);
//     lemma_div_lt tid (rows * columns) (SZ.v tpb);
//     let block_id:   (i: nat { i < (rows / blocksize) * (columns / blocksize) }) = tid / (SZ.v tpb) in
//     let block_id_x: (i: nat { i <= columns / blocksize - 1 })                           = block_id % (columns / blocksize) in // which row of blocks
//     lemma_div_mult_lt block_id (rows / blocksize) (columns / blocksize);
//     let block_id_y: (i: nat { i <= rows / blocksize - 1 })                              = block_id / (columns / blocksize) in // which column of blocks
//     let block_offset: nat = block_id_x * blocksize + block_id_y * blocksize * columns in
//     calc (<=) {
//       block_offset <: int;
//       == {}
//       block_id_x * blocksize + block_id_y * blocksize * columns;
//       <= { lemma_mult_le block_id_x (columns / blocksize - 1) (blocksize) }
//       ((columns / blocksize - 1) * blocksize + block_id_y * blocksize * columns);
//       <= { lemma_mult_le block_id_y (rows / blocksize - 1) (blocksize * columns) }
//       ((columns / blocksize - 1) * blocksize + (rows / blocksize - 1) * blocksize * columns);
//     };
//     let thread_id:   (i: nat { i < SZ.v tpb })       = tid % (SZ.v tpb) in
//     let thread_id_x: (i: nat { i <= blocksize - 1 }) = thread_id % blocksize in
//     let thread_id_y: (i: nat { i <= blocksize - 1 }) = thread_id / blocksize in
//     let thread_offset: nat = thread_id_x + thread_id_y * columns in
//     calc (<=) {
//       thread_offset <: int;
//       == {}
//       thread_id_x + thread_id_y * columns;
//       <= {}
//       (blocksize - 1) + thread_id_y * columns;
//       <= { lemma_mult_le thread_id_y (blocksize - 1) columns }
//       (blocksize - 1) + (blocksize - 1) * columns;
//     };
//     calc (<) {
//       block_offset + thread_offset;
//       <= {}
//       ((columns / blocksize - 1) * blocksize + (rows / blocksize - 1) * blocksize * columns) + ((blocksize - 1) + (blocksize - 1) * columns);
//       == {}
//       1 * columns + (rows - blocksize) * columns - 1 + (blocksize - 1) * columns;
//       == {
//         lemma_mult_distr 1 (rows - blocksize) columns;
//         lemma_mult_distr (1 + rows - blocksize) (blocksize - 1) columns
//       }
//       (1 + rows - blocksize + blocksize - 1) * columns - 1;
//       == {}
//       rows * columns - 1;
//       < {}
//       rows * columns;
//     };
//     // assert (block_offset + thread_offset < (rows - blocksize + 1) * columns - blocksize + blocksize + (blocksize - 1) * columns);
//     // assert (block_offset + thread_offset < rows * columns);
//     block_offset + thread_offset

//     // ((tid / (SZ.v tpb)) % (columns / blocksize)) * blocksize
//     //     + ((tid / (SZ.v tpb)) / (columns / blocksize)) * blocksize * columns
//     //     + (tid % (SZ.v tpb)) % blocksize
//     //     + ((tid % (SZ.v tpb)) / blocksize) * columns;

/// Defined these to try and get `seq_tail_lemma` for free, but that didn't work so had to write the lemma anyway
let seq_head #a (s: seq a { FStar.len s <> 0 }): (r: a { FStar.Seq.index s 0 == r }) = FStar.Seq.index s 0
let seq_tail #a (s: seq a { FStar.len s <> 0 }):
    (r: seq a { FStar.len r == FStar.len s - 1 /\ FStar.Seq.cons (seq_head s) r == s })
    = FStar.Seq.slice s 1 (FStar.len s)

let seq_tail_lemma #a (hd: a) (tl: seq a):
    Lemma (seq_tail (FStar.Seq.cons hd tl) == tl)
    [SMTPat (seq_tail (FStar.Seq.cons hd tl))] = admit()

/// Fold operations we need, I imagine there's a fold in std somewhere but didn't use that for now
let rec multiply (dims: erased (seq pos)): Tot (erased pos) (decreases FStar.len dims) =
    if FStar.len dims = 0 then 1 else seq_head dims * multiply (seq_tail dims)

let rec elementwise_smaller (outs: erased (seq nat)) (dims: erased (seq pos) { FStar.len dims == FStar.len outs }):
    Tot prop (decreases FStar.len dims) =
    if FStar.len dims = 0 then true else
        seq_head outs < seq_head dims /\ elementwise_smaller (seq_tail outs) (seq_tail dims)

/// Properties we don't get inside of `smt.arith.nl=false`
let lemma_div_eq (x y: int) (z: nonzero): Lemma (requires x == y) (ensures x / z == y / z) = ()
let lemma_div_sub1 (x: int) (z: pos): Lemma ((z * x - 1) / z == x - 1) = ()
let lemma_multiply_defn (dims: erased (seq pos)):
    Lemma (reveal (multiply dims) == (if FStar.len dims = 0 then 1 else seq_head dims * multiply (seq_tail dims))) = ()

/// Nick suggested trying this to avoid NL arith non-termination, I don't think it helped that much since I still ran into
/// (probably some other kind of) non-termination in some cases.
#push-options "--z3cliopt 'smt.arith.nl=false'"

let lemma_div_idx (dims: erased (seq pos) { FStar.len dims <> 0 }) (idx: erased nat { idx < multiply dims }):
    Lemma (idx / seq_head dims >= 0 /\ idx / seq_head dims < multiply (seq_tail dims)) =
        let dim = seq_head dims in
        FStar.Math.Lemmas.nat_over_pos_is_nat idx dim;
        calc (<=) {
            idx / dim;
            <= { FStar.Math.Lemmas.lemma_div_le idx (multiply dims - 1) dim }
            (multiply dims - 1) / dim;
            == { lemma_multiply_defn dims }
            (dim * multiply (seq_tail dims) - 1) / dim;
            == { lemma_div_sub1 (multiply (seq_tail dims)) dim }
            multiply (seq_tail dims) - 1;
        }

/// 1st / 2 core functions, splits an index into dimensions.
/// Notice that the last dimension (i.e. when `length dims = 1`) isn't really used
/// since in that case `idx % seq_head dims == idx` and `idx / seq_head dims == 0`,
/// it is important in the `elementwise_smaller` postcondition though.
let rec split_to_dims (dims: erased (seq pos)) (idx: erased nat { idx < multiply dims }):
    Tot (outs: erased (seq nat) { FStar.len dims == FStar.len outs /\ elementwise_smaller outs dims }) (decreases FStar.len dims) =
    if FStar.len dims = 0 then FStar.Seq.empty else
        let new_idx = idx / seq_head dims in
        lemma_div_idx dims idx;
        FStar.Seq.cons (idx % seq_head dims) (split_to_dims (seq_tail dims) new_idx)

let lemma_mult_idx (dims: erased (seq pos) { FStar.len dims <> 0 })
    (out: erased nat { out < seq_head dims }) (new_idx: erased nat { new_idx < multiply (seq_tail dims) }):
    Lemma (out + seq_head dims * new_idx >= 0 /\ out + seq_head dims * new_idx < multiply dims) =
        let dim = seq_head dims in
        let tail_dims = seq_tail dims in
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
let rec join_from_dims (dims: erased (seq pos)) (outs: erased (seq nat) { FStar.len dims == FStar.len outs /\ elementwise_smaller outs dims }):
    Tot (idx: erased nat { idx < multiply dims }) (decreases FStar.len dims) =
    if FStar.len dims = 0 then 0 else
        let new_idx = join_from_dims (seq_tail dims) (seq_tail outs) in
        lemma_mult_idx dims (seq_head outs) new_idx;
        seq_head outs + seq_head dims * new_idx

/// The inductive proof revolves around the fact that `a = (a / b) * b + a % b`
let rec inverse_fwd (dims: erased (seq pos)) (idx: erased nat { idx < multiply dims }):
    Lemma (ensures join_from_dims dims (split_to_dims dims idx) == idx) (decreases FStar.len dims) =
    if FStar.len dims = 0 then () else
        let dim = seq_head dims in
        let tail_dims = seq_tail dims in
        calc (==) {
            join_from_dims dims (split_to_dims dims idx);
            == {}
            hide (lemma_div_idx dims idx;
                let tail_outs = split_to_dims tail_dims (idx / dim) in
                let new_idx = join_from_dims tail_dims tail_outs in
                lemma_mult_idx dims (idx % dim) new_idx;
                idx % dim + dim * new_idx);
            == { inverse_fwd tail_dims (idx / dim) }
            hide (lemma_div_idx dims idx;
                lemma_mult_idx dims (idx % dim) (idx / dim);
                idx % dim + dim * (idx / dim));
            == { FStar.Math.Lemmas.euclidean_division_definition idx dim }
            idx;
        }

/// The inductive proof revolves around the fact that `(a + b * c) % c = a` and `(a + b * c) / c = b` (when `a < c`)
/// i.e. that `/` and `%` can pull out the respective components.
let rec inverse_bwd (dims: erased (seq pos)) (outs: erased (seq nat) { FStar.len dims == FStar.len outs /\ elementwise_smaller outs dims }):
    Lemma (ensures split_to_dims dims (join_from_dims dims outs) == outs) (decreases FStar.len dims) =
    if FStar.len dims = 0 then FStar.Seq.lemma_eq_intro FStar.Seq.empty outs else
        let dim = seq_head dims in
        let tail_dims = seq_tail dims in
        let out = seq_head outs in
        calc (==) {
            split_to_dims dims (join_from_dims dims outs);
            == {}
            hide (let new_idx = join_from_dims tail_dims (seq_tail outs) in
                lemma_mult_idx dims out new_idx;
                let idx = out + dim * new_idx in
                lemma_div_idx dims idx;
                let tail_outs = split_to_dims tail_dims (idx / dim) in
                FStar.Seq.cons (idx % dim) tail_outs);
            == {
                let new_idx = join_from_dims tail_dims (seq_tail outs) in
                FStar.Math.Lemmas.lemma_mod_plus out new_idx dim;
                FStar.Math.Lemmas.small_mod out dim;
                FStar.Math.Lemmas.lemma_div_plus out new_idx dim;
                FStar.Math.Lemmas.small_div out dim
            }
            hide (let new_idx = join_from_dims tail_dims (seq_tail outs) in
                lemma_mult_idx dims out new_idx;
                lemma_div_idx dims (out + dim * new_idx);
                let tail_outs = split_to_dims tail_dims new_idx in
                FStar.Seq.cons out tail_outs);
            == { inverse_bwd tail_dims (seq_tail outs) }
            outs;
        }

/// End of general section
#pop-options

/// Specialise to 4 dimensions `[tcols trows bcols brows]` <--> `[tcols bcols trows brows]`,
/// where `tcols` is the width of the thread block (usually = warp-dim, usually = 32), `trows` is the height of the thread block
/// which must be `<= 1024 / tcols` since that's the maximum threads per block, `bcols` is the width the output matrix divided by
/// `tcols`, and `brows` is the height of the output matrix divided by `trows`.

let seq_0: (r: seq pos { FStar.len r == 0 /\ reveal (multiply r) == 1 }) = FStar.Seq.empty
let seq_1 (x: pos): (r: seq pos { FStar.len r == 1 /\ reveal (multiply r) == x /\ FStar.Seq.index r 0 == x }) =
    let r = FStar.Seq.cons x seq_0     in lemma_multiply_defn r; assert (reveal (multiply r) == seq_head r * 1); r
let seq_2 (x y: pos): (r: seq pos { FStar.len r == 2 /\ reveal (multiply r) == x * y /\ FStar.Seq.index r 0 == x /\ FStar.Seq.index r 1 == y }) =
    let r = FStar.Seq.cons x (seq_1 y) in lemma_multiply_defn r; assert (reveal (multiply r) == seq_head r * y); r
let seq_3 (x y z: pos): (r: seq pos {
    FStar.len r == 3 /\ reveal (multiply r) == x * y * z /\ FStar.Seq.index r 0 == x /\ FStar.Seq.index r 1 == y /\ FStar.Seq.index r 2 == z
}) = let r = FStar.Seq.cons x (seq_2 y z) in lemma_multiply_defn r; assert (reveal (multiply r) == seq_head r * (y * z)); r
let seq_4 (x y z w: pos): (r: seq pos {
    FStar.len r == 4 /\ reveal (multiply r) == x * y * z * w /\ FStar.Seq.index r 0 == x /\ FStar.Seq.index r 1 == y /\ FStar.Seq.index r 2 == z /\ FStar.Seq.index r 3 == w
}) = let r = FStar.Seq.cons x (seq_3 y z w) in lemma_multiply_defn r; assert (reveal (multiply r) == seq_head r * (y * (z * w))); r

let permute_middle #a (i: seq a { FStar.len i == 4 }): (o: seq a { FStar.len o == 4 /\
    FStar.Seq.index i 0 == FStar.Seq.index o 0 /\ FStar.Seq.index i 1 == FStar.Seq.index o 2 /\
    FStar.Seq.index i 2 == FStar.Seq.index o 1 /\ FStar.Seq.index i 3 == FStar.Seq.index o 3
}) = let (i0, i1, i2, i3) = (FStar.Seq.index i 0, FStar.Seq.index i 1, FStar.Seq.index i 2, FStar.Seq.index i 3) in
     FStar.Seq.cons i0 (FStar.Seq.cons i2 (FStar.Seq.cons i1 (FStar.Seq.cons i3 FStar.Seq.empty)))

let lemma_permute_middle_inverse #a (i: seq a { FStar.len i == 4 }): Lemma (permute_middle (permute_middle i) == i)
    = FStar.Seq.lemma_eq_intro (permute_middle (permute_middle i)) i
let lemma_permute_preserves_multiply (i: seq pos { FStar.len i == 4 }): Lemma (multiply i == multiply (permute_middle i)) = ()
let lemma_permute_preserves_lt (i: seq nat { FStar.len i == 4 }) (j: seq pos { FStar.len j == 4 }):
    Lemma (requires elementwise_smaller i j) (ensures elementwise_smaller (permute_middle i) (permute_middle j)) = ()

let thread_id_to_idx (tcols trows bcols brows: pos)
    (tid: erased nat { tid < tcols * trows * bcols * brows }):
    (idx: erased nat { idx < tcols * trows * bcols * brows }) =
    let dims_in = seq_4 tcols trows bcols brows in
    let coords_in = split_to_dims dims_in tid in
    lemma_permute_preserves_multiply dims_in;
    lemma_permute_preserves_lt coords_in dims_in;
    let dims_out = permute_middle dims_in in
    let coords_out = permute_middle coords_in in
    join_from_dims dims_out coords_out

let idx_to_thread_id (tcols trows bcols brows: pos)
    (idx: erased nat { idx < tcols * trows * bcols * brows }):
    (tid: erased nat { tid < tcols * trows * bcols * brows }) =
    let dims_in = seq_4 tcols bcols trows brows in
    let coords_in = split_to_dims dims_in idx in
    lemma_permute_preserves_multiply dims_in;
    lemma_permute_preserves_lt coords_in dims_in;
    let dims_out = permute_middle dims_in in
    let coords_out = permute_middle coords_in in
    join_from_dims dims_out coords_out

let lemma_permute (x y z w: int): Lemma (x * y * z * w == x * z * y * w) = ()
let elementwise_smaller_4 (outs: erased (seq nat)) (dims: erased (seq pos) { FStar.len dims == FStar.len outs }):
    Lemma //(requires FStar.len dims == 4 /\ FStar.Seq.index outs 0 < FStar.Seq.index dims 0 /\ FStar.Seq.index outs 1 < FStar.Seq.index dims 1
          //                                      /\ FStar.Seq.index outs 2 < FStar.Seq.index dims 2 /\ FStar.Seq.index outs 3 < FStar.Seq.index dims 3)
          (ensures  elementwise_smaller outs dims) = admit()

#push-options "--z3cliopt 'smt.arith.nl=false'"

instance titi_permutation (tcols trows bcols brows: pos) : permutation (i: erased nat { i < tcols * trows * bcols * brows }) = {
    f = thread_id_to_idx tcols trows bcols brows;
    g = idx_to_thread_id tcols trows bcols brows;
    // proof : (x: a) -> (y: a) -> squash (f x == y <==> g y == x);
    proof = fun (x y: (i: erased nat { i < tcols * trows * bcols * brows })) -> (
        let f = thread_id_to_idx tcols trows bcols brows in
        let g = idx_to_thread_id tcols trows bcols brows in
        lemma_permute tcols trows bcols brows;
        calc (==>) {
            f x == y;
            ==> { calc (==) {
                g (f x);
                == {
                    let dims_in = seq_4 tcols trows bcols brows in
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
                    let dims_in = seq_4 tcols bcols trows brows in
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

// let thread_id_to_dims (tcols trows bcols brows: pos)
//     (tid: erased nat { tid < tcols * trows * bcols * brows }):
//     (x: erased (nat & nat & nat & nat) { (reveal x)._1 < tcols /\ (reveal x)._2 < trows /\ (reveal x)._3 < bcols /\ (reveal x)._4 < brows }) =
//     let dims = FStar.Seq.cons tcols (FStar.Seq.cons trows (FStar.Seq.cons bcols (FStar.Seq.cons brows FStar.Seq.empty))) in
//     assume (reveal (multiply dims) == tcols * trows * bcols * brows);
//     let outs = split_to_dims dims tid in
//     (seq_head outs, FStar.Seq.index outs 1, FStar.Seq.index outs 2, FStar.Seq.index outs 3)

// let dims_to_thread_id (tcols trows bcols brows: pos)
//     (x: erased (nat & nat & nat & nat) { (reveal x)._1 < tcols /\ (reveal x)._2 < trows /\ (reveal x)._3 < bcols /\ (reveal x)._4 < brows }):
//     (tid: erased nat { tid < tcols * trows * bcols * brows }) =
//     let dims = FStar.Seq.cons tcols (FStar.Seq.cons trows (FStar.Seq.cons bcols (FStar.Seq.cons brows FStar.Seq.empty))) in
//     let outs = FStar.Seq.cons (reveal x)._1 (FStar.Seq.cons (reveal x)._2 (FStar.Seq.cons (reveal x)._3 (FStar.Seq.cons (reveal x)._4 FStar.Seq.empty))) in
//     let tid = join_from_dims dims outs in
//     assume (reveal (multiply dims) == tcols * trows * bcols * brows);
//     tid

// let idx_to_dims (tcols trows bcols brows: pos)
//     (r: erased nat { r < trows * tcols * brows * bcols }):
//     (x: erased (nat & nat & nat & nat) { (reveal x)._1 < tcols /\ (reveal x)._2 < trows /\ (reveal x)._3 < bcols /\ (reveal x)._4 < brows }) =
//     let thread_id_x = r % tcols in
//     let block_id = r / tcols in
//     let block_id_x = block_id % bcols in
//     let idx_y = block_id / bcols in
//     let thread_id_y = idx_y % trows in
//     let block_id_y = idx_y / trows in
//     (thread_id_x, thread_id_y, block_id_x, block_id_y)

// let dims_to_idx (tcols trows bcols brows: pos)
//     (x: erased (nat & nat & nat & nat) { (reveal x)._1 < tcols /\ (reveal x)._2 < trows /\ (reveal x)._3 < bcols /\ (reveal x)._4 < brows }):
//     (r: erased nat { r < trows * tcols * brows * bcols }) =
//     let (thread_id_x, thread_id_y, block_id_x, block_id_y) = x in
//     let idx_y = thread_id_y + block_id_y * trows in
//     let block_id = block_id_x + idx_y * bcols in
//     let r = thread_id_x + block_id * tcols in
//     admit(); r

// let thread_id_to_idx_2 (blocksize: pos) (rows columns: (i: pos { i % blocksize == 0 })) (tid: erased nat { tid < rows * columns }): (r: erased nat { r < rows * columns })
//   = admit();
//     let thread_id_x: nat = tid % blocksize in
//     let thread_row: nat = tid / blocksize in
//     let thread_id_y: nat = thread_row % blocksize in
//     let bid: nat = thread_row / blocksize in
//     let block_id_x: nat = bid % (columns / blocksize) in
//     let block_id_y: nat = bid / (columns / blocksize) in
//     calc (<=) {
//         block_id_y <: int;
//         == {}
//         ((tid / blocksize) / blocksize) / (columns / blocksize);
//         == {
//             FStar.Math.Lemmas.division_multiplication_lemma tid (blocksize) (blocksize);
//             FStar.Math.Lemmas.division_multiplication_lemma tid (blocksize * blocksize) (columns / blocksize)
//         }
//         tid / (blocksize * blocksize * (columns / blocksize));
//         == { FStar.Math.Lemmas.lemma_div_exact columns (blocksize) }
//         tid / (blocksize * columns);
//         <= { FStar.Math.Lemmas.lemma_div_le tid (rows * columns - 1) (blocksize * columns) }
//         (rows * columns - 1) / (blocksize * columns);
//         <= { admit() }
//         rows / (blocksize) - 1;
//     };
//     let idx_y: nat = thread_id_y + block_id_y * blocksize in
//     let block_idx_x: nat = block_id_x + idx_y * (columns / blocksize) in
//     let idx: nat = thread_id_x + block_idx_x * blocksize in
//     calc (<=) {
//         idx <: int;
//         == {}
//         thread_id_x + (block_id_x + (thread_id_y + block_id_y * blocksize) * (columns / blocksize)) * blocksize;
//         == { FStar.Math.Lemmas.lemma_div_exact columns (blocksize); admit() }
//         thread_id_x + block_id_x * blocksize + (thread_id_y + block_id_y * blocksize) * columns;
//         == { admit() }
//         thread_id_x + block_id_x * blocksize + thread_id_y * columns + block_id_y * blocksize * columns;
//         <= { admit() }
//         (blocksize - 1)
//             + (columns / blocksize - 1) * blocksize
//             + (blocksize - 1) * columns
//             + (rows / (blocksize) - 1) * blocksize * columns;
//         == { FStar.Math.Lemmas.lemma_div_exact columns (blocksize); admit() }
//         blocksize - 1 + columns - blocksize + (blocksize - 1) * columns + (rows / (blocksize) - 1) * blocksize * columns;
//         == { admit() }
//         blocksize - 1 + columns - blocksize + blocksize * columns - columns + (rows / (blocksize)) * blocksize * columns - blocksize * columns;
//         == { FStar.Math.Lemmas.lemma_div_exact rows (blocksize) }
//         blocksize - 1 + columns - blocksize + blocksize * columns - columns + rows * columns - blocksize * columns;
//         == { FStar.Math.Lemmas.lemma_div_exact rows (blocksize) }
//         rows * columns - 1;
//     };
//     idx

// let idx_to_thread_id (rows columns: (i: pos { i % blocksize == 0 })) (r: nat { r < rows * columns }): GTot (tid: nat { tid < rows * columns })
//     = lemma_div_mult_lt r rows columns;
//       let row = r / columns in
//       let col = r % columns in
//       assert (row <= rows - 1 /\ col <= columns - 1);
//       let block_row = row / blocksize in
//       let block_col = col / blocksize in
//       assert (block_row <= rows / blocksize - 1 /\ block_col <= columns / blocksize - 1);
//       let row_in_block = row % blocksize in
//       let col_in_block = col % blocksize in
//       admit();
//       block_row * SZ.v tpb * (columns / blocksize) + block_col * SZ.v tpb + row_in_block * blocksize + col_in_block

//     //   ((r / columns) / blocksize) * SZ.v tpb * (columns / blocksize)
//     //   + ((r % columns) / blocksize) * SZ.v tpb
//     //   + ((r / columns) % blocksize) * blocksize
//     //   + ((r % columns) % blocksize)

// let idx_to_thread_id_2 (blocksize: pos) (rows columns: (i: pos { i % blocksize == 0 })) (r: erased nat { r < rows * columns }): (tid: erased nat { tid < rows * columns })
//     = let thread_id_x: nat = r % blocksize in
//       let block_id: nat = r / blocksize in
//       let block_id_x: nat = block_id % (columns / blocksize) in
//       let idx_y: nat = block_id / (columns / blocksize) in
//       let thread_id_y: nat = idx_y % blocksize in
//       let block_id_y: nat = idx_y / blocksize in
//       let bid = block_id_x + block_id_y * (columns / blocksize) in
//       let thread_row = thread_id_y + bid * blocksize in
//       let tid = thread_id_x + thread_row * blocksize in
//       admit();
//       tid

// let lemma_mod_mod (a: int) (b c: pos):
//   Lemma (requires (b % c == 0))
//         (ensures  ((a % b) % c == a % c)) = FStar.Math.Lemmas.modulo_modulo_lemma a c (b / c)

// let lemma_div_div_same (a: int) (b c: pos):
//   Lemma (requires (c % b == 0))
//         (ensures  ((a / b) / (c / b) == a / c)) = FStar.Math.Lemmas.division_multiplication_lemma a b (c / b)

// let lemma_div_mult_same (a: int) (b c: pos):
//   Lemma (requires (b % c == 0))
//         (ensures  (a / b * c == a * c / b)) = ()

// let thread_id_to_idx_inverse (rows columns: (i: pos { i % blocksize == 0 })):
//     Lemma (exists (inv: (r: nat{r < rows * columns} -> Prims.GTot (tid: nat{tid < rows * columns}))).
//             forall (tid: nat { tid < rows * columns }). inv (thread_id_to_idx rows columns tid) = tid) =
//     introduce exists (inv: (r: nat{r < rows * columns} -> Prims.GTot (tid: nat{tid < rows * columns}))).
//                 forall (tid: nat { tid < rows * columns }). inv (thread_id_to_idx rows columns tid) = tid
//     with (idx_to_thread_id rows columns)
//     and introduce forall (tid: nat { tid < rows * columns }). (idx_to_thread_id rows columns) (thread_id_to_idx rows columns tid) = tid
//     with (
//         admit();
//         FStar.Math.Lemmas.division_multiplication_lemma tid (blocksize) (blocksize);
//         calc (==) {
//             thread_id_to_idx rows columns tid <: int;
//             == {}
//             ((tid / (SZ.v tpb)) % (columns / blocksize)) * blocksize
//                 + ((tid / (SZ.v tpb)) / (columns / blocksize)) * blocksize * columns
//                 + (tid % (SZ.v tpb)) % blocksize
//                 + ((tid % (SZ.v tpb)) / blocksize) * columns;
//         };
//         calc (==) {
//             (tid % (SZ.v tpb)) % blocksize + ((tid % (SZ.v tpb)) / blocksize) * columns;
//             == { lemma_mod_mod tid (SZ.v tpb) (blocksize) }
//             tid % blocksize + ((tid % (SZ.v tpb)) / blocksize) * columns;
//         };
//         calc (==) {
//             ((tid / SZ.v tpb) / (columns / blocksize)) * blocksize * columns;
//             == { lemma_div_div_same (tid / blocksize) (blocksize) columns }
//             ((tid / blocksize) / columns) * blocksize * columns;
//             == { admit() } // TODO
//             tid - (tid % (blocksize * columns));
//         };
//         calc (==) {
//             ((tid / (SZ.v tpb)) % (columns / blocksize)) * blocksize;
//             == { FStar.Math.Lemmas.modulo_scale_lemma (tid / (SZ.v tpb)) (blocksize) (columns / blocksize) }
//             ((tid / (SZ.v tpb)) * blocksize) % ((columns / blocksize) * blocksize);
//             == {}
//             ((tid / (SZ.v tpb)) * blocksize) % columns;
//             == {}
//             ((tid / blocksize / blocksize) * blocksize) % columns;
//             == {}
//             (tid / blocksize - (tid / blocksize) % blocksize) % columns;
//         };
//         calc (==) {
//             (idx_to_thread_id rows columns) (thread_id_to_idx rows columns tid);
//             == { admit() }
//             tid;
//         }
//     )

// #push-options "--z3cliopt 'smt.arith.nl=false'"

// instance titi_permutation (blocksize: pos) (rows columns: (i: pos { i % blocksize == 0 })) : permutation (i : erased nat { i < rows * columns }) = {
//     f = thread_id_to_idx_2 blocksize rows columns;
//     g = idx_to_thread_id_2 blocksize rows columns;
//     proof = fun (x y: (i: erased nat { i < rows * columns })) -> (
//         let a_block_id_y = ((x / blocksize) / blocksize) / (columns / blocksize) in
//         let a_thread_id_y = (x / blocksize) % blocksize in
//         let a_block_id_x = ((x / blocksize) / blocksize) % (columns / blocksize) in
//         let a_thread_id_x = x % blocksize in
//         admit();
//         calc (==>) {
//             thread_id_to_idx_2 blocksize rows columns x == y;
//             ==> { calc (==) {
//                 reveal (idx_to_thread_id_2 blocksize rows columns (thread_id_to_idx_2 blocksize rows columns x)) <: int;
//                 == {}
//                 // let y = thread_id_to_idx_2 rows columns x
//                 (let y = a_thread_id_x + (a_block_id_x + (a_thread_id_y + a_block_id_y * blocksize) * (columns / blocksize)) * blocksize in
//                         // idx_to_thread_id_2 rows columns y
//                         y % blocksize
//                             + (((y / blocksize) / (columns / blocksize)) % blocksize
//                                 + ((y / blocksize) % (columns / blocksize)
//                                     + (((y / blocksize) / (columns / blocksize)) / blocksize) * (columns / blocksize)
//                                 ) * blocksize
//                             ) * blocksize);
//                 == {
//                     FStar.Math.Lemmas.modulo_distributivity a_thread_id_x (
//                         (a_block_id_x + (a_thread_id_y + a_block_id_y * blocksize) * (columns / blocksize)) * blocksize
//                     ) (blocksize);
//                     FStar.Math.Lemmas.multiple_modulo_lemma (
//                         a_block_id_x + (a_thread_id_y + a_block_id_y * blocksize) * (columns / blocksize)
//                     ) (blocksize);
//                     admit()
//                 }
//                 // let y = thread_id_to_idx_2 blocksize rows columns x
//                 (let y = a_thread_id_x + (a_block_id_x + (a_thread_id_y + a_block_id_y * blocksize) * (columns / blocksize)) * blocksize in
//                         // idx_to_thread_id_2 blocksize rows columns y
//                         a_thread_id_x % blocksize
//                             + (((y / blocksize) / (columns / blocksize)) % blocksize
//                                 + ((y / blocksize) % (columns / blocksize)
//                                     + (((y / blocksize) / (columns / blocksize)) / blocksize) * (columns / blocksize)
//                                 ) * blocksize
//                             ) * blocksize);
//                 == { admit() }
//                 a_thread_id_x % blocksize
//                     + (a_thread_id_y % blocksize
//                         + (a_block_id_x % (columns / blocksize)
//                             + a_block_id_y * (columns / blocksize)
//                         ) * blocksize
//                     ) * blocksize;
//                 == { admit() }
//                 x % blocksize
//                     + ((x / blocksize) % blocksize
//                         + (((x / blocksize) / blocksize) % (columns / blocksize)
//                             + ((x / blocksize) / blocksize) / (columns / blocksize) * (columns / blocksize)
//                         ) * blocksize
//                     ) * blocksize;
//                 == { admit() }
//                 reveal x <: int;
//             }}
//             idx_to_thread_id_2 blocksize rows columns y == x;
//         };
//         calc (==>) {
//             idx_to_thread_id_2 blocksize rows columns y == x;
//             ==> { calc (==) {
//                 reveal (thread_id_to_idx_2 blocksize rows columns (idx_to_thread_id_2 blocksize rows columns y)) <: int;
//                 == { admit() }
//                 reveal y <: int;
//             }}
//             thread_id_to_idx_2 blocksize rows columns x == y;
//         }
//     )
// }
