module GPU.MatMulOpt.Layout

open FStar.Mul
module SZ = FStar.SizeT
open GPU.MatMulOpt.Kernel

// let block_id (rows columns: nat) (tid: nat { tid < rows * columns }) : GTot nat = tid / (SZ.v tpb)
// let thread_id (rows columns: nat) (tid: nat { tid < rows * columns }) : GTot nat = tid % (SZ.v tpb)

let lemma_div_lt (a b:int) (d:pos { b % d == 0 }):
  Lemma (requires (a < b))
        (ensures  (a / d < b / d)) = ()

let lemma_mod_mult (a b: int) (c d: pos):
  Lemma (requires (a % c == 0 /\ b % d == 0))
        (ensures  ((a * b) % (c * d) == 0)) = admit()

let lemma_div_mult_lt (a b: int) (d: pos):
  Lemma (requires (a < b * d))
        (ensures  (a / d < b)) = ()

let lemma_mult_le (a b: int) (d: nat):
  Lemma (requires (a <= b))
        (ensures  (a * d <= b * d)) = ()

let lemma_mult_distr (a b: int) (d: nat):
  Lemma (ensures  (a * d + b * d == (a + b) * d)) = ()

let thread_id_to_idx (rows columns: (i: pos { i % SZ.v blocksize == 0 })) (tid: nat { tid < rows * columns }): GTot (r: nat { r < rows * columns })
  = admit(); lemma_mod_mult rows columns (SZ.v blocksize) (SZ.v blocksize);
    lemma_div_lt tid (rows * columns) (SZ.v tpb);
    let block_id:   (i: nat { i < (rows / SZ.v blocksize) * (columns / SZ.v blocksize) }) = tid / (SZ.v tpb) in
    let block_id_x: (i: nat { i <= columns / SZ.v blocksize - 1 })                           = block_id % (columns / SZ.v blocksize) in // which row of blocks
    lemma_div_mult_lt block_id (rows / SZ.v blocksize) (columns / SZ.v blocksize);
    let block_id_y: (i: nat { i <= rows / SZ.v blocksize - 1 })                              = block_id / (columns / SZ.v blocksize) in // which column of blocks
    let block_offset: nat = block_id_x * SZ.v blocksize + block_id_y * SZ.v blocksize * columns in
    calc (<=) {
      block_offset <: int;
      == {}
      block_id_x * SZ.v blocksize + block_id_y * SZ.v blocksize * columns;
      <= { lemma_mult_le block_id_x (columns / SZ.v blocksize - 1) (SZ.v blocksize) }
      ((columns / SZ.v blocksize - 1) * SZ.v blocksize + block_id_y * SZ.v blocksize * columns);
      <= { lemma_mult_le block_id_y (rows / SZ.v blocksize - 1) (SZ.v blocksize * columns) }
      ((columns / SZ.v blocksize - 1) * SZ.v blocksize + (rows / SZ.v blocksize - 1) * SZ.v blocksize * columns);
    };
    let thread_id:   (i: nat { i < SZ.v tpb })       = tid % (SZ.v tpb) in
    let thread_id_x: (i: nat { i <= SZ.v blocksize - 1 }) = thread_id % SZ.v blocksize in
    let thread_id_y: (i: nat { i <= SZ.v blocksize - 1 }) = thread_id / SZ.v blocksize in
    let thread_offset: nat = thread_id_x + thread_id_y * columns in
    calc (<=) {
      thread_offset <: int;
      == {}
      thread_id_x + thread_id_y * columns;
      <= {}
      (SZ.v blocksize - 1) + thread_id_y * columns;
      <= { lemma_mult_le thread_id_y (SZ.v blocksize - 1) columns }
      (SZ.v blocksize - 1) + (SZ.v blocksize - 1) * columns;
    };
    calc (<) {
      block_offset + thread_offset;
      <= {}
      ((columns / SZ.v blocksize - 1) * SZ.v blocksize + (rows / SZ.v blocksize - 1) * SZ.v blocksize * columns) + ((SZ.v blocksize - 1) + (SZ.v blocksize - 1) * columns);
      == {}
      1 * columns + (rows - SZ.v blocksize) * columns - 1 + (SZ.v blocksize - 1) * columns;
      == {
        lemma_mult_distr 1 (rows - SZ.v blocksize) columns;
        lemma_mult_distr (1 + rows - SZ.v blocksize) (SZ.v blocksize - 1) columns
      }
      (1 + rows - SZ.v blocksize + SZ.v blocksize - 1) * columns - 1;
      == {}
      rows * columns - 1;
      < {}
      rows * columns;
    };
    // assert (block_offset + thread_offset < (rows - SZ.v blocksize + 1) * columns - SZ.v blocksize + SZ.v blocksize + (SZ.v blocksize - 1) * columns);
    // assert (block_offset + thread_offset < rows * columns);
    block_offset + thread_offset

let idx_to_thread_id (rows columns: (i: pos { i % SZ.v blocksize == 0 })) (r: nat { r < rows * columns }): GTot (tid: nat { tid < rows * columns })
    = lemma_div_mult_lt r rows columns;
      let row = r / columns in
      let col = r % columns in
      assert (row <= rows - 1 /\ col <= columns - 1);
      let block_row = row / SZ.v blocksize in
      let block_col = col / SZ.v blocksize in
      assert (block_row <= rows / SZ.v blocksize - 1 /\ block_col <= columns / SZ.v blocksize - 1);
      let row_in_block = row % SZ.v blocksize in
      let col_in_block = col % SZ.v blocksize in
      admit();
      block_row * SZ.v tpb * (columns / SZ.v blocksize) + block_col * SZ.v tpb + row_in_block * SZ.v blocksize + col_in_block

    //   ((r / columns) / SZ.v blocksize) * SZ.v tpb * (columns / SZ.v blocksize)
    //   + ((r % columns) / SZ.v blocksize) * SZ.v tpb
    //   + ((r / columns) % SZ.v blocksize) * SZ.v blocksize
    //   + ((r % columns) % SZ.v blocksize)

let lemma_mod_mod (a: int) (b c: pos):
  Lemma (requires (b % c == 0))
        (ensures  ((a % b) % c == a % c)) = FStar.Math.Lemmas.modulo_modulo_lemma a c (b / c)

let lemma_div_div_same (a: int) (b c: pos):
  Lemma (requires (c % b == 0))
        (ensures  ((a / b) / (c / b) == a / c)) = FStar.Math.Lemmas.division_multiplication_lemma a b (c / b)

// let lemma_div_mult_same (a: int) (b c: pos):
//   Lemma (requires (b % c == 0))
//         (ensures  (a / b * c == a * c / b)) = ()

let thread_id_to_idx_inverse (rows columns: (i: pos { i % SZ.v blocksize == 0 })):
    Lemma (exists (inv: (r: nat{r < rows * columns} -> Prims.GTot (tid: nat{tid < rows * columns}))).
            forall (tid: nat { tid < rows * columns }). inv (thread_id_to_idx rows columns tid) = tid) =
    introduce exists (inv: (r: nat{r < rows * columns} -> Prims.GTot (tid: nat{tid < rows * columns}))).
                forall (tid: nat { tid < rows * columns }). inv (thread_id_to_idx rows columns tid) = tid
    with (idx_to_thread_id rows columns)
    and introduce forall (tid: nat { tid < rows * columns }). (idx_to_thread_id rows columns) (thread_id_to_idx rows columns tid) = tid
    with (
        admit();
        FStar.Math.Lemmas.division_multiplication_lemma tid (SZ.v blocksize) (SZ.v blocksize);
        calc (==) {
            thread_id_to_idx rows columns tid <: int;
            == {}
            ((tid / (SZ.v tpb)) % (columns / SZ.v blocksize)) * SZ.v blocksize
                + ((tid / (SZ.v tpb)) / (columns / SZ.v blocksize)) * SZ.v blocksize * columns
                + (tid % (SZ.v tpb)) % SZ.v blocksize
                + ((tid % (SZ.v tpb)) / SZ.v blocksize) * columns;
        };
        calc (==) {
            (tid % (SZ.v tpb)) % SZ.v blocksize + ((tid % (SZ.v tpb)) / SZ.v blocksize) * columns;
            == { lemma_mod_mod tid (SZ.v tpb) (SZ.v blocksize) }
            tid % SZ.v blocksize + ((tid % (SZ.v tpb)) / SZ.v blocksize) * columns;
        };
        calc (==) {
            ((tid / SZ.v tpb) / (columns / SZ.v blocksize)) * SZ.v blocksize * columns;
            == { lemma_div_div_same (tid / SZ.v blocksize) (SZ.v blocksize) columns }
            ((tid / SZ.v blocksize) / columns) * SZ.v blocksize * columns;
            == { admit() } // TODO
            tid - (tid % (SZ.v blocksize * columns));
        };
        calc (==) {
            ((tid / (SZ.v tpb)) % (columns / SZ.v blocksize)) * SZ.v blocksize;
            == { FStar.Math.Lemmas.modulo_scale_lemma (tid / (SZ.v tpb)) (SZ.v blocksize) (columns / SZ.v blocksize) }
            ((tid / (SZ.v tpb)) * SZ.v blocksize) % ((columns / SZ.v blocksize) * SZ.v blocksize);
            == {}
            ((tid / (SZ.v tpb)) * SZ.v blocksize) % columns;
            == {}
            ((tid / SZ.v blocksize / SZ.v blocksize) * SZ.v blocksize) % columns;
            == {}
            (tid / SZ.v blocksize - (tid / SZ.v blocksize) % SZ.v blocksize) % columns;
        };
        calc (==) {
            (idx_to_thread_id rows columns) (thread_id_to_idx rows columns tid);
            == { admit() }
            tid;
        }
    )
