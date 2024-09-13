module GPU.MatMul.Impure

#lang-pulse

#push-options "--fuel 1 --ifuel 1"

open GPU

module U64 = FStar.UInt64
module SZ = FStar.SizeT

let gpu_pts_to_matrix #a (rows columns: nat) (ga : gpu_array a (rows * columns)) (shared: erased nat{shared > 0}) (s: erased (seq a)): slprop =
  gpu_pts_to_array ga #(1.0R /. Real.of_int shared) s

ghost
fn gpu_matrix_share_underspec
  (#a:Type u#0)
  (#uid: int)
  (rows columns: nat)
  (ga : gpu_array a (rows * columns))
  (shared: erased nat{shared > 0})
  (s: erased (seq a) { Seq.length s == rows * columns })
  requires gpu_pts_to_matrix #a rows columns ga 1 s
  ensures bigstar #uid 0 shared (fun _ -> gpu_pts_to_matrix #a rows columns ga shared s)

ghost
fn gpu_matrix_unshare_underspec
  (#a:Type u#0)
  (#uid: int)
  (rows columns: nat)
  (ga : gpu_array a (rows * columns))
  (shared: erased nat{shared > 0})
  (s: erased (seq a) { Seq.length s == rows * columns })
  requires bigstar #uid 0 shared (fun _ -> gpu_pts_to_matrix #a rows columns ga shared s)
  ensures  gpu_pts_to_matrix #a rows columns ga 1 s


// TODO: Make --cmi work and put this in the fst

[@@CPrologue "__device__"]
inline_for_extraction noextract
fn gpu_matrix_read
  (#a:Type0)
  (#rows #columns: sz)
  (ga : gpu_array a (rows * columns))
  (#shared: erased nat{shared > 0})
  (#s: erased (seq a) { Seq.length s == rows * columns })
  (row: sz{SZ.v row < rows})
  (col: sz{SZ.v col < columns})
  requires gpu ** gpu_pts_to_matrix rows columns ga shared s
  returns v: a
  // TODO: is the assert here opaque?
  ensures gpu ** gpu_pts_to_matrix rows columns ga shared s ** pure (assert ((SZ.v row + 1) * columns <= rows * columns); v == Seq.index s (row * columns + SZ.v col))
{
  open FStar.SizeT;
  assume (pure (forall (x:nat). SizeT.fits x)); // CHEATING overflow
  unfold gpu_pts_to_matrix rows columns ga shared s;
  unfold gpu_pts_to_array ga #(Real.one /. Real.of_int shared) s;
  // TODO: strange that commenting this out causes an error
  assert (pure ((row + 1) * columns <= rows * columns));
  let idx = row *^ columns +^ col;
  let v = gpu_array_read #a #(rows * columns) #0 #(rows * columns) ga #(Real.one /. Real.of_int shared) idx #s;
  fold gpu_pts_to_array ga #(Real.one /. Real.of_int shared) s;
  fold gpu_pts_to_matrix rows columns ga shared s;
  v
}

// fixme, function above extracts wrongly (returns void* instead of uint64_t in
// the specialization). If the inline_for_extraction was not there above, the
// resulting C code would not typecheck.
[@@CPrologue "__device__"]
inline_for_extraction noextract
fn gpu_matrix_read_u64
  (#rows #columns: sz)
  (ga : gpu_array u64 (rows * columns))
  (#shared: erased nat{shared > 0})
  (#s: erased (seq u64) { Seq.length s == rows * columns })
  (row: sz{SZ.v row < rows})
  (col: sz{SZ.v col < columns})
  requires gpu ** gpu_pts_to_matrix rows columns ga shared s
  returns v: u64
  // TODO: is the assert here opaque?
  ensures gpu ** gpu_pts_to_matrix rows columns ga shared s ** pure (assert ((SZ.v row + 1) * columns <= rows * columns); v == Seq.index s (row * columns + SZ.v col))
{
  open FStar.SizeT;
  assume (pure (forall (x:nat). SizeT.fits x)); // CHEATING overflow
  unfold gpu_pts_to_matrix rows columns ga shared s;
  unfold gpu_pts_to_array ga #(Real.one /. Real.of_int shared) s;
  // TODO: strange that commenting this out causes an error
  assert (pure ((row + 1) * columns <= rows * columns));
  let idx = row *^ columns +^ col;
  let v = gpu_array_read #u64 #(rows * columns) #0 #(rows * columns) ga #(Real.one /. Real.of_int shared) idx #s;
  fold gpu_pts_to_array ga #(Real.one /. Real.of_int shared) s;
  fold gpu_pts_to_matrix rows columns ga shared s;
  v
}

#pop-options
