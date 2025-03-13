module Kuiper.MatMul.Tiled.Inst

#lang-pulse
open Kuiper
open Kuiper.MatMulCPU
open Kuiper.MatMulGPU.Type
module R = Kuiper.Matrix.Reprs
module M4 = Kuiper.Matrix4
module SZ = FStar.SizeT
open Kuiper.EMatrix4 { ematrix4 }

(* TODO: Fit this into the non-tiled types and make it generic
and nice. Currently we just friend and instantiate one version
manually. *)

open Kuiper.MatMul.Tiled
friend Kuiper.MatMul.Tiled

inline_for_extraction noextract
let clayout4_from_clayout
  (#rows #cols : szp)
  (tile : szp{ tile /? rows /\ tile /? cols })
  (#l : R.mlayout (rows * tile) (cols * tile))
  (c : R.clayout l)
  : M4.clayout4 l = {
    parent = c;
    c_mrows = rows;
    c_mcols = cols;
    c_brows = tile;
    c_bcols = tile;
}

inline_for_extraction noextract
let coerce_eq (#a:Type) (#b:Type) (_:squash (a == b)) (x:a) : b = x

let row_major4 : M4.mrepr4 =
  fun rows cols brows bcols ->
    R.row_major (rows * brows) (cols * bcols)

inline_for_extraction noextract
let parent4 (rows cols : szp) (tile : szp)
  (_ : squash (SZ.fits (rows * tile) /\ SZ.fits (cols * tile) /\ SZ.fits ((rows * tile) * (cols * tile))))
  : R.clayout (R.row_major (SZ.v rows * SZ.v tile) (SZ.v cols * SZ.v tile))
  (* ARGHHHHH Need to inline. *)
  = [@@inline_let] let rr = rows *^ tile in
    [@@inline_let] let cc = cols *^ tile in
    coerce_eq () <| R.crepr_row_major.map rr cc

inline_for_extraction noextract
instance clayout4_row_major
  (rows cols : szp)
  (tile : szp { FStar.SizeT.fits ((rows * tile) * (cols * tile)) })
  : M4.clayout4 (row_major4 rows cols tile tile) = {
    c_mrows = rows;
    c_mcols = cols;
    c_brows = tile;
    c_bcols = tile;
    parent = parent4 rows cols tile ();
  }

inline_for_extraction noextract
fn matmul_cpu
  (tile : szp)
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp{three_fits rows shared cols})
  (a : vec u64)
  (b : vec u64)
  (#sa : erased (seq u64){ len sa == rows * shared })
  (#sb : erased (seq u64){ len sb == shared * cols })
  preserves
    cpu **
    (a |-> sa) **
    (b |-> sb)
  requires
    (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
       is not needed for all kernels. *)
    pure (SZ.fits (rows * shared) /\ SZ.fits (shared * cols) /\ SZ.fits (rows * cols)) **
    pure (tile /? rows /\ tile /? cols /\ tile /? shared) **
    pure (rows * cols <= max_blocks)
  returns
    c : vec u64
  ensures
    exists* sc. c |-> sc
{
  let mcols = cols /^ tile;
  let mshared = shared /^ tile;
  let mrows = rows /^ tile;

  let gA = M4.gpu_matrix_alloc0 #u64 _ _ _ _ (row_major4 mrows   mshared tile tile);
  let gB = M4.gpu_matrix_alloc0 #u64 _ _ _ _ (row_major4 mshared mcols   tile tile);
  let gC = M4.gpu_matrix_alloc0 #u64 _ _ _ _ (row_major4 mrows   mcols   tile tile);

  // assume (pure (rows * shared == R.mlayout_size (row_major4 mrows mshared tile tile)));
  // assume (pure (shared * cols == R.mlayout_size (row_major4 mshared mcols tile tile)));
  // assume (pure (rows * cols   == R.mlayout_size (row_major4 mrows mcols tile tile)));
  M4.gpu_matrix_from_array gB b;
  M4.gpu_matrix_from_array gA a;

  with vc. assert gC |-> vc;

  assume (pure (mrows * mcols <= max_blocks));
  assume (pure (tile * tile <= max_threads));
  matmul_gpu tile _ _ _ gA gB gC;

  let c = Pulse.Lib.Vec.alloc #u64 zero (SZ.mul rows cols);
  M4.gpu_matrix_to_array c gC;

  M4.gpu_matrix_free gA;
  M4.gpu_matrix_free gB;
  M4.gpu_matrix_free gC;

  c
}

let matmul_u64_rrr
  (tile : szp)
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp{three_fits rows shared cols})
  (a : vec u64)
  (b : vec u64)
  (#sa : erased (seq u64){ len sa == rows * shared })
  (#sb : erased (seq u64){ len sb == shared * cols })
  = matmul_cpu tile #rows #shared #cols a b #sa #sb

let matmul_u64_rrr_tile32
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp{three_fits rows shared cols})
  (a : vec u64)
  (b : vec u64)
  (#sa : erased (seq u64){ len sa == rows * shared })
  (#sb : erased (seq u64){ len sb == shared * cols })
  = matmul_cpu 32sz #rows #shared #cols a b #sa #sb
