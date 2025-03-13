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
  (bdim : szp{ bdim /? rows /\ bdim /? cols })
  (#l : R.mlayout (rows * bdim) (cols * bdim))
  (c : R.clayout l)
  : M4.clayout4 l = {
    parent = c;
    c_mrows = rows;
    c_mcols = cols;
    c_brows = bdim;
    c_bcols = bdim;
}

inline_for_extraction noextract
let coerce_eq (#a:Type) (#b:Type) (_:squash (a == b)) (x:a) : b = x

let row_major4 : M4.mrepr4 =
  fun rows cols brows bcols ->
    R.row_major (rows * brows) (cols * bcols)

inline_for_extraction noextract
let parent4 (rows cols : szp) (bdim : szp)
  (_ : squash (SZ.fits (rows * bdim) /\ SZ.fits (cols * bdim) /\ SZ.fits ((rows * bdim) * (cols * bdim))))
  : R.clayout (R.row_major (SZ.v rows * SZ.v bdim) (SZ.v cols * SZ.v bdim))
  (* ARGHHHHH Need to inline. *)
  = [@@inline_let] let rr = rows *^ bdim in
    [@@inline_let] let cc = cols *^ bdim in
    coerce_eq () <| R.crepr_row_major.map rr cc

inline_for_extraction noextract
instance clayout4_row_major
  (rows cols : szp)
  (bdim : szp { FStar.SizeT.fits ((rows * bdim) * (cols * bdim)) })
  : M4.clayout4 (row_major4 rows cols bdim bdim) = {
    c_mrows = rows;
    c_mcols = cols;
    c_brows = bdim;
    c_bcols = bdim;
    parent = parent4 rows cols bdim ();
  }

inline_for_extraction noextract
fn matmul_cpu
  (bdim : szp)
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
    pure (bdim /? rows /\ bdim /? cols /\ bdim /? shared) **
    pure (rows * cols <= max_blocks)
  returns
    c : vec u64
  ensures
    exists* sc. c |-> sc
{
  let mcols = cols /^ bdim;
  let mshared = shared /^ bdim;
  let mrows = rows /^ bdim;

  let gA = M4.gpu_matrix_alloc0 #u64 _ _ _ _ (row_major4 mrows   mshared bdim bdim);
  let gB = M4.gpu_matrix_alloc0 #u64 _ _ _ _ (row_major4 mshared mcols   bdim bdim);
  let gC = M4.gpu_matrix_alloc0 #u64 _ _ _ _ (row_major4 mrows   mcols   bdim bdim);

  // assume (pure (rows * shared == R.mlayout_size (row_major4 mrows mshared bdim bdim)));
  // assume (pure (shared * cols == R.mlayout_size (row_major4 mshared mcols bdim bdim)));
  // assume (pure (rows * cols   == R.mlayout_size (row_major4 mrows mcols bdim bdim)));
  M4.gpu_matrix_from_array gB b;
  M4.gpu_matrix_from_array gA a;

  with vc. assert gC |-> vc;

  assume (pure (mrows * mcols <= max_blocks));
  assume (pure (bdim * bdim <= max_threads));
  matmul_gpu bdim _ _ _ gA gB gC;

  let c = Pulse.Lib.Vec.alloc #u64 zero (SZ.mul rows cols);
  M4.gpu_matrix_to_array c gC;

  M4.gpu_matrix_free gA;
  M4.gpu_matrix_free gB;
  M4.gpu_matrix_free gC;

  c
}

let matmul_u64_rrr
  (bdim : szp)
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp{three_fits rows shared cols})
  (a : vec u64)
  (b : vec u64)
  (#sa : erased (seq u64){ len sa == rows * shared })
  (#sb : erased (seq u64){ len sb == shared * cols })
  = matmul_cpu bdim #rows #shared #cols a b #sa #sb

let matmul_u64_rrr_tile32
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp{three_fits rows shared cols})
  (a : vec u64)
  (b : vec u64)
  (#sa : erased (seq u64){ len sa == rows * shared })
  (#sb : erased (seq u64){ len sb == shared * cols })
  = matmul_cpu 32sz #rows #shared #cols a b #sa #sb
