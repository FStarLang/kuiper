module Kuiper.Poly.Softmax

// This is a very naive implementation of softmax on the GPU,
// which uses three separate kernels launches (exp, reduce, divide).
// A fused version is possible.

#lang-pulse
open Kuiper
module Array = Kuiper.Array
(* ^ Why do I need this? Is it because Kuiper is a module and not a namespace? *)
module Vec = Pulse.Lib.Vec
module SZ = Kuiper.SizeT
module KS = Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec
open Kuiper.Real
open Kuiper.Approximates

(* From the CPU, read one element from a gpu array. *)
inline_for_extraction noextract
fn arr_read_1
  (#et : Type0) {| sized et |}
  (init : et) // silly
  (#len : erased nat)
  (a : gpu_array et len)
  (#f : perm)
  preserves cpu ** on gpu_loc (a |-> Frac f 'va)
  requires pure (len > 0 /\ is_global_array a)
  returns  x : et
  ensures  pure (Seq.length 'va > 0 /\ x == Seq.head 'va)
{
  gpu_pts_to_ref_located a; (* automate *)
  let ca = Pulse.Lib.Vec.alloc init 1sz;
  (* FIXME: Need to give lenght of ca?!? *)
  gpu_memcpy_device_to_host' #_ #_ #1 ca 0sz a 0sz 1sz;
  let x = ca.(0sz);
  Pulse.Lib.Vec.free ca;
  x;
}


ghost
fn explode_setup
  (#et : Type0)
  (lena : szp { lena < max_blocks })
  (a : gpu_array et lena)
  (#s: erased (Seq.seq et) { Seq.length s == SZ.v lena })
  ()
  norewrite
  requires
    (a |-> s)
  ensures
    (forall+ (bid : natlt lena).
      gpu_pts_to_cell a bid (Seq.index s bid)) **
    emp
{ gpu_array_slice_1 a; }

ghost
fn explode_teardown
  (#et : Type0)
  (f : et -> et)
  (lena : szp { lena < max_blocks })
  (a : gpu_array et lena)
  (#s : erased (Seq.seq et) { Seq.length s == SZ.v lena })
  ()
  norewrite
  requires
    (forall+ (bid : natlt lena).
      gpu_pts_to_cell a bid (f (s @! bid))) **
    emp
  ensures
    (a |-> (Kuiper.Seq.Common.seq_map f s))
{
  forevery_map
    (fun (i:natlt lena) -> gpu_pts_to_cell a i (f (s @! i)))
    (fun (i:natlt lena) -> gpu_pts_to_cell a i ((KS.seq_map f s)@!i))
    fn x { () };
  gpu_array_unslice_1 a #_ #(KS.seq_map f s)
}

inline_for_extraction noextract
fn kf_map
  (#et : Type0)
  (#lena : erased nat)
  (#s : erased (Seq.seq et) { Seq.length s == lena })
  (f: et -> et)
  (a : gpu_array et lena)
  (bid : szlt lena)
  ()
  requires
    gpu **
    gpu_pts_to_cell a bid (s@!bid) **
    block_id lena bid
  ensures
    gpu **
    gpu_pts_to_cell a bid (f (s@!bid)) **
    block_id lena bid
{
  let i = bid; rewrite each _ as i;
  assert (pure (i < lena));
  assert (pure (SZ.v i == bid));
  unfold (gpu_pts_to_cell a i _);
  let x = gpu_array_read a i;
  let ex = f x;
  gpu_array_write a i ex;
  (* a bit tedious to do this seq rewrite; would be nice to have a way to
     instruct the solver to do an extensional equality on this argument *)
  with ss. assert (gpu_pts_to_slice a (SZ.v i) (SZ.v i + 1) ss);
  assert pure (Seq.equal ss (seq![ex]));
  fold gpu_pts_to_cell a i ex;
  rewrite each i as _;
  ()
}

inline_for_extraction noextract
let kmap
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena < max_blocks })
  (a : gpu_array et lena { is_global_array a })
  (#s : erased (Seq.seq et) { Seq.length s == SZ.v lena })
: kernel_desc
    (a |-> s)
    (a |-> Kuiper.Seq.Common.seq_map f s) =
{
  nblk = lena;
  f = kf_map f a;

  teardown = explode_teardown f lena a;
  setup    = explode_setup lena a;
  kpre =  (fun (i:natlt lena) -> gpu_pts_to_cell a i (s@!i));
  kpost = (fun (i:natlt lena) -> gpu_pts_to_cell a i (f (s@!i)));
  frame = emp;
  kpost_sendable=solve;
  kpre_sendable=solve;
} <: kernel_desc_m_1 _ _

let map_div_avg (#et:Type0) {| floating et |} (s:Seq.seq et) (avg:et) =
  let open Kuiper.Seq.Common in
  let exps = seq_map exp s in
  seq_map (fun x -> div x avg) s

let softmax_spec (#et:Type0) {| floating et |} (s:Seq.seq et) =
  let open Kuiper.Seq.Common in
  let exps = seq_map exp s in
  let avg = seq_fold_left add zero exps in
  map_div_avg exps avg

let rec sum_non_zero
    (s:seq real { forall (i:natlt (Seq.length s)). Seq.index s i >. 0.0R })
    (acc:real)
: Lemma
  (requires Seq.length s > 0)
  (ensures Kuiper.Seq.Common.seq_fold_left add acc s >. acc)
  (decreases Seq.length s)
= if Seq.length s = 1 then ()
  else
    let open Kuiper.Seq.Common in
    let SCons hd tl = view_seq s in
    sum_non_zero tl (add acc hd <: real)

let softmax_approx
    (#et:Type0) {| floating et, real_like et, floating_real_like et |}
    (s0:seq et) (r0:seq real { s0 %~ r0 /\ Seq.length r0 > 0 })
: Lemma
  (ensures Kuiper.Seq.Common.(
      forall (avg:et). avg %~ sum (seq_map rexp r0) ==>
      seq_map (fun x -> div x avg) (seq_map exp s0) %~ softmax_real r0))
= let exps = KS.seq_map rexp r0 in
  sum_non_zero exps 0.0R;
  Classical.forall_intro_2 (fun x -> Classical.move_requires (exp_approx #et x));
  Classical.forall_intro_4 (fun (x y : et) (r : real) -> Classical.move_requires (div_approx #et x y r));
  ()

inline_for_extraction noextract
fn softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lena : szp { 0 < SZ.v lena /\ lena < max_threads })
  (a : gpu_array et lena { is_global_array a })
  (#s: erased (Seq.seq et))
  (#r: erased (Seq.seq real)  { Seq.length r == SizeT.v lena /\ (s<:seq et) %~ r /\ lena > 0 })
  preserves cpu
  requires on gpu_loc (a |-> s) ** pure (lena <= max_blocks)
  ensures  (exists* s'. on gpu_loc (a |-> s') ** pure (s' %~ softmax_real r))
{
  gpu_pts_to_ref_located a; (* recall length, automate *)

  (* Pointwise exponentiation. *)
  launch_sync (kmap exp lena a #s);

  (* Compute average. Need swap space since hreduce trashes the input. *)
  let a' = Array.gpu_array_alloc #et lena;
  gpu_memcpy_device_to_device a' a lena;

  Classical.forall_intro_2 (fun x -> Classical.move_requires (exp_approx #et x));

  Kuiper.Poly.HReduce.reduce lena a' #(KS.seq_map exp s) #(KS.seq_map rexp r);
  let avg = arr_read_1 zero a';
  gpu_array_free a';

  (* Divide by average *)
  with s'. assert on gpu_loc (a |-> s');
  launch_sync (kmap (fun x -> div x avg) lena a #s');

  softmax_approx s r;
  ()
}

inline_for_extraction noextract
fn softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lena : szp { lena < max_threads })
  (a : Vec.lvec et lena)
  (#s : erased (Seq.seq et))
  (#r : erased (Seq.seq real) { Seq.length r == SizeT.v lena /\ s %~ r /\ lena > 0 })
  preserves cpu
  requires (a |-> s) ** pure (lena <= max_blocks)
  ensures  (exists* s'. a |-> s' ** pure (s' %~ softmax_real r))
{
  let ga = Array.gpu_array_alloc #et lena;
  Array.gpu_memcpy_host_to_device ga a lena;
  softmax_gpu ga #_ #r;
  gpu_memcpy_device_to_host a ga lena;
  Array.gpu_array_free ga
}
