module Kuiper.Poly.Softmax

(* Much of this module is layout-polymorphic, but
we fix to l1_forward at the end so we can use arr_read_1. This
should not be the case once there are more flexible memcpy's. *)

// This is a very naive implementation of softmax on the GPU,
// which uses three separate kernels launches (exp, reduce, divide).
// A fused version is possible.

#lang-pulse
module Array1 = Kuiper.Array1

open Kuiper
module Vec = Pulse.Lib.Vec
module SZ = Kuiper.SizeT
module KS = Kuiper.Seq.Common

open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

(* From the CPU, read one element from a flat array1. *)
inline_for_extraction noextract
fn arr_read_1
  (#et : Type0) {| sized et |}
  (#len : erased nat)
  (a : Array1.t et (l1_forward len))
  (i : szlt len)
  (#f : perm)
  (#va : erased (lseq et len))
  preserves
    cpu
  preserves
    on gpu_loc (a |-> Frac f va)
  returns
    x : et
  ensures
    pure (x == va @! i)
{
  let ca = Pulse.Lib.Vec.alloc #et default 1sz;

  map_loc gpu_loc
    #(a |-> Frac f va)
    #(core a |-> Frac f va)
    fn _ {
      Array1.lower a;
      assert pure (Seq.equal (to_seq (l1_forward len) va) va);
      rewrite core a |-> Frac f (to_seq (l1_forward len) va)
           as core a |-> Frac f va;
    };

  (* FIXME: Need to give length of ca?!? *)
  gpu_memcpy_device_to_host' #_ #_ #1 ca 0sz (core a) i 1sz;

  let x = ca.(0sz);
  Pulse.Lib.Vec.free ca;

  map_loc gpu_loc
    #(core a |-> Frac f va)
    #(a |-> Frac f va)
    fn _ {
      Array1.raise' (l1_forward len) (core a);
      rewrite each from_array (l1_forward len) (core a) as a;
      assert pure (Seq.equal (from_seq (l1_forward len) va) va);
      ();
    };

  x;
}

ghost
fn explode_setup
  (#et : Type0)
  (lena : nat)
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#s : erased (lseq et lena))
  ()
  norewrite
  requires
    (a |-> s)
  ensures
    (forall+ (bid : natlt lena).
      Cell a bid |-> (Seq.index s bid)) **
    pure (SZ.fits (layout_size l))
{
  Array1.pts_to_ref a;
  Array1.explode a;
}

ghost
fn explode_teardown
  (#et : Type0)
  (f : et -> et)
  (lena : nat)
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#s : erased (lseq et lena))
  ()
  norewrite
  requires
    (forall+ (bid : natlt lena).
      Cell a bid |-> (f (s @! bid))) **
    pure (SZ.fits (layout_size l))
  ensures
    a |-> (Kuiper.Seq.Common.seq_map f s <: lseq et lena)
{
  forevery_map
    (fun (i:natlt lena) -> Cell a i |-> (f (s @! i)))
    (fun (i:natlt lena) -> Cell a i |-> ((KS.seq_map f s)@!i))
    fn x { () };
  Array1.implode a;
}

inline_for_extraction noextract
fn kf_map
  (#et : Type0)
  (f : et -> et)
  (#lena : erased nat)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (#s : erased (lseq et lena) )
  (bid : szlt lena)
  ()
  requires
    gpu **
    Cell a (bid <: natlt lena) |-> (s@!bid) **
    block_id lena bid
  ensures
    gpu **
    Cell a (bid <: natlt lena) |-> (f (s@!bid)) **
    block_id lena bid
{
  let x = Array1.read_cell a bid;
  Array1.write_cell a bid (f x);
}

inline_for_extraction noextract
let kmap
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (#_ : squash (Array1.is_global a))
  (#s : erased (lseq et lena))
: kernel_desc
    (a |-> s)
    (a |-> (Kuiper.Seq.Common.seq_map f s <: lseq et lena)) =
{
  nblk = lena;
  f = kf_map f a;

  frame    = pure (SZ.fits (layout_size l));
  teardown = explode_teardown f lena a;
  setup    = explode_setup lena a;
  kpre =  (fun (i:natlt lena) -> Cell a i |-> (s@!i));
  kpost = (fun (i:natlt lena) -> Cell a i |-> (f (s@!i)));
  kpost_sendable = solve;
  kpre_sendable  = solve;
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
  (#lena : szp)
  (a : array1 et (l1_forward lena) { is_global a })
  (#va: erased (lseq et lena))
  (ra: erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (lena <= max_threads) **
    pure (va %~ ra)
  ensures
    exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ softmax_real ra)
{
  (* Pointwise exponentiation. *)
  launch_sync (kmap exp lena a);

  (* Compute average. Need swap space since hreduce trashes the input. *)
  let a' = Array1.alloc0 #et lena (l1_forward lena);
  Array1.memcpy_device_to_device a' a lena;

  Classical.forall_intro_2 (fun x -> Classical.move_requires (exp_approx #et x));

  Kuiper.Poly.HReduce.reduce lena a' #(KS.seq_map exp va) (KS.seq_map rexp ra);
  with va'. assert on gpu_loc (a' |-> va');
  let avg = arr_read_1 a' 0sz;
  Array1.free a';

  (* Divide by average *)
  with s'. assert on gpu_loc (a |-> s');
  launch_sync (kmap (fun x -> div x avg) lena a);

  softmax_approx va ra;
  ()
}

inline_for_extraction noextract
fn softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lena : szp)
  (a : Vec.lvec et lena)
  (#va : erased (lseq et lena))
  (ra  : erased (lseq real lena))
  preserves
    cpu
  requires
    a |-> va **
    pure (lena <= max_threads) **
    pure (va %~ ra)
  ensures
    exists* (va' : lseq et lena).
      a |-> va' **
      pure (va' %~ softmax_real ra)
{
  let ga = Array1.alloc0 #et lena (l1_forward lena);
  Array1.memcpy_host_to_device ga a lena;
  softmax_gpu ga ra;
  Array1.memcpy_device_to_host' a 0sz ga 0sz lena;
  Array1.free ga;
  ()
}
