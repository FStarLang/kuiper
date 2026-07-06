module Kuiper.Kernel.Map

(* Simple kernel: pointwise map of a function on an array. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Shareable
module SZ = Kuiper.SizeT
module TMap = Kuiper.Kernel.TMap

inline_for_extraction noextract
fn map_gpu
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#s: erased (chest1 et lena))
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)
{
  TMap.map_gpu (CCons lena CNil) f lena a;
}

inline_for_extraction noextract
fn map_host
  (#et : Type0) {| sized et |}
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (a : Pulse.Lib.Vec.lvec et lena)
  (#s: erased (lseq et lena))
  preserves cpu
  requires  a |-> s
  ensures   a |-> lseq_map f s
{
  let ga = alloc0 #et lena (l1_forward lena);
  with em. assert on gpu_loc (ga |-> em);

  (* Host -> device. *)
  map_loc gpu_loc
    #(ga |-> em)
    #(core ga |-> to_seq (l1_forward lena) em)
    fn _ { tensor_concr ga; };
  gpu_memcpy_host_to_device (core ga) a lena;
  map_loc gpu_loc
    #(core ga |-> reveal s)
    #(ga |-> from_seq (l1_forward lena) s)
    fn _ {
      tensor_abs' (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> from_seq (l1_forward lena) s)
           as (ga |-> from_seq (l1_forward lena) s);
    };

  map_gpu f lena ga;

  (* Device -> host. *)
  with res. assert on gpu_loc (ga |-> res);
  map_loc gpu_loc
    #(ga |-> res)
    #(core ga |-> to_seq (l1_forward lena) res)
    fn _ { tensor_concr ga; };
  gpu_memcpy_device_to_host a (core ga) lena;
  map_loc gpu_loc
    #(core ga |-> to_seq (l1_forward lena) res)
    #(ga |-> res)
    fn _ {
      tensor_abs (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> reveal res)
           as (ga |-> reveal res);
    };
  free ga;

  assert pure (Seq.equal (to_seq (l1_forward lena) res) (lseq_map f s));
  ();
}

instance shareable_tensor_pts_to
  (#r : nat) (#d : shape r) (#et : Type0)
  (#l : tlayout d)
  (a : tensor et l)
  (#s : chest d et)
  (base : perm)
  : shareable (fun fr -> on gpu_loc (a |-> Frac (base *. fr) s))
  = magic()

inline_for_extraction noextract
fn map_gpu2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (a : array1 et la)
  (b : array1 et lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#sa : erased (chest1 et lena))
  (#sb : erased (chest1 et lena))
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires  on gpu_loc (a |-> sa)
  ensures   on gpu_loc (a |-> chest1_map2 f sa sb)
{
  fn ff (#fr: perm) (i : conc (lena @| INil)) (x : et)
    norewrite
    preserves gpu ** on gpu_loc (b |-> Frac (fb *. fr) sb)
    returns r : et
    ensures pure (r == f x (acc sb (up i)))
  {
    elim_gpu (b |-> Frac (fb *. fr) sb);
    let y = tensor_read b i;
    let res = f x y;
    intro_gpu (b |-> Frac (fb *. fr) sb);
    res
  };
  rewrite on gpu_loc (b |-> Frac fb sb) as on gpu_loc (b |-> Frac (fb *. 1.0R) sb);
  launch_sync (TMap.kmap
    (CCons lena CNil)
    (fun f -> on gpu_loc (b |-> Frac (fb *. f) sb))
    #(shareable_tensor_pts_to #_ #_ #_ #lb b #sb fb)
    (fun (i : abs (lena @| INil)) (x r : et) -> r == f x (acc sb i))
    ff
    lena a #_ #_ #1.0R);
  rewrite on gpu_loc (b |-> Frac (fb *. 1.0R) sb) as on gpu_loc (b |-> Frac fb sb);

  with sa'. assert on gpu_loc (a |-> sa');
  assert pure (equal (chest1_map2 f sa sb) sa');

  ()
}
