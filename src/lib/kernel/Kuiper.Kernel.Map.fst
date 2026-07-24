module Kuiper.Kernel.Map

#lang-pulse

open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Shareable
module SZ = Kuiper.SizeT
module TMap = Kuiper.Kernel.TMap

let shape1 (lena : nat) : shape 1 = lena @| INil

unfold let index1 (#len : nat) (i : abs (shape1 len)) : natlt len =
  let (j, ()) = i in
  j

inline_for_extraction noextract unfold
let cindex1 (#len : erased nat) (i : conc (shape1 len)) : szlt len =
  let (j, ()) = i in
  j

let cindex1_up (#len : erased nat) (i : conc (shape1 len))
  : Lemma (cindex1 i == SZ.uint_to_t (index1 (up i)))
  = ()

instance triple_shareable
  (ra : perm -> slprop) {| shareable ra |}
  (rb : perm -> slprop) {| shareable rb |}
  (rc : perm -> slprop) {| shareable rc |}
  (fa fb fc : perm)
  : shareable (fun fr -> ra (fa *. fr) ** rb (fb *. fr) ** rc (fc *. fr))
  = double_shareable ra
      (fun fr -> rb (fb *. fr) ** rc (fc *. fr))
      fa 1.0R

inline_for_extraction noextract
fn map_gpu
  (#et : Type0)
  (f : et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#s : chest1 et lena)
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> chest_map f s)
{
  TMap.map_gpu (CCons lena CNil) f lena a;
}

inline_for_extraction noextract
fn ff_to
  (#it #ot : Type0) (#len : erased nat)
  (f : it -> ot)
  (#li : layout1 len) {| ctlayout li |}
  (input : array1 it li)
  (#si : chest1 it len) (#fr : perm)
  (i : conc (shape1 len)) (x : ot)
  norewrite
  preserves gpu ** (input |-> Frac fr si)
  returns r : ot
  ensures pure (r == f (acc si (up i)))
{
  let y = tensor_read input i;
  f y;
}

inline_for_extraction noextract
fn map_gpu_to
  (#it #ot : Type0)
  (f : it -> ot)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#li : layout1 lena) {| ctlayout li |}
  (#lo : layout1 lena) {| ctlayout lo |}
  (input : array1 it li { is_global input })
  (output : array1 ot lo { is_global output })
  (#si : chest1 it lena)
  (#so : chest1 ot lena)
  (#fi : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fi si)
  requires on gpu_loc (output |-> so)
  ensures  on gpu_loc (output |-> mk1 (fun i -> f (acc1 si i)))
{
  launch_sync (TMap.kmap
    (CCons lena CNil)
    (fun fr -> input |-> Frac fr si)
    #(tensor_pts_to_shareable input si)
    (fun i _ r -> r == f (acc si i))
    (ff_to f input)
    lena output #so #_ #fi);
  with so'. assert on gpu_loc (output |-> so');
  assert pure (equal so' (mk1 (fun i -> f (acc1 si i))));
}

inline_for_extraction noextract
fn ff2
  (#et : Type0) (#len : erased nat)
  (f : et -> et -> et)
  (#lb : layout1 len) {| ctlayout lb |}
  (b : array1 et lb)
  (#sb : chest1 et len) (#fr : perm)
  (i : conc (shape1 len)) (x : et)
  norewrite
  preserves gpu ** (b |-> Frac fr sb)
  returns r : et
  ensures pure (r == f x (acc sb (up i)))
{
  let y = tensor_read b i;
  f x y;
}

inline_for_extraction noextract
fn map_gpu2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (a : array1 et la { is_global a })
  (b : array1 et lb { is_global b })
  (#sa #sb : chest1 et lena)
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (a |-> sa)
  ensures  on gpu_loc (a |-> chest1_map2 f sa sb)
{
  launch_sync (TMap.kmap
    (CCons lena CNil)
    (fun fr -> b |-> Frac fr sb)
    #(tensor_pts_to_shareable b sb)
    (fun i x r -> r == f x (acc sb i))
    (ff2 f b)
    lena a #sa #_ #fb);
  with sa'. assert on gpu_loc (a |-> sa');
  assert pure (equal sa' (chest1_map2 f sa sb));
}

inline_for_extraction noextract
fn ff2_to
  (#at #bt #ot : Type0) (#len : erased nat)
  (f : at -> bt -> ot)
  (#la : layout1 len) (#lb : layout1 len) {| ctlayout la, ctlayout lb |}
  (a : array1 at la) (b : array1 bt lb)
  (#sa : chest1 at len) (#sb : chest1 bt len)
  (#fa #fb #fr : perm)
  (i : conc (shape1 len)) (x : ot)
  norewrite
  preserves gpu ** (a |-> Frac (fa *. fr) sa) ** (b |-> Frac (fb *. fr) sb)
  returns r : ot
  ensures pure (r == f (acc sa (up i)) (acc sb (up i)))
{
  let x = tensor_read a i;
  let y = tensor_read b i;
  f x y;
}

inline_for_extraction noextract
fn map_gpu2_to
  (#at #bt #ot : Type0)
  (f : at -> bt -> ot)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (#lo : layout1 lena) {| ctlayout lo |}
  (a : array1 at la { is_global a })
  (b : array1 bt lb { is_global b })
  (output : array1 ot lo { is_global output })
  (#sa : chest1 at lena)
  (#sb : chest1 bt lena)
  (#so : chest1 ot lena)
  (#fa #fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (output |-> so)
  ensures  on gpu_loc (output |-> mk1 (fun i -> f (acc1 sa i) (acc1 sb i)))
{
  launch_sync (TMap.kmap
    (CCons lena CNil)
    (fun fr -> (a |-> Frac (fa *. fr) sa) ** (b |-> Frac (fb *. fr) sb))
    #(double_shareable (fun fr -> a |-> Frac fr sa) (fun fr -> b |-> Frac fr sb) fa fb)
    (fun i _ r -> r == f (acc sa i) (acc sb i))
    (ff2_to f a b)
    lena output #so #_ #1.0R);
  with so'. assert on gpu_loc (output |-> so');
  assert pure (equal so' (mk1 (fun i -> f (acc1 sa i) (acc1 sb i))));
}

inline_for_extraction noextract
fn ff3
  (#et : Type0) (#len : erased nat)
  (f : et -> et -> et -> et)
  (#lb : layout1 len) (#lc : layout1 len) {| ctlayout lb, ctlayout lc |}
  (b : array1 et lb) (c : array1 et lc)
  (#sb #sc : chest1 et len) (#fb #fc #fr : perm)
  (i : conc (shape1 len)) (x : et)
  norewrite
  preserves gpu ** (b |-> Frac (fb *. fr) sb) ** (c |-> Frac (fc *. fr) sc)
  returns r : et
  ensures pure (r == f x (acc sb (up i)) (acc sc (up i)))
{
  let y = tensor_read b i;
  let z = tensor_read c i;
  f x y z;
}

inline_for_extraction noextract
fn map_gpu3
  (#et : Type0)
  (f : et -> et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (#lc : layout1 lena) {| ctlayout lc |}
  (a : array1 et la { is_global a })
  (b : array1 et lb { is_global b })
  (c : array1 et lc { is_global c })
  (#sa #sb #sc : chest1 et lena)
  (#fb #fc : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb) ** on gpu_loc (c |-> Frac fc sc)
  requires on gpu_loc (a |-> sa)
  ensures  on gpu_loc (a |-> chest1_map3 f sa sb sc)
{
  launch_sync (TMap.kmap
    (CCons lena CNil)
    (fun fr -> (b |-> Frac (fb *. fr) sb) ** (c |-> Frac (fc *. fr) sc))
    #(double_shareable (fun fr -> b |-> Frac fr sb) (fun fr -> c |-> Frac fr sc) fb fc)
    (fun i x r -> r == f x (acc sb i) (acc sc i))
    (ff3 f b c)
    lena a #sa #_ #1.0R);
  with sa'. assert on gpu_loc (a |-> sa');
  assert pure (equal sa' (chest1_map3 f sa sb sc));
}

inline_for_extraction noextract
fn ff3_to
  (#at #bt #ct #ot : Type0) (#len : erased nat)
  (f : at -> bt -> ct -> ot)
  (#la : layout1 len) (#lb : layout1 len) (#lc : layout1 len) {| ctlayout la, ctlayout lb, ctlayout lc |}
  (a : array1 at la) (b : array1 bt lb) (c : array1 ct lc)
  (#sa : chest1 at len) (#sb : chest1 bt len) (#sc : chest1 ct len)
  (#fa #fb #fc #fr : perm)
  (i : conc (shape1 len)) (x : ot)
  norewrite
  preserves gpu ** (a |-> Frac (fa *. fr) sa) ** (b |-> Frac (fb *. fr) sb) ** (c |-> Frac (fc *. fr) sc)
  returns r : ot
  ensures pure (r == f (acc sa (up i)) (acc sb (up i)) (acc sc (up i)))
{
  let x = tensor_read a i;
  let y = tensor_read b i;
  let z = tensor_read c i;
  f x y z;
}

inline_for_extraction noextract
fn map_gpu3_to
  (#at #bt #ct #ot : Type0)
  (f : at -> bt -> ct -> ot)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (#lc : layout1 lena) {| ctlayout lc |}
  (#lo : layout1 lena) {| ctlayout lo |}
  (a : array1 at la { is_global a })
  (b : array1 bt lb { is_global b })
  (c : array1 ct lc { is_global c })
  (output : array1 ot lo { is_global output })
  (#sa : chest1 at lena)
  (#sb : chest1 bt lena)
  (#sc : chest1 ct lena)
  (#so : chest1 ot lena)
  (#fa #fb #fc : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb) ** on gpu_loc (c |-> Frac fc sc)
  requires on gpu_loc (output |-> so)
  ensures  on gpu_loc (output |-> mk1 (fun i -> f (acc1 sa i) (acc1 sb i) (acc1 sc i)))
{
  launch_sync (TMap.kmap
    (CCons lena CNil)
    (fun fr -> (a |-> Frac (fa *. fr) sa) ** ((b |-> Frac (fb *. fr) sb) ** (c |-> Frac (fc *. fr) sc)))
    #(triple_shareable
        (fun fr -> a |-> Frac fr sa)
        (fun fr -> b |-> Frac fr sb)
        (fun fr -> c |-> Frac fr sc)
        fa fb fc)
    (fun i _ r -> r == f (acc sa i) (acc sb i) (acc sc i))
    (ff3_to f a b c)
    lena output #so #_ #1.0R);
  with so'. assert on gpu_loc (output |-> so');
  assert pure (equal so' (mk1 (fun i -> f (acc1 sa i) (acc1 sb i) (acc1 sc i))));
}

inline_for_extraction noextract
fn map_host
  (#et : Type0) {| sized et |}
  (f : et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (a : Pulse.Lib.Vec.lvec et lena)
  (#s : erased (lseq et lena))
  preserves cpu
  requires a |-> s
  ensures  a |-> lseq_map f s
{
  let ga = alloc0 #et lena (l1_forward lena);
  with em. assert on gpu_loc (ga |-> em);
  map_loc gpu_loc #(ga |-> em) #(core ga |-> to_seq (l1_forward lena) em)
    fn _ { tensor_concr ga; };
  gpu_memcpy_host_to_device (core ga) a lena;
  map_loc gpu_loc #(core ga |-> reveal s) #(ga |-> from_seq (l1_forward lena) s)
    fn _ {
      tensor_abs' (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> from_seq (l1_forward lena) s)
           as (ga |-> from_seq (l1_forward lena) s);
    };
  map_gpu f lena ga;
  with res. assert on gpu_loc (ga |-> res);
  map_loc gpu_loc #(ga |-> res) #(core ga |-> to_seq (l1_forward lena) res)
    fn _ { tensor_concr ga; };
  gpu_memcpy_device_to_host a (core ga) lena;
  map_loc gpu_loc #(core ga |-> to_seq (l1_forward lena) res) #(ga |-> res)
    fn _ {
      tensor_abs (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> reveal res)
           as (ga |-> res);
    };
  free ga;
  assert pure (Seq.equal (to_seq (l1_forward lena) res) (lseq_map f s));
}

inline_for_extraction noextract
fn ff_mapi
  (#et : Type0) (#len : erased nat { SZ.fits len })
  (f : et -> (i : SZ.t { SZ.v i < len }) -> et)
  (#fr : perm) (i : conc (shape1 len)) (x : et)
  norewrite
  preserves gpu ** emp
  returns r : et
  ensures pure (r == mapi_value f x (index1 (up i)))
{
  cindex1_up i;
  f x (cindex1 i);
}

inline_for_extraction noextract
fn mapi_gpu
  (#et : Type0)
  (lena : szp { lena <= max_blocks * max_threads /\ lena <= 2147483648 /\ lena > 0 /\ SZ.fits lena })
  (f : et -> (i : SZ.t { SZ.v i < lena }) -> et)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#s : chest1 et lena)
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> map_chest1i f s)
{
  launch_sync (TMap.kmap
    (CCons lena CNil)
    (fun _ -> emp)
    (fun i x r -> r == mapi_value f x (index1 i))
    (ff_mapi f)
    lena a #s #_ #1.0R);
  with s'. assert on gpu_loc (a |-> s');
  assert pure (equal s' (map_chest1i f s));
}
