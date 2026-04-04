module Kuiper.Example.Array4

#lang-pulse
open Kuiper
open Kuiper.Array4
module Array4 = Kuiper.Array4
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.TensorLayout
open Kuiper.Index
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let layout (d0 d1 d2 d3 : nat) : layout d0 d1 d2 d3 =
  pack <|
  g_grouped_by 0 d0 <|
  g_grouped_by 0 d1 <|
  g_grouped_by 0 d2 <|
  g_grouped_by 0 d3 <|
  lunit

inline_for_extraction noextract
class auto_cinj (#n : erased nat) (#d : erased (idesc n)) (#k : erased nat)
  (f : abs d @~> natlt k) =
  {
    ff : (x:conc d -> y:SZ.t{SZ.v y == f.f (up x)});
  }

inline_for_extraction noextract
class csizeof (#n : erased nat) (d : (idesc n)) =
  {
    v : (v : SZ.t{SZ.v v == sizeof d});
  }

inline_for_extraction noextract
instance csizeof_INil : csizeof INil = {
  v = 1sz;
}

inline_for_extraction noextract
instance csizeof_ICons
  (#n : erased nat)
  (d0 : SZ.t)
  (d1 : idesc n)
  (c_d1 : csizeof d1)
  (#_ : squash (SZ.fits (d0 * c_d1.v)))
  : csizeof (ICons d0 d1) =
  {
    v = SZ.mul d0 c_d1.v;
  }

inline_for_extraction noextract
instance csizeof_insert
  (#n : pos)
  (i : natlt n)
  (d0 : SZ.t)
  (#d1 : idesc (n-1))
  (c_d1 : csizeof d1{SZ.fits (d0 * c_d1.v)})
  : csizeof #n (insert_i #(n-1) i d0 d1) =
  {
    v = SZ.mul d0 c_d1.v;
  }

inline_for_extraction noextract
instance cunit : auto_cinj lunit = {
  ff = (fun _ -> 0sz);
}

inline_for_extraction noextract
let cff (#a #b : _) (c : cbij a b) : (a -> b) =
  match c with | { cff } -> cff

inline_for_extraction noextract
let c_grouped_by_f
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : idesc n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (c_sub : auto_cinj sub)
  (idx:conc (insert_i i k d))
  : szlt (sizeof (insert_i i k d))
  = modulo_insert i k d;
    let maj, min = c_bring_forward_ff i (insert_i i k d) idx in
    let { ff } = c_sub in
    let sub_i : szlt (sizeof d) = ff min in
    assume (SZ.fits (maj * cs.v)); // Where will this come from?
    let offset : SZ.t = maj *^ cs.v in
    assume (SZ.fits (offset + sub_i)); // idem
    let r = offset +^ sub_i in
    r

let lem_c_grouped_by_f_ok
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : idesc n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (c_sub : auto_cinj sub)
  (idx:conc (insert_i i k d))
  : Lemma (SZ.v (c_grouped_by_f i k c_sub idx) == g_grouped_by_f i k sub (up idx))
          [SMTPat (c_grouped_by_f i k c_sub idx)]
  = let amaj, amin = (abs_bring_forward_bij i (insert_i i k d)).ff (up idx) in
    let maj, min = c_bring_forward_ff i (insert_i i k d) idx in
    assume (SZ.fits (maj * cs.v)); // Where will this come from?
    lemma_c_bring_forward_ff_ok i (insert_i i k d) idx;
    bring_forward_commute2 i (insert_i i k d) maj min;
    assert (SZ.v maj == amaj);
    // assume (up min == amin);
    let asub_i : natlt (sizeof d) = sub.f amin in
    let { ff } = c_sub in
    let sub_i = ff min in
    let aoffset = amaj * sizeof d in
    assume (SZ.fits (aoffset + sub_i)); // Where will this come from?
    let offset = maj *^ cs.v in
    ()

inline_for_extraction noextract
instance c_grouped_by
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : idesc n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (c_sub : auto_cinj sub)
  : auto_cinj (g_grouped_by i k sub) =
  {
    ff = (fun idx -> c_grouped_by_f i k c_sub idx);
  }

inline_for_extraction noextract
instance c_grouped_by_concrete_i_0'
  (#n: erased nat{n > 0})
  (k : erased nat)
  (#d : idesc (n-1))
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (c_sub : auto_cinj sub)
  : auto_cinj #n (g_grouped_by #(n-1) 0 k sub) =
  c_grouped_by 0sz k c_sub

inline_for_extraction noextract
instance close (#n : erased nat) (#d: idesc n) (f : layout_f_for d) (c_f : auto_cinj f)
  (#_ : squash (all_fit d))
  : ctlayout (pack f) =
  {
    culen   = magic(); // should fix this, or remove this field altogether
    all_fit = ();
    cimap   = c_f.ff;
  }

// VERY brittle postprocessing to make sure we get a 1st-order function. Would
// not be needed if strict_on_arguments worked properly on recursive functions
// (it seems not to).
[@@Tac.(postprocess_with (fun () ->
           norm [iota; delta; zeta_full; zeta; primops];
           trefl ()))]
inline_for_extraction noextract
instance blah
  (d0 : SZ.t{SZ.fits d0})
  (d1 : SZ.t{SZ.fits d1})
  (d2 : SZ.t{SZ.fits d2})
  (d3 : SZ.t{SZ.fits d3})
  (#_ : squash (SZ.fits (d0 * d1 * d2 * d3) /\ SZ.fits (d1 * d2 * d3) /\ SZ.fits (d2 * d3)))
  : ctlayout (layout d0 d1 d2 d3)
  =
  close _ <|
  c_grouped_by 0sz _ <| // need this one to get things going...
  solve_debug

fn test0 (m : array4 u32 (layout 3 5 4 2))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : ctlayout (layout 10 10 10 10) = blah 10sz 10sz 10sz 10sz

fn test1 (m : array4 u32 (layout 10 10 10 10))
  preserves m |-> 's
  returns u32
{
  let v = Array4.(m.(1sz, 2sz, 3sz, 4sz));
  v
}

fn test2 (m : array4 u32 (layout 10 10 10 10))
  requires m |-> 's
  ensures  m |-> Kuiper.EMatrix4.mupd 's 1 2 3 4 42ul
{
  Array4.(m.(1sz, 2sz, 3sz, 4sz) <- 42ul);
  ()
}
