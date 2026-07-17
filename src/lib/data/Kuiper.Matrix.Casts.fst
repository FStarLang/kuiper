module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Bijection

ghost
fn tensor_abij
  (#et : Type0)
  (#r1 : nat) (#s1 : shape r1)
  (#r2 : nat) (#s2 : shape r2)
  (b : abs s1 =~ abs s2)
  (#l : tlayout s1)
  (a : tensor et l)
  (#s : chest s1 et)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (layout_bij b l) |-> Frac f (chest_bij b s)
{
  let l' = layout_bij b l;
  let a' = from_array l' (core a);
    assert rewrites_to a' (from_array (layout_bij b l) (core a));

  tensor_ilower a;
  forevery_iso b _;
  forevery_map
    #(abs s2)
    (fun i ->
      pts_to_cell (core a) #f
        (l.imap.f (b.gg i))
        (acc s (b.gg i))
    )
    (fun i ->
      pts_to_cell (core a') #f
        (l'.imap.f i)
        (acc (chest_bij b s) i)
    )
    fn i {
      rewrite
        pts_to_cell (core a) #f
          (l.imap.f (b.gg i))
          (acc s (b.gg i))
      as
        pts_to_cell (core a') #f
          (l'.imap.f i)
          (acc (chest_bij b s) i);
      ()
    };
  tensor_iraise a';
}

ghost
fn t1_to_t2
  (#et : Type0)
  (#len : nat)
  (#l : layout1 len)
  (a : array1 et l)
  (#s : chest1 et len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (l1_to_l2 l) |-> Frac f (c1_to_c2 s)
{
  tensor_abij (bij12 len) a;
  rewrite each chest_bij (bij12 len) s as c1_to_c2 s;
}

ghost
fn t2_to_t1
  (#et : Type0)
  (#len : nat)
  (#l : layout2 1 len)
  (a : array2 et l)
  (#s : chest2 et 1 len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (l2_to_l1 l) |-> Frac f (c2_to_c1 s)
{
  tensor_abij (bij_sym (bij12 len)) a;
  rewrite each chest_bij (bij_sym (bij12 len)) s as c2_to_c1 s;
}
